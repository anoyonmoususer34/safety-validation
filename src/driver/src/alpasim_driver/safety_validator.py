import numpy as np
from alpasim_driver.physics import (
    world_to_ego,
    obstacle_forward_speed,
    obstacle_lateral_speed,
    relative_forward_speed,
    rear_closing_speed,
    compute_time_to_collision,
    compute_stopping_distance,
    compute_required_deceleration,
    compute_front_stopping_distance,
    compute_rss_max_safe_speed,
)


'''
- lateral_margin_m:
  an obstacle is in the path's corridor when its side is within this of the ego's side
- prediction_horizon:
  how far ahead to reason about obstacles
- reaction_time:
  delay before braking begins
- max_decel:
  magnitude of max feasible deceleration
- min_ttc / emergency_ttc:
  TTC thresholds for nominal slowdown vs emergency brake
- safety_buffer:
  extra margin beyond exact stopping distance
- min_confidence:
  min obstacle confidence to even consider
- rear_min_ttc:
  TTC threshold (from behind) for rear-collision speed-floor enforcement
- rear_emergency_ttc:
  TTC threshold (from behind) for immediate rear-collision action
- rear_prediction_horizon:
  how far behind (m) to look for following obstacles
- enable_emergency / enable_stopping_distance / enable_low_ttc:
  switch the emergency-TTC brake, stopping-distance slowdown and low-TTC speed cap
  on or off independently
- ego_front_m / ego_rear_m:
  rig origin to the ego's front / rear bumper; distances to obstacles are bumper gaps
- ego_half_width_m:
  an obstacle alongside the ego (no bumper gap) only counts if it overlaps this width
- crossing_horizon_s / crossing_min_lateral_speed:
  an obstacle outside the corridor also counts if, moving sideways toward the path at
  least crossing_min_lateral_speed, it will be in the corridor while the ego passes it,
  and the ego reaches it within crossing_horizon_s (0 switches this off)

A proposed action is {"target_speed", "steering_angle", "path_xy"}: the speed the plan
drives, its heading change (for traces) and its (T, 2) rig-frame waypoints.
'''
_HOLD_STOP_MAX_STEPS = 50   # ~5 s at 10 Hz before auto-clearing the latch
_HOLD_STOP_CLEAR_MARGIN = 8.0  # m — obstacle must reappear this far past held distance to clear latch
_HOLD_STOP_CONFIRM_STEPS = 2  # consecutive brake-to-zero steps before arming: one false alarm must not hold the ego
_HOLD_STOP_MAX_EGO_SPEED = 0.5  # m/s: the latch holds a stop, so it arms only once the ego has stopped
# An obstacle alongside (gap 0) counts only when moving at least this fast: braking cannot
# avoid a parked car or wall beside the ego, and LiDAR, seeing only its near side, places
# it too close.  A moving one may be cutting in.
_ALONGSIDE_MIN_SPEED = 2.0  # m/s


class SafetyValidator:
    def __init__(
        self,
        lateral_margin_m=0.5,
        prediction_horizon=60.0,
        reaction_time=0.5,
        max_decel=6.0,
        min_ttc=2.0,
        emergency_ttc=1.0,
        safety_buffer=2.0,
        min_confidence=0.1,
        confidence_weight_power=2.0,
        rear_min_ttc=2.0,
        rear_emergency_ttc=0.8,
        rear_prediction_horizon=30.0,
        enable_emergency=True,
        enable_stopping_distance=True,
        enable_low_ttc=True,
        ego_front_m=4.093,
        ego_rear_m=1.3,
        ego_half_width_m=1.05,
        crossing_horizon_s=3.0,
        crossing_min_lateral_speed=0.5,
    ):
        self.lateral_margin_m = lateral_margin_m
        self.prediction_horizon = prediction_horizon
        self.reaction_time = reaction_time
        self.max_decel = max_decel
        self.min_ttc = min_ttc
        self.emergency_ttc = emergency_ttc
        self.safety_buffer = safety_buffer
        self.min_confidence = min_confidence
        self.confidence_weight_power = confidence_weight_power
        self.rear_min_ttc = rear_min_ttc
        self.rear_emergency_ttc = rear_emergency_ttc
        self.rear_prediction_horizon = rear_prediction_horizon
        self.enable_emergency = enable_emergency
        self.enable_stopping_distance = enable_stopping_distance
        self.enable_low_ttc = enable_low_ttc
        self.ego_front_m = ego_front_m
        self.ego_rear_m = ego_rear_m
        self.ego_half_width_m = ego_half_width_m
        self.crossing_horizon_s = crossing_horizon_s
        self.crossing_min_lateral_speed = crossing_min_lateral_speed
        # Hold-stop latch: once ego is braked to 0 for a front obstacle, remember the
        # distance and keep the ego stopped even when LiDAR temporarily loses the obstacle
        # (e.g. sensor min-range cut-off when ego is very close). Clears when the obstacle
        # reappears clearly further away or after _HOLD_STOP_MAX_STEPS consecutive steps.
        self._hold_stop_distance: float | None = None
        self._hold_stop_steps_remaining: int = 0
        self._brake_to_zero_steps: int = 0

    '''
    Validate the proposed action against physics-based safety constraints given the current ego state and list of perceived obstacles
    '''
    def validate(self, ego_state, proposed_action, obstacles):
        ego_speed = max(ego_state.get("speed", 0.0), 0.0)
        previous_brake_to_zero_steps = self._brake_to_zero_steps
        self._brake_to_zero_steps = 0
        validated_action = {
            "target_speed": max(proposed_action.get("target_speed", ego_speed), 0.0),
            "steering_angle": proposed_action.get("steering_angle", 0.0),
            "brake": False,
            "override_reason": None,
        }
        violations, trace_lines = [], []
        trace_data = {
            "inputs": {
                "ego_state": ego_state,
                "proposed_action": proposed_action,
                "num_obstacles_received": len(obstacles),
            },
            "relevant_obstacles": [],
            "per_obstacle_assessment": [],
            "summary_metrics": {},
            "actions_taken": [],
        }
        trace_lines.append(
            "Safety validator invoked with proposed action "
            f"(target_speed={validated_action['target_speed']:.2f} m/s, "
            f"steering_angle={validated_action['steering_angle']:.3f} rad)."
        )
        relevant_obstacles = self._collect_relevant_obstacles(ego_state, proposed_action, obstacles)

        # Hold-stop latch: if we previously braked to 0 for an obstacle and now LiDAR has
        # lost it (sensor min-range dead zone when ego is very close), inject a ghost so the
        # ego stays stopped rather than driving through.
        # The latch is armed only by the brake-to-zero branches below.
        # The latch is for an obstacle ahead: one alongside is passed, not lost.
        obstacles_ahead = [obs for obs in relevant_obstacles if not obs["alongside"]]
        if obstacles_ahead and self._hold_stop_distance is not None:
            closest = min(obs["longitudinal_distance"] for obs in obstacles_ahead)
            if closest > self._hold_stop_distance + _HOLD_STOP_CLEAR_MARGIN:
                # Obstacle reappeared clearly further away — it moved, path is free.
                self._hold_stop_distance = None
                self._hold_stop_steps_remaining = 0
            else:
                self._hold_stop_distance = closest
                self._hold_stop_steps_remaining = _HOLD_STOP_MAX_STEPS
        elif self._hold_stop_distance is not None:
            self._hold_stop_steps_remaining -= 1
            if self._hold_stop_steps_remaining > 0:
                ghost = {
                    "id": "hold_stop_ghost",
                    "source": "hold_stop",
                    "confidence": 1.0,
                    "longitudinal_distance": self._hold_stop_distance,
                    "lateral_distance": 0.0,
                    "path_lateral_error": 0.0,
                    "forward_speed": 0.0,
                    "risk_weight": 1.0,
                    "alongside": False,
                    "crossing": False,
                    "x": ego_state.get("x", 0.0),
                    "y": ego_state.get("y", 0.0),
                }
                relevant_obstacles = [ghost]
            else:
                self._hold_stop_distance = None

        trace_lines.append(
            f"Received {len(obstacles)} obstacles; "
            f"{len(relevant_obstacles)} considered relevant to predicted path."
            + (f" [hold-stop latch active at {self._hold_stop_distance:.1f} m]"
               if self._hold_stop_distance is not None else "")
        )

        trace_data["relevant_obstacles"] = [
            {
                "id": obs.get("id"),
                "source": obs.get("source", "unknown"),
                "confidence": obs.get("confidence", 1.0),
                "longitudinal_distance": obs["longitudinal_distance"],
                "lateral_distance": obs["lateral_distance"],
                "path_lateral_error": obs["path_lateral_error"],
                "forward_speed": obs["forward_speed"],
                "risk_weight": obs["risk_weight"],
                "crossing": obs["crossing"],
            }
            for obs in relevant_obstacles
        ]

        if not relevant_obstacles:
            trace_lines.append("No relevant obstacles detected along predicted path. Action accepted.")
            trace_data["summary_metrics"] = {
                "worst_ttc": float("inf"),
                "max_required_decel": 0.0,
                "min_safe_speed": validated_action["target_speed"],
            }
            result = self._build_result(
                        is_safe=True,
                        proposed_action=proposed_action,
                        validated_action=validated_action,
                        violations=violations,
                        trace_lines=trace_lines,
                        trace_data=trace_data,
                    )
            return result

        assessments = []
        worst_ttc = float("inf")
        max_required_decel = 0.0
        min_safe_speed = validated_action["target_speed"]
        emergency_triggered = False
        for obs in relevant_obstacles:
            distance = obs["longitudinal_distance"]
            obstacle_speed = obs["forward_speed"]
            confidence = obs.get("confidence", 1.0)
            risk_weight = obs["risk_weight"]
            closing_speed = max(relative_forward_speed(validated_action["target_speed"], obstacle_speed), 0.0)
            ttc = compute_time_to_collision(distance, closing_speed)
            stopping_distance = compute_stopping_distance(
                validated_action["target_speed"],
                reaction_time=self.reaction_time,
                max_decel=self.max_decel,
            )
            # The vehicle ahead is assumed to brake as hard as the ego can, so it
            # adds its own stopping distance to the room the ego has.
            front_stopping_distance = compute_front_stopping_distance(obstacle_speed, self.max_decel)
            required_decel = compute_required_deceleration(
                validated_action["target_speed"],
                distance - self.safety_buffer + front_stopping_distance,
                reaction_time=self.reaction_time,
            )
            safe_speed = compute_rss_max_safe_speed(
                gap=distance - self.safety_buffer,
                front_speed=obstacle_speed,
                response_time=self.reaction_time,
                max_accel=0.0,
                min_brake=self.max_decel,
                front_max_brake=self.max_decel,
            )
            confidence_adjusted_ttc = ttc / max(risk_weight, 1e-6)
            assessment = {
                "id": obs.get("id"),
                "source": obs.get("source", "unknown"),
                "confidence": confidence,
                "risk_weight": risk_weight,
                "distance": distance,
                "path_lateral_error": obs["path_lateral_error"],
                "forward_speed": obstacle_speed,
                "closing_speed": closing_speed,
                "ttc": ttc,
                "confidence_adjusted_ttc": confidence_adjusted_ttc,
                "stopping_distance": stopping_distance,
                "required_decel": required_decel,
                "safe_speed": safe_speed,
                "flags": [],
            }

            if confidence_adjusted_ttc < worst_ttc:
                worst_ttc = confidence_adjusted_ttc
            if required_decel > max_required_decel:
                max_required_decel = required_decel
            if safe_speed < min_safe_speed:
                min_safe_speed = safe_speed
            # A car alongside cannot be avoided by stopping hard, and a hard stop invites a
            # rear-end; it gets the gentler slowdowns below, which let a merging car in.
            # The filter marks detections too unreliable to stop hard for (radar-only
            # ones when LiDAR is running); those get the slowdowns only.
            if (self.enable_emergency and not obs["alongside"] and obs.get("may_emergency_brake", True)
                    and confidence >= self.min_confidence and confidence_adjusted_ttc < self.emergency_ttc):
                assessment["flags"].append("EMERGENCY_TTC_VIOLATION")
                emergency_triggered = True
            if self.enable_stopping_distance and confidence >= self.min_confidence and stopping_distance + self.safety_buffer > distance + front_stopping_distance:
                assessment["flags"].append("STOPPING_DISTANCE_VIOLATION")
            if self.enable_low_ttc and confidence >= self.min_confidence and confidence_adjusted_ttc < self.min_ttc:
                assessment["flags"].append("LOW_TTC_VIOLATION")

            assessments.append(assessment)
        trace_data["per_obstacle_assessment"] = assessments
        trace_data["summary_metrics"] = {
            "worst_confidence_adjusted_ttc": worst_ttc,
            "max_required_decel": max_required_decel,
            "min_safe_speed_across_obstacles": min_safe_speed,
        }

        # Sort logged obstacle assessments by severity for easier debugging/demo purposes
        assessments_sorted = sorted(
            assessments,
            key=lambda a: (
                a["confidence_adjusted_ttc"],
                a["distance"],
            )
        )
        for a in assessments_sorted:
            trace_lines.append(
                "Obstacle assessment: "
                f"id={a['id']}, source={a['source']}, confidence={a['confidence']:.2f}, "
                f"distance={a['distance']:.2f} m, "
                f"path_error={a['path_lateral_error']:.2f} m, "
                f"ttc={a['ttc']:.2f} s, "
                f"adj_ttc={a['confidence_adjusted_ttc']:.2f} s, "
                f"required_decel={a['required_decel']:.2f} m/s^2, "
                f"safe_speed={a['safe_speed']:.2f} m/s, "
                f"flags={a['flags']}."
            )

        # Emergency brake if any sufficiently confident obstacle looks immediately dangerous!
        if emergency_triggered:
            violations.append("EMERGENCY_TTC_VIOLATION")
            validated_action["target_speed"] = 0.0
            validated_action["brake"] = True
            validated_action["override_reason"] = "emergency_ttc"
            self._arm_hold_stop(relevant_obstacles, previous_brake_to_zero_steps, ego_speed)
            trace_lines.append(
                "At least one high-confidence obstacle triggered the emergency TTC rule. "
                "Emergency braking override issued."
            )
            trace_data["actions_taken"].append({
                "action": "emergency_brake",
                "reason": "emergency_ttc",
            })
            result = self._build_result(
                        is_safe=False,
                        proposed_action=proposed_action,
                        validated_action=validated_action,
                        violations=violations,
                        trace_lines=trace_lines,
                        trace_data=trace_data,
                    )
            return result

        # Slowdown based on the most restrictive safe speed across all relevant obstacles
        if self.enable_stopping_distance and min_safe_speed < validated_action["target_speed"]:
            violations.append("STOPPING_DISTANCE_VIOLATION")
            old_speed = validated_action["target_speed"]
            validated_action["target_speed"] = min_safe_speed
            validated_action["override_reason"] = "stopping_distance"
            trace_lines.append(
                f"Aggregated stopping-distance analysis reduced target speed from "
                f"{old_speed:.2f} to {validated_action['target_speed']:.2f} m/s."
            )
            trace_data["actions_taken"].append({
                "action": "speed_reduction",
                "reason": "stopping_distance",
                "old_speed": old_speed,
                "new_speed": validated_action["target_speed"],
            })

        # if any confident obstacle still yields low TTC, avoid outputting a speed above the slowest relevant obstacle's forward speed.
        low_ttc_obstacles = [
            a for a in assessments
            if "LOW_TTC_VIOLATION" in a["flags"]
        ]

        if low_ttc_obstacles:
            violations.append("LOW_TTC_VIOLATION")
            slowest_relevant_speed = min(
                max(a["forward_speed"], 0.0) for a in low_ttc_obstacles
            )
            old_speed = validated_action["target_speed"]
            validated_action["target_speed"] = min(
                validated_action["target_speed"],
                slowest_relevant_speed,
            )
            if validated_action["target_speed"] <= 1e-6:
                validated_action["brake"] = True
                self._arm_hold_stop(relevant_obstacles, previous_brake_to_zero_steps, ego_speed)
            if validated_action["override_reason"] is None:
                validated_action["override_reason"] = "low_ttc"

            trace_lines.append(
                f"Low-TTC condition detected across one or more obstacles. "
                f"Target speed capped from {old_speed:.2f} to "
                f"{validated_action['target_speed']:.2f} m/s."
            )
            trace_data["actions_taken"].append({
                "action": "speed_cap",
                "reason": "low_ttc",
                "old_speed": old_speed,
                "new_speed": validated_action["target_speed"],
            })

        # ── Rear collision check (off when rear_prediction_horizon is 0) ─────────
        rear_obstacles = self._collect_rear_obstacles(ego_state, obstacles)
        if rear_obstacles:
            rear_speed_floor = 0.0
            rear_emergency = False
            rear_assessments = []

            for obs in rear_obstacles:
                gap = obs["gap_behind"]
                obs_speed = obs["forward_speed"]
                confidence = obs.get("confidence", 1.0)
                risk_weight = obs["risk_weight"]

                # TTC computed against the validated (potentially already reduced) speed
                closing = rear_closing_speed(validated_action["target_speed"], obs_speed)
                ttc_rear = compute_time_to_collision(gap, closing)
                adj_ttc_rear = ttc_rear / max(risk_weight, 1e-6)

                assessment = {
                    "id": obs.get("id"),
                    "source": obs.get("source", "unknown"),
                    "confidence": confidence,
                    "gap_behind": gap,
                    "obstacle_speed": obs_speed,
                    "closing_speed": closing,
                    "ttc_rear": ttc_rear,
                    "adj_ttc_rear": adj_ttc_rear,
                    "flags": [],
                }

                if confidence >= self.min_confidence and adj_ttc_rear < self.rear_emergency_ttc:
                    assessment["flags"].append("REAR_EMERGENCY_TTC")
                    rear_emergency = True
                    # Floor: at minimum match the obstacle's speed so gap stops closing
                    rear_speed_floor = max(rear_speed_floor, obs_speed)
                elif confidence >= self.min_confidence and adj_ttc_rear < self.rear_min_ttc:
                    assessment["flags"].append("REAR_LOW_TTC")
                    # Soft floor: the checks above must not slow the plan further;
                    # the plan's own braking is left alone.
                    rear_speed_floor = max(rear_speed_floor, proposed_action["target_speed"])

                trace_lines.append(
                    f"Rear obstacle: id={obs.get('id')}, source={obs.get('source','unknown')}, "
                    f"confidence={confidence:.2f}, gap={gap:.2f} m, "
                    f"obs_speed={obs_speed:.2f} m/s, closing={closing:.2f} m/s, "
                    f"ttc_rear={ttc_rear:.2f} s, adj_ttc={adj_ttc_rear:.2f} s, "
                    f"flags={assessment['flags']}."
                )
                rear_assessments.append(assessment)

            trace_data["rear_obstacle_assessment"] = rear_assessments

            if rear_speed_floor > validated_action["target_speed"]:
                violations.append("REAR_COLLISION_RISK")
                old_speed = validated_action["target_speed"]

                if "EMERGENCY_TTC_VIOLATION" in violations:
                    # Front emergency brake takes absolute priority — flag but don't override.
                    trace_lines.append(
                        f"Rear collision risk detected (floor={rear_speed_floor:.2f} m/s) but "
                        "front emergency brake takes priority — cannot prevent rear collision."
                    )
                else:
                    validated_action["target_speed"] = rear_speed_floor
                    if validated_action.get("brake"):
                        validated_action["brake"] = False
                    validated_action["override_reason"] = "rear_collision_risk"
                    trace_lines.append(
                        f"Rear collision risk: speed floor raised from {old_speed:.2f} to "
                        f"{rear_speed_floor:.2f} m/s to maintain gap to following obstacle."
                    )
                    trace_data["actions_taken"].append({
                        "action": "rear_speed_floor",
                        "reason": "rear_collision_risk",
                        "old_speed": old_speed,
                        "new_speed": validated_action["target_speed"],
                        "emergency": rear_emergency,
                    })

        is_safe = len(violations) == 0

        if is_safe:
            trace_lines.append("All physics-based safety checks passed. Proposed action validated.")
        else:
            trace_lines.append(
                "Proposed action was modified based on aggregated physics-based safety constraints."
            )
        result = self._build_result(
                    is_safe=is_safe,
                    proposed_action=proposed_action,
                    validated_action=validated_action,
                    violations=self._deduplicate_preserve_order(violations),
                    trace_lines=trace_lines,
                    trace_data=trace_data,
                ) 
        return result

    '''
    Arm the latch so the ego stays stopped if the sensors lose the obstacle at close range.
    It arms only once the ego has stopped, for a real detection in the ego's own path (not
    just the corridor's margin), on consecutive steps: the latch's own ghost must not extend
    the hold, and a detection seen once, or one the ego would pass, may be a false alarm.
    '''
    def _arm_hold_stop(self, relevant_obstacles, previous_brake_to_zero_steps, ego_speed):
        real_obstacles = [
            obs for obs in relevant_obstacles
            if obs.get("source") != "hold_stop"
            and not obs["alongside"]
            and abs(obs["path_lateral_error"]) <= self.ego_half_width_m + obs.get("half_width_m", 0.0)
        ]
        if not real_obstacles or ego_speed > _HOLD_STOP_MAX_EGO_SPEED:
            return
        self._brake_to_zero_steps = previous_brake_to_zero_steps + 1
        if self._brake_to_zero_steps >= _HOLD_STOP_CONFIRM_STEPS:
            self._hold_stop_distance = min(obs["longitudinal_distance"] for obs in real_obstacles)
            self._hold_stop_steps_remaining = _HOLD_STOP_MAX_STEPS

    '''
    RSS baseline: the highest speed that keeps the RSS longitudinal safe distance to every
    obstacle in the proposed path's corridor.  Confidence only gates detections through
    min_confidence; it does not weight risk.  The gap is measured like the other checks:
    obstacle distance minus safety_buffer.
    '''
    def rss_safe_speed(self, ego_state, proposed_action, obstacles, max_accel, min_brake):
        # RSS's longitudinal rule covers obstacles already in the corridor only.
        relevant = [
            obs for obs in self._collect_relevant_obstacles(ego_state, proposed_action, obstacles)
            if not obs["crossing"]
        ]
        assessments = []
        safe_speed = float("inf")
        for obs in relevant:
            speed = compute_rss_max_safe_speed(
                gap=obs["longitudinal_distance"] - self.safety_buffer,
                front_speed=obs["forward_speed"],
                response_time=self.reaction_time,
                max_accel=max_accel,
                min_brake=min_brake,
                front_max_brake=self.max_decel,
            )
            assessments.append({
                "id": obs.get("id"),
                "source": obs.get("source", "unknown"),
                "distance": obs["longitudinal_distance"],
                "forward_speed": obs["forward_speed"],
                "rss_safe_speed": speed,
            })
            safe_speed = min(safe_speed, speed)
        return safe_speed, assessments

    '''
    Convert the obstacle list into a list of path-relevant obstacles (judged against the predicted curved path)
    '''
    def _collect_relevant_obstacles(self, ego_state, proposed_action, obstacles):
        relevant = []

        for obs in obstacles:
            confidence = obs.get("confidence", 1.0)
            if confidence < self.min_confidence:
                continue

            longitudinal, lateral = world_to_ego(
                ego_state,
                obs["x"],
                obs["y"],
            )
            if longitudinal <= 0 or longitudinal > self.prediction_horizon:
                continue

            predicted_path_lateral = self._path_lateral_offset(longitudinal, proposed_action["path_xy"])
            path_lateral_error = lateral - predicted_path_lateral
            gap = longitudinal - self.ego_front_m - obs.get("half_length_m", 0.0)
            fwd_speed = obstacle_forward_speed(ego_state, obs)
            crossing = False
            if not self._in_corridor(path_lateral_error, obs):
                crossing = self._crosses_into_corridor(
                    path_lateral_error, obs, gap, ego_state, fwd_speed
                )
                if not crossing:
                    continue
            if gap <= 0.0 and not self._counts_alongside(path_lateral_error, obs, fwd_speed):
                continue

            risk_weight = self._confidence_risk_weight(confidence)
            relevant.append({
                **obs,
                "longitudinal_distance": max(gap, 0.0),
                "alongside": gap <= 0.0,
                "lateral_distance": lateral,
                "predicted_path_lateral": predicted_path_lateral,
                "path_lateral_error": path_lateral_error,
                "forward_speed": fwd_speed,
                "risk_weight": risk_weight,
                "crossing": crossing,
            })
        return relevant

    '''
    Collect obstacles that are behind the ego and within the rear prediction corridor.
    We do not filter by approach speed here — that is done in validate() against
    the post-front-check validated speed, which may be lower than the current ego speed.
    '''
    def _collect_rear_obstacles(self, ego_state, obstacles):
        rear_relevant = []
        for obs in obstacles:
            confidence = obs.get("confidence", 1.0)
            if confidence < self.min_confidence:
                continue
            longitudinal, lateral = world_to_ego(ego_state, obs["x"], obs["y"])
            # Behind ego and within the rear horizon
            if longitudinal >= 0 or abs(longitudinal) > self.rear_prediction_horizon:
                continue
            if not self._in_corridor(lateral, obs):
                continue
            gap = abs(longitudinal) - self.ego_rear_m - obs.get("half_length_m", 0.0)
            fwd_speed = obstacle_forward_speed(ego_state, obs)
            if gap <= 0.0 and not self._counts_alongside(lateral, obs, fwd_speed):
                continue
            risk_weight = self._confidence_risk_weight(confidence)
            rear_relevant.append({
                **obs,
                "gap_behind": max(gap, 0.0),
                "lateral_distance": lateral,
                "forward_speed": fwd_speed,
                "risk_weight": risk_weight,
            })
        return rear_relevant

    '''
    Whether an obstacle at this lateral offset from the ego's path comes within lateral_margin_m of the ego's width
    '''
    def _in_corridor(self, lateral_offset, obs):
        return abs(lateral_offset) <= self.ego_half_width_m + obs.get("half_width_m", 0.0) + self.lateral_margin_m

    '''
    Whether an obstacle outside the corridor, moving sideways toward the path, will be in
    the corridor at some point while the ego passes it: from when the ego's front reaches
    it until the ego's rear has passed it, at the current closing speed
    '''
    def _crosses_into_corridor(self, path_lateral_error, obs, gap, ego_state, forward_speed):
        if self.crossing_horizon_s <= 0.0 or gap <= 0.0:
            return False
        lateral_speed = obstacle_lateral_speed(ego_state, obs)
        if abs(lateral_speed) < self.crossing_min_lateral_speed:
            return False
        if lateral_speed * path_lateral_error >= 0.0:  # not moving toward the path
            return False
        closing = max(ego_state.get("speed", 0.0), 0.0) - forward_speed
        if closing <= 0.0:
            return False
        arrive = gap / closing
        if arrive > self.crossing_horizon_s:
            return False
        passed = (gap + self.ego_front_m + self.ego_rear_m + 2.0 * obs.get("half_length_m", 0.0)) / closing
        first = path_lateral_error + lateral_speed * arrive
        last = path_lateral_error + lateral_speed * passed
        half_width = self.ego_half_width_m + obs.get("half_width_m", 0.0) + self.lateral_margin_m
        return min(first, last) <= half_width and max(first, last) >= -half_width

    '''
    Whether an obstacle alongside the ego, at this lateral offset from its path, counts: it
    overlaps the ego's width and is moving (see _ALONGSIDE_MIN_SPEED)
    '''
    def _counts_alongside(self, lateral_offset, obs, forward_speed):
        overlaps = abs(lateral_offset) <= self.ego_half_width_m + obs.get("half_width_m", 0.0)
        return overlaps and abs(forward_speed) >= _ALONGSIDE_MIN_SPEED

    '''
    The plan's lateral offset at a given longitudinal distance; past either end of the plan, the offset at that end
    '''
    def _path_lateral_offset(self, longitudinal_distance, path_xy):
        forward = np.maximum.accumulate(path_xy[:, 0])
        return float(np.interp(longitudinal_distance, forward, path_xy[:, 1]))
    
    '''
    Convert raw confidence into a risk weight
    Higher confidence -> larger risk weight -> smaller adjusted TTC -> more conservative behavior
    '''
    def _confidence_risk_weight(self, confidence):
        confidence = max(0.0, min(confidence, 1.0))
        return max(confidence ** self.confidence_weight_power, 1e-3)

    # Convert steering angle into a human-friendly phrase.
    def _describe_steering_action(self, steering_angle):
        abs_angle = abs(steering_angle)
        if abs_angle < 0.02:
            direction = "continue straight"
        elif steering_angle > 0:
            if abs_angle < 0.08:
                direction = "nudge left"
            elif abs_angle < 0.18:
                direction = "turn left slightly"
            else:
                direction = "turn left"
        else:
            if abs_angle < 0.08:
                direction = "nudge right"
            elif abs_angle < 0.18:
                direction = "turn right slightly"
            else:
                direction = "turn right"
        return direction


    # Generate a short, presentation-friendly decision trace.
    def _generate_human_readable_trace(self, proposed_action, validated_action, violations):
        proposed_speed = proposed_action.get("target_speed", 0.0)
        proposed_steering = proposed_action.get("steering_angle", 0.0)
        validated_speed = validated_action.get("target_speed", 0.0)
        validated_steering = validated_action.get("steering_angle", 0.0)
        brake = validated_action.get("brake", False)
        proposed_direction = self._describe_steering_action(proposed_steering)
        validated_direction = self._describe_steering_action(validated_steering)

        if brake:
            return (
                f"Alpamayo indicated the vehicle should {proposed_direction} "
                f"at {proposed_speed:.1f} m/s. "
                f"Our safety enforcer overrode this action with emergency braking "
                f"due to an immediate collision risk."
            )
        if not violations:
            return (
                f"Alpamayo indicated the vehicle should {proposed_direction} "
                f"at {proposed_speed:.1f} m/s. "
                f"Our safety enforcer agreed that this action is physically safe."
            )
        if "REAR_COLLISION_RISK" in violations and "EMERGENCY_TTC_VIOLATION" not in violations:
            front_also = any(
                v in violations
                for v in ("STOPPING_DISTANCE_VIOLATION", "LOW_TTC_VIOLATION")
            )
            if front_also:
                return (
                    f"Alpamayo indicated the vehicle should {proposed_direction} "
                    f"at {proposed_speed:.1f} m/s. "
                    f"Our safety enforcer detected both a front obstacle and a fast-following "
                    f"vehicle from behind. Speed was held at {validated_speed:.1f} m/s to "
                    f"balance both risks."
                )
            return (
                f"Alpamayo indicated the vehicle should {proposed_direction} "
                f"at {proposed_speed:.1f} m/s. "
                f"Our safety enforcer raised the minimum speed to {validated_speed:.1f} m/s "
                f"to prevent a rear collision with a fast-following vehicle."
            )
        if abs(validated_speed - proposed_speed) > 1e-6 and proposed_direction == validated_direction:
            if "STOPPING_DISTANCE_VIOLATION" in violations:
                return (
                    f"Alpamayo indicated the vehicle should {proposed_direction} "
                    f"at {proposed_speed:.1f} m/s. "
                    f"Our safety enforcer reduced the speed to {validated_speed:.1f} m/s "
                    f"to satisfy stopping-distance constraints."
                )
            if "LOW_TTC_VIOLATION" in violations:
                return (
                    f"Alpamayo indicated the vehicle should {proposed_direction} "
                    f"at {proposed_speed:.1f} m/s. "
                    f"Our safety enforcer reduced the speed to {validated_speed:.1f} m/s "
                    f"because the predicted time-to-collision was too low."
                )
        return (
            f"Alpamayo indicated the vehicle should {proposed_direction} "
            f"at {proposed_speed:.1f} m/s. "
            f"Our safety enforcer modified the action to {validated_direction} "
            f"at {validated_speed:.1f} m/s based on physics-based safety checks."
        )

    '''
    Build final returned object
    '''
    def _build_result(self, is_safe, proposed_action, validated_action, violations, trace_lines, trace_data):
        return {
            "is_safe": is_safe,
            "validated_action": validated_action,
            "violations": violations,
            "decision_trace": "\n".join(trace_lines),
            "human_readable_trace": self._generate_human_readable_trace(
                proposed_action,
                validated_action,
                violations,
            ),
            "trace_data": trace_data,
        }

    '''
    Remove duplicate violations but still preserve order
    '''
    def _deduplicate_preserve_order(self, items):
        seen = set()
        result = []
        for item in items:
            if (item not in seen):
                seen.add(item)
                result.append(item)
        return result