import math
from typing import Dict, Any, Tuple


"""
Convert an object's world-frame position into ego-frame coordinates
"""
def world_to_ego(ego_state, obj_x, obj_y):
    ego_x, ego_y, heading = ego_state["x"], ego_state["y"], ego_state["heading"]
    dx = obj_x - ego_x
    dy = obj_y - ego_y
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    longitudinal = dx * cos_h + dy * sin_h
    lateral = -dx * sin_h + dy * cos_h
    return longitudinal, lateral


"""
Project obstacle velocity onto ego vehicle's forward axis to get obstacle's forward speed wrt world frame
"""
def obstacle_forward_speed(ego_state, obstacle):
    heading = ego_state["heading"]
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    vx = obstacle.get("vx", 0.0)
    vy = obstacle.get("vy", 0.0)
    return vx * cos_h + vy * sin_h


"""
Obstacle velocity component across the ego's heading (positive to the ego's left)
"""
def obstacle_lateral_speed(ego_state, obstacle):
    heading = ego_state["heading"]
    vx = obstacle.get("vx", 0.0)
    vy = obstacle.get("vy", 0.0)
    return -vx * math.sin(heading) + vy * math.cos(heading)


"""
Relative closing speed along ego forward direction (positive means ego is nearing the obstacle)
"""
def relative_forward_speed(ego_speed, obstacle_speed):
    return ego_speed - obstacle_speed


"""
Closing speed of an obstacle behind ego (positive means it is catching up with ego)
"""
def rear_closing_speed(ego_speed, obstacle_speed):
    return obstacle_speed - ego_speed


"""
Compute time to collision (TTC) in seconds.
"""
def compute_time_to_collision(distance, relative_speed):
    if relative_speed <= 1e-6:
        return float("inf")
    return distance / relative_speed


"""
Compute stopping distance
stopping_distance = reaction_dist + braking_dist
                  = v * t_reaction + v^2 / (2a)
"""
def compute_stopping_distance(speed, reaction_time=0.5, max_decel=6.0):
    if max_decel <= 0:
        raise ValueError("max_decel must be positive")
    speed = max(speed, 0.0)
    reaction_distance = speed * reaction_time
    braking_distance = (speed ** 2) / (2.0 * max_decel)
    return reaction_distance + braking_distance


"""
Compute the deceleration magnitude required to stop within the given distance after accounting for reaction distance
"""
def compute_required_deceleration(speed, distance, reaction_time=0.5):
    speed = max(speed, 0.0)
    remaining_distance = distance - speed * reaction_time
    if remaining_distance <= 0:
        return float("inf")
    if speed <= 1e-6:
        return 0.0
    return (speed ** 2) / (2.0 * remaining_distance)


"""
Distance a vehicle ahead still covers if it brakes to a stop at max_decel. A vehicle not moving forward adds none.
"""
def compute_front_stopping_distance(front_speed, max_decel=6.0):
    if max_decel <= 0:
        raise ValueError("max_decel must be positive")
    return max(front_speed, 0.0) ** 2 / (2.0 * max_decel)


"""
Maximum ego speed v whose RSS longitudinal safe distance (Shalev-Shwartz et al., 2017)
to a vehicle ahead driving at front_speed fits within gap:
    v * rho + accel * rho^2 / 2 + (v + rho * accel)^2 / (2 * min_brake) - front_speed^2 / (2 * front_max_brake) <= gap
Substituting u = v + rho * accel gives u^2 / (2 * min_brake) + rho * u - (gap + accel * rho^2 / 2 + front_term) <= 0.
"""
def compute_rss_max_safe_speed(gap, front_speed, response_time, max_accel, min_brake, front_max_brake):
    if min_brake <= 0 or front_max_brake <= 0:
        raise ValueError("min_brake and front_max_brake must be positive")
    front_speed = max(front_speed, 0.0)
    budget = gap + 0.5 * max_accel * response_time ** 2 + front_speed ** 2 / (2.0 * front_max_brake)
    if budget <= 0:
        return 0.0
    u = min_brake * (-response_time + math.sqrt(response_time ** 2 + 2.0 * budget / min_brake))
    return max(u - response_time * max_accel, 0.0)


    """
    Lightweight geometric filter for whether an obstacle is relevant to the ego path.
        longitudinal: obstacle distance ahead in ego frame
        lateral: obstacle left/right offset in ego frame
        lane_half_width: half-width of the path corridor in meters
        prediction_horizon: max forward distance to consider in meters
    """
def is_obstacle_in_path(longitudinal, lateral, lane_half_width=1.75, prediction_horizon=60.0):
    return ((longitudinal > 0.0) and (longitudinal <= prediction_horizon) and (abs(lateral) <= lane_half_width))