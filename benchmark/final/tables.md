## Table 1: injected hazards prevented (closed-loop validator, logged policy plans)

| policy        | sensors | prevented         | 95% CI (scenes resampled) | new collisions, followers react 1.0 s | new collisions, followers react 1.5 s | new collisions, recorded followers |
|---------------|---------|-------------------|---------------------------|---------------------------------------|---------------------------------------|------------------------------------|
| all three     | gt      | 639 / 814 (78.5%) | 66.6-90.4%                | 173 (148 from behind)                 | 214 (199 from behind)                 | 356 (347 from behind)              |
| all three     | radar   | 313 / 814 (38.5%) | 29.5-47.2%                | 68 (50 from behind)                   | 95 (91 from behind)                   | 165 (165 from behind)              |
| all three     | lidar   | 452 / 814 (55.5%) | 42.2-69.2%                | 71 (59 from behind)                   | 119 (114 from behind)                 | 225 (224 from behind)              |
| all three     | fused   | 507 / 814 (62.3%) | 48.5-76.6%                | 86 (66 from behind)                   | 121 (108 from behind)                 | 242 (240 from behind)              |
| Alpamayo 1.5  | gt      | 400 / 494 (81.0%) | 72.2-89.8%                | 101 (89 from behind)                  | 117 (114 from behind)                 | 180 (179 from behind)              |
| Alpamayo 1.5  | radar   | 165 / 494 (33.4%) | 26.9-40.2%                | 38 (26 from behind)                   | 44 (42 from behind)                   | 70 (70 from behind)                |
| Alpamayo 1.5  | lidar   | 301 / 494 (60.9%) | 50.9-70.9%                | 49 (41 from behind)                   | 70 (68 from behind)                   | 126 (125 from behind)              |
| Alpamayo 1.5  | fused   | 323 / 494 (65.4%) | 54.9-75.8%                | 51 (39 from behind)                   | 67 (62 from behind)                   | 131 (129 from behind)              |
| SimForge-D1.5 | gt      | 128 / 177 (72.3%) | 49.7-92.7%                | 22 (16 from behind)                   | 43 (38 from behind)                   | 73 (72 from behind)                |
| SimForge-D1.5 | radar   | 65 / 177 (36.7%)  | 22.6-51.3%                | 12 (8 from behind)                    | 25 (23 from behind)                   | 35 (35 from behind)                |
| SimForge-D1.5 | lidar   | 98 / 177 (55.4%)  | 30.4-78.0%                | 4 (3 from behind)                     | 19 (18 from behind)                   | 46 (46 from behind)                |
| SimForge-D1.5 | fused   | 102 / 177 (57.6%) | 30.5-81.4%                | 10 (3 from behind)                    | 23 (16 from behind)                   | 48 (48 from behind)                |
| VaVAM         | gt      | 111 / 143 (77.6%) | 58.1-94.2%                | 50 (43 from behind)                   | 54 (47 from behind)                   | 103 (96 from behind)               |
| VaVAM         | radar   | 83 / 143 (58.0%)  | 41.0-74.2%                | 18 (16 from behind)                   | 26 (26 from behind)                   | 60 (60 from behind)                |
| VaVAM         | lidar   | 53 / 143 (37.1%)  | 18.0-57.0%                | 18 (15 from behind)                   | 30 (28 from behind)                   | 53 (53 from behind)                |
| VaVAM         | fused   | 82 / 143 (57.3%)  | 37.8-77.6%                | 25 (24 from behind)                   | 31 (30 from behind)                   | 63 (63 from behind)                |

## Crossing-prediction ablation

| policy       | sensors | hazard       | without crossing  | with crossing     |
|--------------|---------|--------------|-------------------|-------------------|
| all three    | gt      | braking lead | 74 / 111 (66.7%)  | 75 / 111 (67.6%)  |
| all three    | gt      | cut-in       | 209 / 313 (66.8%) | 296 / 313 (94.6%) |
| all three    | gt      | pedestrian   | 178 / 390 (45.6%) | 268 / 390 (68.7%) |
| all three    | radar   | braking lead | 57 / 111 (51.4%)  | 57 / 111 (51.4%)  |
| all three    | radar   | cut-in       | 162 / 313 (51.8%) | 162 / 313 (51.8%) |
| all three    | radar   | pedestrian   | 94 / 390 (24.1%)  | 94 / 390 (24.1%)  |
| all three    | lidar   | braking lead | 69 / 111 (62.2%)  | 67 / 111 (60.4%)  |
| all three    | lidar   | cut-in       | 179 / 313 (57.2%) | 205 / 313 (65.5%) |
| all three    | lidar   | pedestrian   | 105 / 390 (26.9%) | 180 / 390 (46.2%) |
| all three    | fused   | braking lead | 70 / 111 (63.1%)  | 65 / 111 (58.6%)  |
| all three    | fused   | cut-in       | 210 / 313 (67.1%) | 241 / 313 (77.0%) |
| all three    | fused   | pedestrian   | 129 / 390 (33.1%) | 201 / 390 (51.5%) |
| Alpamayo 1.5 | gt      | braking lead | 50 / 66 (75.8%)   | 50 / 66 (75.8%)   |
| Alpamayo 1.5 | gt      | cut-in       | 112 / 182 (61.5%) | 170 / 182 (93.4%) |
| Alpamayo 1.5 | gt      | pedestrian   | 106 / 246 (43.1%) | 180 / 246 (73.2%) |
| Alpamayo 1.5 | radar   | braking lead | 39 / 66 (59.1%)   | 39 / 66 (59.1%)   |
| Alpamayo 1.5 | radar   | cut-in       | 83 / 182 (45.6%)  | 83 / 182 (45.6%)  |
| Alpamayo 1.5 | radar   | pedestrian   | 43 / 246 (17.5%)  | 43 / 246 (17.5%)  |
| Alpamayo 1.5 | lidar   | braking lead | 51 / 66 (77.3%)   | 49 / 66 (74.2%)   |
| Alpamayo 1.5 | lidar   | cut-in       | 121 / 182 (66.5%) | 136 / 182 (74.7%) |
| Alpamayo 1.5 | lidar   | pedestrian   | 59 / 246 (24.0%)  | 116 / 246 (47.2%) |
| Alpamayo 1.5 | fused   | braking lead | 52 / 66 (78.8%)   | 47 / 66 (71.2%)   |
| Alpamayo 1.5 | fused   | cut-in       | 134 / 182 (73.6%) | 149 / 182 (81.9%) |
| Alpamayo 1.5 | fused   | pedestrian   | 74 / 246 (30.1%)  | 127 / 246 (51.6%) |

## Per hazard variant (fused)

| policy       | hazard            | cases | prevented        |
|--------------|-------------------|-------|------------------|
| Alpamayo 1.5 | cut_in_10m        | 62    | 54 / 58 (93.1%)  |
| Alpamayo 1.5 | cut_in_10m_brake  | 62    | 44 / 62 (71.0%)  |
| Alpamayo 1.5 | cut_in_5m         | 62    | 51 / 62 (82.3%)  |
| Alpamayo 1.5 | lead_brake_4      | 35    | 19 / 22 (86.4%)  |
| Alpamayo 1.5 | lead_brake_6      | 35    | 14 / 22 (63.6%)  |
| Alpamayo 1.5 | lead_brake_8      | 35    | 14 / 22 (63.6%)  |
| Alpamayo 1.5 | ped_run           | 62    | 23 / 61 (37.7%)  |
| Alpamayo 1.5 | ped_run_occluded  | 62    | 30 / 61 (49.2%)  |
| Alpamayo 1.5 | ped_walk          | 62    | 35 / 62 (56.5%)  |
| Alpamayo 1.5 | ped_walk_occluded | 62    | 39 / 62 (62.9%)  |
| all three    | cut_in_10m        | 110   | 89 / 99 (89.9%)  |
| all three    | cut_in_10m_brake  | 110   | 72 / 110 (65.5%) |
| all three    | cut_in_5m         | 110   | 80 / 104 (76.9%) |
| all three    | lead_brake_4      | 64    | 25 / 37 (67.6%)  |
| all three    | lead_brake_6      | 64    | 20 / 37 (54.1%)  |
| all three    | lead_brake_8      | 64    | 20 / 37 (54.1%)  |
| all three    | ped_run           | 110   | 35 / 88 (39.8%)  |
| all three    | ped_run_occluded  | 110   | 39 / 88 (44.3%)  |
| all three    | ped_walk          | 110   | 62 / 107 (57.9%) |
| all three    | ped_walk_occluded | 110   | 65 / 107 (60.7%) |

## Baselines (fused)

| method           | policy       | prevented         | braking lead     | cut-in            | pedestrian        | new collisions (followers 1.0 s) | steps acted on (normal driving) | rollouts with an emergency stop |
|------------------|--------------|-------------------|------------------|-------------------|-------------------|----------------------------------|---------------------------------|---------------------------------|
| Ours             | all three    | 507 / 814 (62.3%) | 65 / 111 (58.6%) | 241 / 313 (77.0%) | 201 / 390 (51.5%) | 86 (66 from behind)              | 3.3%                            | 19 / 105 (18.1%)                |
| Ours             | Alpamayo 1.5 | 323 / 494 (65.4%) | 47 / 66 (71.2%)  | 149 / 182 (81.9%) | 127 / 246 (51.6%) | 51 (39 from behind)              | 3.3%                            | 19 / 105 (18.1%)                |
| TTC-only braking | all three    | 164 / 814 (20.1%) | 18 / 111 (16.2%) | 113 / 313 (36.1%) | 33 / 390 (8.5%)   | 48 (40 from behind)              | 0.1%                            | 2 / 105 (1.9%)                  |
| TTC-only braking | Alpamayo 1.5 | 90 / 494 (18.2%)  | 9 / 66 (13.6%)   | 74 / 182 (40.7%)  | 7 / 246 (2.8%)    | 21 (15 from behind)              | 0.1%                            | 2 / 105 (1.9%)                  |
| RSS              | all three    | 469 / 814 (57.6%) | 63 / 111 (56.8%) | 227 / 313 (72.5%) | 179 / 390 (45.9%) | 111 (89 from behind)             | 11.5%                           | 0 / 105 (0.0%)                  |
| RSS              | Alpamayo 1.5 | 313 / 494 (63.4%) | 54 / 66 (81.8%)  | 150 / 182 (82.4%) | 109 / 246 (44.3%) | 67 (53 from behind)              | 11.5%                           | 0 / 105 (0.0%)                  |

## Sensitivity sweep (Alpamayo 1.5, fused)

| setting                                     | prevented         | braking lead    | cut-in            | pedestrian        | new collisions (followers 1.0 s) | steps acted on (normal driving) | rollouts with an emergency stop |
|---------------------------------------------|-------------------|-----------------|-------------------|-------------------|----------------------------------|---------------------------------|---------------------------------|
| defaults (beta 2, horizon 3 s, max decel 6) | 323 / 494 (65.4%) | 47 / 66 (71.2%) | 149 / 182 (81.9%) | 127 / 246 (51.6%) | 51 (39 from behind)              | 3.3%                            | 19 / 105 (18.1%)                |
| beta 0                                      | 386 / 494 (78.1%) | 52 / 66 (78.8%) | 162 / 182 (89.0%) | 172 / 246 (69.9%) | 91 (76 from behind)              | 3.9%                            | 35 / 105 (33.3%)                |
| beta 1                                      | 349 / 494 (70.6%) | 47 / 66 (71.2%) | 150 / 182 (82.4%) | 152 / 246 (61.8%) | 60 (48 from behind)              | 3.5%                            | 23 / 105 (21.9%)                |
| crossing horizon 1 s                        | 311 / 494 (63.0%) | 47 / 66 (71.2%) | 148 / 182 (81.3%) | 116 / 246 (47.2%) | 56 (43 from behind)              | 2.7%                            | 19 / 105 (18.1%)                |
| crossing horizon 2 s                        | 323 / 494 (65.4%) | 47 / 66 (71.2%) | 149 / 182 (81.9%) | 127 / 246 (51.6%) | 53 (40 from behind)              | 3.2%                            | 19 / 105 (18.1%)                |
| crossing horizon 4 s                        | 324 / 494 (65.6%) | 47 / 66 (71.2%) | 150 / 182 (82.4%) | 127 / 246 (51.6%) | 54 (39 from behind)              | 3.9%                            | 19 / 105 (18.1%)                |
| max decel 4 m/s^2                           | 223 / 494 (45.1%) | 38 / 66 (57.6%) | 110 / 182 (60.4%) | 75 / 246 (30.5%)  | 30 (19 from behind)              | 3.7%                            | 19 / 105 (18.1%)                |
| max decel 8 m/s^2                           | 373 / 494 (75.5%) | 53 / 66 (80.3%) | 159 / 182 (87.4%) | 161 / 246 (65.4%) | 80 (64 from behind)              | 3.2%                            | 19 / 105 (18.1%)                |

## Selection tiers: 1522 steps where the model's own plan failed a check (Alpamayo 1.5, normal driving and crashes, fused)

| tier                                       | steps | % of flagged steps |
|--------------------------------------------|-------|--------------------|
| 1: model plan kept (awaiting confirmation) | 563   | 37.0               |
| 2: switched to a safe candidate            | 68    | 4.5                |
| 3: slowed, on another candidate            | 23    | 1.5                |
| 3: slowed, on the model's candidate        | 868   | 57.0               |

## N = 1 vs N = 6 candidates, Alpamayo 1.5 rollouts, fused

| candidates   | interventions (slowdown steps) | speed reduction per intervention, median (m/s) | route lost per rollout without a crash, mean (m) | at-fault crashes avoided (followers 1.0 s) | new crashes (followers 1.0 s) |
|--------------|--------------------------------|------------------------------------------------|--------------------------------------------------|--------------------------------------------|-------------------------------|
| N = 6 (ours) | 891                            | 1.84                                           | 3.18                                             | 1 / 1 (100.0%)                             | 0                             |
| N = 1        | 926                            | 1.73                                           | 3.23                                             | 1 / 1 (100.0%)                             | 0                             |

## McNemar, ours vs. RSS

814 paired cases; ours only: 122, RSS only: 84, both: 385, neither: 223; exact two-sided p = 0.0098


## First constraint to fire, prevented cases, fused, all policies

| family     | first               | cases |
|------------|---------------------|-------|
| cut_in     | STOPPING_DISTANCE   | 184   |
| cut_in     | EMERGENCY_TTC       | 56    |
| cut_in     | REAR_COLLISION_RISK | 1     |
| lead_brake | STOPPING_DISTANCE   | 64    |
| lead_brake | EMERGENCY_TTC       | 1     |
| ped        | STOPPING_DISTANCE   | 173   |
| ped        | EMERGENCY_TTC       | 28    |

## Commanded deceleration, fused

| cases                             | slowdown steps | median (m/s^2) | 95th pct (m/s^2) | > 4 m/s^2 |
|-----------------------------------|----------------|----------------|------------------|-----------|
| prevented hazards                 | 30571          | 4.75           | 6.0              | 51.3%     |
| missed hazards                    | 4050           | 6.0            | 6.0              | 61.8%     |
| normal driving (Alpamayo replays) | 891            | 3.0            | 6.0              | 32.7%     |

## Impact speed of missed hazards, ego relative to hazard, fused

| hazard       | misses | baseline median (m/s) | with validator median (m/s) | reduced |
|--------------|--------|-----------------------|-----------------------------|---------|
| all          | 307    | 11.3                  | 6.7                         | 84%     |
| braking lead | 46     | 14.0                  | 3.7                         | 54%     |
| cut-in       | 72     | 6.8                   | 3.5                         | 76%     |
| pedestrian   | 189    | 12.9                  | 8.1                         | 94%     |

## Occluded vs. open pedestrians, fused

| hazard            | cases | prevented | first slowdown after onset, median (s) | slowed before onset (share) |
|-------------------|-------|-----------|----------------------------------------|-----------------------------|
| ped_run           | 110   | 35        | 0.18                                   | 0.224                       |
| ped_run_occluded  | 110   | 39        | 0.48                                   | 0.827                       |
| ped_walk          | 110   | 62        | 0.71                                   | 0.038                       |
| ped_walk_occluded | 110   | 65        | 0.51                                   | 0.075                       |

## Braking lead, with vs. without crossing prediction, fused: 5 lost, 0 gained

| scene    | hazard       | model        | with  | with_first              | with_t   | without | without_first               | without_t |
|----------|--------------|--------------|-------|-------------------------|----------|---------|-----------------------------|-----------|
| 0499fb41 | lead_brake_6 | Alpamayo 1.5 | false | EMERGENCY_TTC_VIOLATION | 8.841853 | true    | STOPPING_DISTANCE_VIOLATION | 10.241853 |
| 0499fb41 | lead_brake_8 | Alpamayo 1.5 | false | EMERGENCY_TTC_VIOLATION | 8.841853 | true    | STOPPING_DISTANCE_VIOLATION | 10.041853 |
| 0499fb41 | lead_brake_4 | Alpamayo 1.5 | false | EMERGENCY_TTC_VIOLATION | 7.941853 | true    | STOPPING_DISTANCE_VIOLATION | 9.141853  |
| 0499fb41 | lead_brake_6 | Alpamayo 1.5 | false | EMERGENCY_TTC_VIOLATION | 7.941853 | true    | STOPPING_DISTANCE_VIOLATION | 8.741853  |
| 0499fb41 | lead_brake_8 | Alpamayo 1.5 | false | EMERGENCY_TTC_VIOLATION | 7.941853 | true    | STOPPING_DISTANCE_VIOLATION | 8.541853  |

