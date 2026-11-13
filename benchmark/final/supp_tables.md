## Sec. 2: rollouts and applicability

| model | rollouts | scenes |
|---|---|---|
| Alpamayo 1.5 | 66 | 50 |
| SimForge-D1.5 | 24 | 8 |
| VaVAM | 24 | 8 |

1140 injections, 962 applicable; not applicable by hazard:

| model | hazard | len |
|---|---|---|
| Alpamayo 1.5 | cut_in_10m | 4 |
| Alpamayo 1.5 | cut_in_10m_brake | 4 |
| Alpamayo 1.5 | cut_in_5m | 4 |
| Alpamayo 1.5 | lead_brake_4 | 31 |
| Alpamayo 1.5 | lead_brake_6 | 31 |
| Alpamayo 1.5 | lead_brake_8 | 31 |
| Alpamayo 1.5 | ped_run | 4 |
| Alpamayo 1.5 | ped_run_occluded | 4 |
| Alpamayo 1.5 | ped_walk | 4 |
| Alpamayo 1.5 | ped_walk_occluded | 4 |
| SimForge-D1.5 | lead_brake_4 | 16 |
| SimForge-D1.5 | lead_brake_6 | 16 |
| SimForge-D1.5 | lead_brake_8 | 16 |
| VaVAM | lead_brake_4 | 3 |
| VaVAM | lead_brake_6 | 3 |
| VaVAM | lead_brake_8 | 3 |

## Table 1: prevention by hazard (%, all policies pooled; n = baseline collisions)

| hazard | n | gt | radar | lidar | fused |
|---|---|---|---|---|---|
| Lead brakes, 4 m/s^2 | 37 | 70 | 54 | 70 | 68 |
| Lead brakes, 6 m/s^2 | 37 | 65 | 51 | 57 | 54 |
| Lead brakes, 8 m/s^2 | 37 | 68 | 49 | 54 | 54 |
| Cut-in, 10 m | 99 | 100 | 76 | 80 | 90 |
| Cut-in, 5 m | 104 | 100 | 9 | 69 | 77 |
| Cut-in, 10 m, then brakes | 110 | 85 | 71 | 49 | 65 |
| Pedestrian walks | 107 | 73 | 39 | 50 | 58 |
|   occluded by van | 107 | 73 | 47 | 57 | 61 |
| Pedestrian runs | 88 | 64 | 1 | 41 | 40 |
|   occluded by van | 88 | 64 | 1 | 33 | 44 |

## Tables 2-4: Alpamayo 1.5 (injections, baseline collisions, prevented per sensing condition)

| hazard | Inj. | Base | gt | radar | lidar | fused |
|---|---|---|---|---|---|---|
| Lead brakes, 4 m/s^2 | 35 | 22 | 18 | 14 | 20 | 19 |
| Lead brakes, 6 m/s^2 | 35 | 22 | 16 | 13 | 15 | 14 |
| Lead brakes, 8 m/s^2 | 35 | 22 | 16 | 12 | 14 | 14 |
| Cut-in, 10 m | 62 | 58 | 58 | 44 | 49 | 54 |
| Cut-in, 5 m | 62 | 62 | 62 | 1 | 49 | 51 |
| Cut-in, 10 m, then brakes | 62 | 62 | 50 | 38 | 38 | 44 |
| Pedestrian walks | 62 | 62 | 48 | 17 | 33 | 35 |
|   occluded by van | 62 | 62 | 48 | 24 | 36 | 39 |
| Pedestrian runs | 62 | 61 | 42 | 1 | 23 | 23 |
|   occluded by van | 62 | 61 | 42 | 1 | 24 | 30 |
| Total | 539 | 494 | 400 | 165 | 301 | 323 |
| New front/side, recorded followers |  |  | 1 | 0 | 1 | 2 |
| New rear-end, recorded followers |  |  | 179 | 70 | 125 | 129 |
| New front/side, followers react 1.0 s |  |  | 12 | 12 | 8 | 12 |
| New rear-end, followers react 1.0 s |  |  | 89 | 26 | 41 | 39 |

## Tables 2-4: SimForge-D1.5 (injections, baseline collisions, prevented per sensing condition)

| hazard | Inj. | Base | gt | radar | lidar | fused |
|---|---|---|---|---|---|---|
| Lead brakes, 4 m/s^2 | 8 | 3 | 3 | 3 | 3 | 3 |
| Lead brakes, 6 m/s^2 | 8 | 3 | 3 | 3 | 3 | 3 |
| Lead brakes, 8 m/s^2 | 8 | 3 | 3 | 3 | 3 | 3 |
| Cut-in, 10 m | 24 | 24 | 24 | 20 | 19 | 19 |
| Cut-in, 5 m | 24 | 24 | 24 | 0 | 17 | 17 |
| Cut-in, 10 m, then brakes | 24 | 24 | 19 | 16 | 11 | 12 |
| Pedestrian walks | 24 | 24 | 12 | 9 | 12 | 12 |
|   occluded by van | 24 | 24 | 12 | 11 | 12 | 12 |
| Pedestrian runs | 24 | 24 | 14 | 0 | 13 | 12 |
|   occluded by van | 24 | 24 | 14 | 0 | 5 | 9 |
| Total | 192 | 177 | 128 | 65 | 98 | 102 |
| New front/side, recorded followers |  |  | 1 | 0 | 0 | 0 |
| New rear-end, recorded followers |  |  | 72 | 35 | 46 | 48 |
| New front/side, followers react 1.0 s |  |  | 6 | 4 | 1 | 7 |
| New rear-end, followers react 1.0 s |  |  | 16 | 8 | 3 | 3 |

## Tables 2-4: VaVAM (injections, baseline collisions, prevented per sensing condition)

| hazard | Inj. | Base | gt | radar | lidar | fused |
|---|---|---|---|---|---|---|
| Lead brakes, 4 m/s^2 | 21 | 12 | 5 | 3 | 3 | 3 |
| Lead brakes, 6 m/s^2 | 21 | 12 | 5 | 3 | 3 | 3 |
| Lead brakes, 8 m/s^2 | 21 | 12 | 6 | 3 | 3 | 3 |
| Cut-in, 10 m | 24 | 17 | 17 | 11 | 11 | 16 |
| Cut-in, 5 m | 24 | 18 | 18 | 8 | 6 | 12 |
| Cut-in, 10 m, then brakes | 24 | 24 | 24 | 24 | 5 | 16 |
| Pedestrian walks | 24 | 21 | 18 | 16 | 9 | 15 |
|   occluded by van | 24 | 21 | 18 | 15 | 13 | 14 |
| Pedestrian runs | 24 | 3 | 0 | 0 | 0 | 0 |
|   occluded by van | 24 | 3 | 0 | 0 | 0 | 0 |
| Total | 231 | 143 | 111 | 83 | 53 | 82 |
| New front/side, recorded followers |  |  | 7 | 0 | 0 | 0 |
| New rear-end, recorded followers |  |  | 96 | 60 | 53 | 63 |
| New front/side, followers react 1.0 s |  |  | 7 | 2 | 3 | 1 |
| New rear-end, followers react 1.0 s |  |  | 43 | 16 | 15 | 24 |

## Table 5: prevented (%) by ego speed at hazard onset (all policies; n = baseline collisions)

| speed | n | gt | radar | lidar | fused |
|---|---|---|---|---|---|
| < 8 | 179 | 95.0 | 45.3 | 62.6 | 76.0 |
| 8-12 | 335 | 87.5 | 45.7 | 65.7 | 71.0 |
| 12-16 | 167 | 69.5 | 33.5 | 45.5 | 50.9 |
| > 16 | 133 | 45.1 | 17.3 | 33.1 | 36.1 |

## Table 8: outcomes over the 814 baseline collisions, recorded followers (upper bound)

| outcome | gt | radar | lidar | fused |
|---|---|---|---|---|
| Hazard prevented | 639 | 313 | 452 | 507 |
| Clean | 316 | 169 | 252 | 286 |
| New front/side | 4 | 0 | 1 | 2 |
| New rear-end | 324 | 150 | 208 | 225 |

## Table 8: outcomes over the 814 baseline collisions, followers react after 1.0 s

| outcome | gt | radar | lidar | fused |
|---|---|---|---|---|
| Hazard prevented | 639 | 313 | 452 | 507 |
| Clean | 494 | 256 | 394 | 438 |
| New front/side | 20 | 18 | 9 | 19 |
| New rear-end | 131 | 44 | 53 | 54 |

## Sec. 5: applicable injections where the baseline does not collide (fused)

- cases: 148; validator intervenes: 106; new rear-end (recorded followers): 15; new rear-end (followers react 1.0 s): 12; new front/side: 0

## Sec. 6: collisions not prevented with fusion

- misses: 307; braking-limited (also missed with perfect perception): 157 (cut-in 14, lead 32, pedestrian 111); sensing-limited: 150 (cut-in 58, lead 14, pedestrian 78)
- intervened in 276 of 307 misses, first intervention 1.5 s (IQR 1.0-2.1 s) before the baseline impact; prevented cases 1.7 s (IQR 1.3-2.5 s)
- no intervention: 31 (cut-in 7, lead 12, pedestrian 12); by hazard: cut_in_10m 1, cut_in_5m 6, lead_brake_4 4, lead_brake_6 4, lead_brake_8 4, ped_run 3, ped_run_occluded 3, ped_walk 3, ped_walk_occluded 3
- fusion vs lidar alone: prevents 62 that lidar misses, loses 7
- fusion vs radar alone: prevents 232 that radar misses, loses 38
- fusion prevents 25 that neither sensor prevents alone

## Sensor robustness sweep (fused, all policies)

| setting | prevented | lead | cut-in | pedestrian | new rear-end (react 1.0 s) | intervened |
|---|---|---|---|---|---|---|
| default (clutter 1.5, p_d 0.9, no dropout) | 507/814 (62.3%) | 65/111 | 241/313 | 201/390 | 66 | 889 |
| radar clutter 0 / scan | 507/814 (62.3%) | 65/111 | 239/313 | 203/390 | 69 | 884 |
| radar clutter 3 / scan | 517/814 (63.5%) | 66/111 | 243/313 | 208/390 | 75 | 910 |
| radar p_d 0.95 | 509/814 (62.5%) | 65/111 | 244/313 | 200/390 | 68 | 888 |
| radar p_d 0.80 | 497/814 (61.1%) | 65/111 | 233/313 | 199/390 | 65 | 881 |
| LiDAR dropout 30% | 499/814 (61.3%) | 68/111 | 239/313 | 192/390 | 72 | 892 |
| LiDAR dropout 60% | 478/814 (58.7%) | 67/111 | 231/313 | 180/390 | 69 | 889 |

