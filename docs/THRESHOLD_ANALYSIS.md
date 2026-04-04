# Threshold Analysis — V2 Dataset

Model: XGBoost | 606 features | Val set: 1,715 calls (23 tickets, 8 in completed)

## Threshold Sweep

| Thresh | F1 | Prec | Recall | Flagged | Correct | Wrong | Missed | Comp ✓ | Comp ✗ | Comp Missed |
|--------|------|------|--------|---------|---------|-------|--------|--------|--------|-------------|
| 0.40 | .278 | .165 | .870 | 121 | 20 | 101 | 3 | 7 | 22 | 1 |
| 0.50 | .323 | .211 | .696 | 76 | 16 | 60 | 7 | 6 | 12 | 2 |
| 0.55 | .366 | .254 | .652 | 59 | 15 | 44 | 8 | 5 | 7 | 3 |
| 0.60 | .417 | .306 | .652 | 49 | 15 | 34 | 8 | 5 | 5 | 3 |
| 0.65 | .367 | .297 | .478 | 37 | 11 | 26 | 12 | 4 | 4 | 4 |
| 0.70 | .357 | .303 | .435 | 33 | 10 | 23 | 13 | 3 | 3 | 5 |
| 0.75 | .444 | .455 | .435 | 22 | 10 | 12 | 13 | 3 | 2 | 5 |
| **0.79** | **.476** | **.526** | **.435** | **19** | **10** | **9** | **13** | **3** | **2** | **5** |
| 0.85 | .471 | .727 | .348 | 11 | 8 | 3 | 15 | 2 | 0 | 6 |
| 0.90 | .214 | .600 | .130 | 5 | 3 | 2 | 20 | 1 | 0 | 7 |
| 0.95 | .083 | 1.00 | .043 | 1 | 1 | 0 | 22 | 0 | 0 | 8 |

## Key Thresholds

- **0.79 (selected)**: Best F1 (0.476) with precision > 50%. 10/23 caught, 9 false positives.
- **0.85**: Best precision/F1 balance. 73% precision, only 3 false positives, but misses 15 tickets.
- **0.60**: Best recall/F1 balance. Catches 15/23 (65%) but 34 false positives.

## Completed Calls (443 total, 8 tickets)

At threshold 0.79: catches 3/8 completed tickets with 2 false positives.
At threshold 0.85: catches 2/8 with 0 false positives.
At threshold 0.60: catches 5/8 with 5 false positives.

## Notes

- Ticket rate: 1.34% (23/1715) — extreme class imbalance
- Most tickets are in incomplete (10) and completed (8) calls
- Completed ticket patterns: STT number misrecording (weight 262→62, 187→87)
- Non-completed ticket patterns: outcome mislabeling, agent failures
