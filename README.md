# Opening range breakout, NQ

One-minute Nasdaq-100 futures, 2022-12-27 through 2025-12-11. Sixty-four opening-range / target / stop combinations, one NQ contract on $100,000, and a walk-forward that is not allowed to pick parameters after it has seen the quarter.

Targets and stops are index points, not premium. There is no filter beyond the open.

## What is in here

| Path | What it is |
|---|---|
| `research/orb_nq.py` | The engine. Same fill, friction, and walk-forward rules as the research note. |
| `notebooks/opening_range_breakout.ipynb` | Every table from the note: verdict, both walk-forwards, folds, quarter / year / month attribution, and all 64 cells. |
| `data/nq_orb_results.xlsx` | Those same tables as sheets, plus quarter and month attribution for every cell. |
| `data/orb_results.json` | The machine-readable result the note is rendered from. |
| `data/Dataset_NQ_1min_2022_2025.csv` | The tape. 1,048,575 bars, which is Excel’s row ceiling, ending 2025-12-11 20:52. RTH is 09:30–16:00 New York. |

Regenerate the JSON from the tape:

```bash
python research/orb_nq.py
```

The script writes `data/orb_results.json` when it is run from a clone of this repo.

## Drawdown, and why two views of one cell used to disagree

Percent drawdown is the deepest peak-to-trough decline on the daily equity of that path. The dollar figure is the loss at that same trough.

An earlier build stored the percentage at the *largest dollar* loss instead. On OR 30 / target 150 / stop 150 that printed a full-sample drawdown of 14.7%, because a later, smaller-percent dip happened on a larger account. The 2023 row of the same cell was a 20.1% decline. A year cannot be deeper, in percent, than the full sample. The full-sample figure for that cell is now 20.14%, the same event as 2023.

The year table in the note that ends at $222,220 is this cell: OR 30 / target 150 / stop 150, +6,111 points, +122.22%. The row above it in the grid, OR 30 / target 150 / stop 200, is a different cell: +6,291.75 points, +125.84%, ending at $225,835, full-sample drawdown 21.42%.

Compounded yearly returns already matched the grid. Points times $20 already matched the change in equity. Only the drawdown column was the wrong event.

## Rules, short version

Session is RTH. The opening range is the first 5, 15, 30, or 60 minutes. The first touch of the high is a long and the first touch of the low is a short. One trade a day. A bar that trades both sides of the range is skipped. Stops and targets are measured from the signal price. Still-open trades flatten on the last bar of the session.

Base friction is one tick and $4.50 round turn. Stress is two ticks and $8. Gross is the same path with neither. Sharpe is daily simple returns on marked equity, including flat sessions, rf = 0, sample standard deviation, sqrt(252). It is omitted under 20 sessions.

The walk-forward chooses the highest in-sample Sharpe with at least 40 trades. Ties break toward more points, then a shorter range, then a tighter stop, then a tighter target. Parameters for a quarter are chosen only on sessions before that quarter.
