#!/usr/bin/env python3
"""NQ opening-range breakout grid with chronological walk-forward.

Point targets and stops are NQ index points, not option premium.
One NQ contract. Brackets are measured from the signal price
(OR high/low, or the open when the bar gaps through). Friction is
applied after the path is known, so gross / base / stress share a path.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

SRC_CANDIDATES = (
    Path("/workspace/attachments/Dataset_NQ_1min_2022_2025.csv"),
    Path(__file__).resolve().parent.parent / "data" / "Dataset_NQ_1min_2022_2025.csv",
)
OUT_CANDIDATES = (
    Path("/workspace/src/data/orb_results.json"),
    Path(__file__).resolve().parent.parent / "data" / "orb_results.json",
)


def _pick(paths: tuple[Path, ...], prefer_existing: bool) -> Path:
    if prefer_existing:
        for p in paths:
            if p.exists():
                return p
    for p in paths:
        if p.parent.exists():
            return p
    return paths[-1]


SRC = _pick(SRC_CANDIDATES, prefer_existing=True)
OUT = _pick(OUT_CANDIDATES, prefer_existing=False)

POINT_VALUE = 20.0
START_EQ = 100_000.0
TICK = 0.25
COMM_BASE = 4.50 / POINT_VALUE  # points
COMM_STRESS = 8.00 / POINT_VALUE
ORS = (5, 15, 30, 60)
TPS = (50, 100, 150, 200)
SLS = (50, 100, 150, 200)
SPLIT = 20240701  # combo-level IS / OOS cut
# Expanding walk-forward OOS quarters: [start, end)
FOLDS = [
    (20230701, 20231001),
    (20231001, 20240101),
    (20240101, 20240401),
    (20240401, 20240701),
    (20240701, 20241001),
    (20241001, 20250101),
    (20250101, 20250401),
    (20250401, 20250701),
    (20250701, 20251001),
    (20251001, 20260101),
]
MIN_TRADES = 40
ROLL_DAYS = 252


def ymd_to_iso(d: int) -> str:
    return f"{d // 10000:04d}-{(d // 100) % 100:02d}-{d % 100:02d}"


def quarter_of(d: int) -> tuple[int, int]:
    m = (d // 100) % 100
    return d // 10000, (m - 1) // 3 + 1


def month_of(d: int) -> int:
    return d // 100  # YYYYMM


# --- unit checks for the fill model ------------------------------------------------

def _sess(bars):
    return {"o": [b[0] for b in bars], "h": [b[1] for b in bars], "l": [b[2] for b in bars], "c": [b[3] for b in bars]}


def self_check() -> None:
    # next bar target only (entry bar does not reach the target)
    pnl, reason, _ = resolve_exit(_sess([(100, 101, 99, 100.5), (100.5, 160, 100, 150)]), 1, 100, 0, False, 50, 50)
    assert reason == "target" and pnl == 50, (pnl, reason)
    # both touched on the bar after entry -> stop
    pnl, reason, both = resolve_exit(_sess([(100, 101, 99, 100.5), (100, 160, 40, 150)]), 1, 100, 0, False, 50, 50)
    assert reason == "stop" and pnl == -50 and both, (pnl, reason, both)
    # gap through stop fills at the open
    pnl, reason, _ = resolve_exit(_sess([(100, 101, 99, 100), (40, 45, 39, 42)]), 1, 100, 0, False, 50, 50)
    assert reason == "stop" and abs(pnl - (40 - 100)) < 1e-9, (pnl, reason)
    # time exit
    pnl, reason, _ = resolve_exit(_sess([(100, 101, 99, 100), (100, 110, 95, 108), (108, 112, 107, 109)]), 1, 100, 0, False, 50, 50)
    assert reason == "time" and abs(pnl - 9) < 1e-9, (pnl, reason)
    # gap entry, stop on the entry bar
    pnl, reason, _ = resolve_exit(_sess([(130, 140, 70, 120)]), 1, 130, 0, True, 50, 50)
    assert reason == "stop" and pnl == -50, (pnl, reason)
    # non-gap breakout bar trades the target and not the stop
    pnl, reason, _ = resolve_exit(_sess([(100, 160, 99, 150), (150, 151, 149, 150)]), 1, 100, 0, False, 50, 50)
    assert reason == "target" and pnl == 50, (pnl, reason)
    # non-gap breakout bar trades both -> ambiguous stop
    pnl, reason, both = resolve_exit(_sess([(100, 160, 40, 150)]), 1, 100, 0, False, 50, 50)
    assert reason == "stop" and pnl == -50 and both, (pnl, reason)
    # non-gap breakout bar wicks through the stop only; that wick is ignored; time exit later
    pnl, reason, _ = resolve_exit(_sess([(100, 110, 40, 105), (105, 108, 104, 106)]), 1, 100, 0, False, 50, 50)
    assert reason == "time" and abs(pnl - 6) < 1e-9, (pnl, reason)
    # Max drawdown is the deepest percentage, not the percentage at the largest dollar loss.
    # 100 → 200 → 160 is a 20% / $40 drawdown. 160 → 500 → 450 is only 10%, but $50.
    sm = summarize([20220103, 20220104, 20220105, 20220106], [100.0, -40.0, 340.0, -50.0], [], "base", start_eq=100.0)
    assert abs(sm["max_dd_pct"] - 20.0) < 1e-9, sm
    assert abs(sm["max_dd_usd"] - 40.0) < 1e-9, sm
    print("self_check ok")


def net_of(gross: float, reason: str, mode: str) -> float:
    exit_slip = 0.0 if reason == "target" else 1.0
    if mode == "gross":
        return gross
    if mode == "base":
        return gross - TICK - TICK * exit_slip - COMM_BASE
    if mode == "stress":
        return gross - 2 * TICK - 2 * TICK * exit_slip - COMM_STRESS
    raise KeyError(mode)


def load_sessions() -> list[dict]:
    buckets: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    n_raw = 0
    with SRC.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[0] != "timestamp ET":
            raise SystemExit(f"unexpected header {header[:3]}")
        for row in reader:
            n_raw += 1
            ts = row[0]
            date_s, time_s = ts.rsplit(" ", 1)
            hs, ms = time_s.split(":")
            h, m = int(hs), int(ms)
            minutes = h * 60 + m
            # RTH stamps 09:30 through 16:00 inclusive
            if minutes < 9 * 60 + 30 or minutes > 16 * 60:
                continue
            mon_s, day_s, year_s = date_s.split("/")
            d = int(year_s) * 10000 + int(mon_s) * 100 + int(day_s)
            off = minutes - (9 * 60 + 30)
            buckets[d].append(
                (off, round(float(row[1]), 2), round(float(row[2]), 2), round(float(row[3]), 2), round(float(row[4]), 2))
            )
    sessions = []
    dup_days = 0
    for d in sorted(buckets):
        rows = sorted(buckets[d], key=lambda r: r[0])
        dedup = []
        seen = set()
        for r in rows:
            if r[0] in seen:
                dup_days += 1
                continue
            seen.add(r[0])
            dedup.append(r)
        if not dedup or dedup[0][0] != 0:
            # no 09:30 bar — not a usable RTH open
            continue
        offs = [r[0] for r in dedup]
        o = [r[1] for r in dedup]
        h = [r[2] for r in dedup]
        l = [r[3] for r in dedup]
        c = [r[4] for r in dedup]
        sessions.append(
            {
                "d": d,
                "off": offs,
                "o": o,
                "h": h,
                "l": l,
                "c": c,
                "open": o[0],
                "close": c[-1],
                "complete": offs[-1] == 390 and len(offs) == 391,
            }
        )
    print(f"raw rows {n_raw} rth sessions {len(sessions)} dup-minute drops {dup_days}")
    return sessions


def simulate_day(sess: dict) -> dict:
    """Return entries keyed by OR length and the bar path needed for exits.

    Entry is independent of target and stop.
    """
    offs = sess["off"]
    o, h, l, c = sess["o"], sess["h"], sess["l"], sess["c"]
    index = {off: i for i, off in enumerate(offs)}
    last_off = offs[-1]
    out = {"d": sess["d"], "open": sess["open"], "close": sess["close"], "ors": {}}
    for n in ORS:
        needed = list(range(n))
        if any(m not in index for m in needed):
            out["ors"][n] = {"status": "incomplete_or"}
            continue
        orh = max(h[index[m]] for m in needed)
        orl = min(l[index[m]] for m in needed)
        width = orh - orl
        if width <= 0:
            out["ors"][n] = {"status": "zero_width", "width": width}
            continue
        entry = None
        ambiguous = False
        for i, off in enumerate(offs):
            if off < n:
                continue
            if off >= last_off:
                break
            long_hit = h[i] >= orh
            short_hit = l[i] <= orl
            if long_hit and short_hit:
                ambiguous = True
                entry = None
                break
            if long_hit:
                gap = o[i] >= orh
                px = o[i] if gap else orh
                entry = (1, px, i, gap)
                break
            if short_hit:
                gap = o[i] <= orl
                px = o[i] if gap else orl
                entry = (-1, px, i, gap)
                break
        out["ors"][n] = {
            "status": "ambiguous" if ambiguous else ("flat" if entry is None else "trade"),
            "width": width,
            "entry": entry,
        }
    return out


def resolve_exit(sess: dict, side: int, entry_px: float, entry_i: int, gap: bool, tp: int, sl: int):
    """Gross points from the signal price, before friction.

    Gap fills are managed on the entry bar (the open is the fill).
    Non-gap fills: a target printed on the breakout bar is taken, because a
    continuous path from an open on the near side of the range to a high/low
    beyond the target must pass the entry and then the target. A stop printed
    on that same bar is ignored unless the bar also reaches the target — then
    the sequence is ambiguous and the stop is assumed. Pre-breakout wicks are
    not treated as stops.
    """
    o, h, l, c = sess["o"], sess["h"], sess["l"], sess["c"]
    stop_px = entry_px - side * sl
    tgt_px = entry_px + side * tp
    last = len(o) - 1
    both = False
    if gap:
        start = entry_i
    else:
        hj, lj = h[entry_i], l[entry_i]
        if side == 1:
            hit_tgt = hj >= tgt_px
            hit_stop = lj <= stop_px
        else:
            hit_tgt = lj <= tgt_px
            hit_stop = hj >= stop_px
        if hit_tgt and hit_stop:
            return -float(sl), "stop", True
        if hit_tgt:
            return float(tp), "target", False
        start = entry_i + 1
    for j in range(start, last + 1):
        oj, hj, lj, cj = o[j], h[j], l[j], c[j]
        if side == 1:
            gapped_stop = oj <= stop_px
            hit_stop = lj <= stop_px
            hit_tgt = hj >= tgt_px
            if gapped_stop:
                return (oj - entry_px), "stop", both
            if hit_stop and hit_tgt:
                return -float(sl), "stop", True
            if hit_stop:
                return -float(sl), "stop", both
            if hit_tgt:
                return float(tp), "target", both
        else:
            gapped_stop = oj >= stop_px
            hit_stop = hj >= stop_px
            hit_tgt = lj <= tgt_px
            if gapped_stop:
                return (entry_px - oj), "stop", both
            if hit_stop and hit_tgt:
                return -float(sl), "stop", True
            if hit_stop:
                return -float(sl), "stop", both
            if hit_tgt:
                return float(tp), "target", both
        if j == last:
            return (cj - entry_px) * side, "time", both
    return None


def all_trades(sessions: list[dict]) -> tuple[dict, dict]:
    """trades[(or,tp,sl)] = list of trade dicts aligned to session order (only days with a trade)."""
    keys = [(n, tp, sl) for n in ORS for tp in TPS for sl in SLS]
    trades = {k: [] for k in keys}
    diag = {
        "by_or": {},
        "same_bar_stop_and_target": 0,
        "sessions": len(sessions),
        "complete_sessions": sum(1 for s in sessions if s["complete"]),
        "early_close_sessions": sum(1 for s in sessions if not s["complete"]),
    }
    width_sum = {n: 0.0 for n in ORS}
    width_n = {n: 0 for n in ORS}
    status_n = {n: defaultdict(int) for n in ORS}
    both_n = 0
    for sess in sessions:
        sim = simulate_day(sess)
        for n in ORS:
            info = sim["ors"][n]
            status_n[n][info["status"]] += 1
            if "width" in info:
                width_sum[n] += info["width"]
                width_n[n] += 1
            if info["status"] != "trade":
                continue
            side, px, i, gap = info["entry"]
            for tp in TPS:
                for sl in SLS:
                    resolved = resolve_exit(sess, side, px, i, gap, tp, sl)
                    if resolved is None:
                        continue
                    gross, reason, both = resolved
                    if both:
                        both_n += 1
                    trades[(n, tp, sl)].append(
                        {
                            "d": sess["d"],
                            "side": side,
                            "gross": gross,
                            "reason": reason,
                        }
                    )
    diag["same_bar_stop_and_target"] = both_n
    for n in ORS:
        diag["by_or"][str(n)] = {
            "status": dict(status_n[n]),
            "avg_width": round(width_sum[n] / width_n[n], 2) if width_n[n] else None,
            "width_samples": width_n[n],
        }
    return trades, diag


def rth_gaps(sessions: list[dict]) -> list[dict]:
    gaps = []
    prev = None
    for s in sessions:
        if prev is not None:
            gap = s["open"] - prev["close"]
            if abs(gap) >= 200:
                gaps.append(
                    {
                        "from": ymd_to_iso(prev["d"]),
                        "to": ymd_to_iso(s["d"]),
                        "points": round(gap, 2),
                    }
                )
        prev = s
    return gaps


def _sharpe(rets: list[float], min_n: int = 20):
    n = len(rets)
    if n < min_n:
        return None, (None if n < 5 else _vol(rets))
    mu = sum(rets) / n
    var = sum((r - mu) ** 2 for r in rets) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None, 0.0
    vol = sd * math.sqrt(252)
    sharpe = (mu / sd) * math.sqrt(252)
    return sharpe, vol


def _vol(rets: list[float]):
    n = len(rets)
    if n < 5:
        return None
    mu = sum(rets) / n
    var = sum((r - mu) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(252)


def build_daily(dates: list[int], trade_list: list[dict], mode: str) -> list[float]:
    pnl = {t["d"]: 0.0 for t in trade_list}
    # overwrite properly — sum in case of surprises
    pnl = defaultdict(float)
    for t in trade_list:
        pnl[t["d"]] += net_of(t["gross"], t["reason"], mode) * POINT_VALUE
    return [pnl.get(d, 0.0) for d in dates]


def summarize(dates: list[int], daily_usd: list[float], trade_list: list[dict], mode: str, start_eq: float = START_EQ) -> dict:
    eq = start_eq
    peak = start_eq
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    rets = []
    eod = []
    for pnl in daily_usd:
        prev = eq
        eq = eq + pnl
        eod.append(eq)
        if prev > 0:
            rets.append(pnl / prev)
        else:
            rets.append(0.0)
        if eq > peak:
            peak = eq
        dd_usd = peak - eq
        dd_pct = dd_usd / peak if peak > 0 else 0.0
        # Percentage and dollars are one event: the deepest percent decline.
        # A later, larger dollar loss at a smaller percent must not replace it.
        # Otherwise a single year can print a deeper drawdown than the full sample.
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_usd = dd_usd
    wins = losses = scratches = 0
    long_n = short_n = 0
    long_pts = short_pts = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    reasons = {"target": 0, "stop": 0, "time": 0}
    net_pts = 0.0
    win_pts = []
    loss_pts = []
    for t in trade_list:
        pts = net_of(t["gross"], t["reason"], mode)
        net_pts += pts
        reasons[t["reason"]] += 1
        if t["side"] == 1:
            long_n += 1
            long_pts += pts
        else:
            short_n += 1
            short_pts += pts
        if pts > 1e-9:
            wins += 1
            gross_win += pts
            win_pts.append(pts)
        elif pts < -1e-9:
            losses += 1
            gross_loss += pts
            loss_pts.append(pts)
        else:
            scratches += 1
    n = len(trade_list)
    if n == 0 and daily_usd:
        net_pts = sum(daily_usd) / POINT_VALUE
    sharpe, vol = _sharpe(rets)
    end_eq = eod[-1] if eod else start_eq
    ret = (end_eq / start_eq - 1.0) if start_eq else None
    # calendar span
    if len(dates) >= 2:
        y0, m0, d0 = dates[0] // 10000, (dates[0] // 100) % 100, dates[0] % 100
        y1, m1, d1 = dates[-1] // 10000, (dates[-1] // 100) % 100, dates[-1] % 100
        from datetime import date

        span = (date(y1, m1, d1) - date(y0, m0, d0)).days / 365.25
    else:
        span = 0.0
    cagr = None
    if span >= 0.25 and end_eq > 0 and start_eq > 0:
        cagr = (end_eq / start_eq) ** (1 / span) - 1
    pf = (gross_win / abs(gross_loss)) if gross_loss < 0 else None
    avg_win = (sum(win_pts) / len(win_pts)) if win_pts else None
    avg_loss = (sum(loss_pts) / len(loss_pts)) if loss_pts else None
    payoff = (avg_win / abs(avg_loss)) if avg_win is not None and avg_loss not in (None, 0) else None
    calmar = (cagr / max_dd_pct) if cagr is not None and max_dd_pct > 1e-9 else None
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": (wins / n) if n else None,
        "net_points": round(net_pts, 2),
        "net_usd": round(net_pts * POINT_VALUE, 2),
        "return_pct": None if ret is None else round(ret * 100, 2),
        "start_equity": round(start_eq, 2),
        "end_equity": round(end_eq, 2),
        "max_dd_pct": round(max_dd_pct * 100, 2),
        "max_dd_usd": round(max_dd_usd, 2),
        "vol_ann_pct": None if vol is None else round(vol * 100, 2),
        "sharpe": None if sharpe is None else round(sharpe, 3),
        "profit_factor": None if pf is None else round(pf, 3),
        "avg_points": round(net_pts / n, 2) if n else None,
        "avg_win_points": None if avg_win is None else round(avg_win, 2),
        "avg_loss_points": None if avg_loss is None else round(avg_loss, 2),
        "payoff": None if payoff is None else round(payoff, 3),
        "cagr_pct": None if cagr is None else round(cagr * 100, 2),
        "calmar": None if calmar is None else round(calmar, 3),
        "n_days": len(dates),
        "exposure_pct": round(100.0 * n / len(dates), 1) if dates else None,
        "long_trades": long_n,
        "short_trades": short_n,
        "long_points": round(long_pts, 2),
        "short_points": round(short_pts, 2),
        "exits": reasons,
        "span_years": round(span, 3),
    }


def bh_daily(dates: list[int], close_of: dict[int, float], open_of: dict[int, float]) -> list[float]:
    """Mark-to-market daily P&L of 1 long NQ, entered at the first session's open.

    Costs match the strategy's base friction on the single round turn:
    entry +1 tick, final mark is mid (close); a one-time commission and
    exit tick are NOT smeared into daily marks (immaterial vs the path).
    Entry tick is charged on day 1.
    """
    if not dates:
        return []
    entry = open_of[dates[0]] + TICK
    prev_mark = entry
    out = []
    for i, d in enumerate(dates):
        mark = close_of[d]
        pnl = (mark - prev_mark) * POINT_VALUE
        if i == 0:
            pnl -= COMM_BASE * POINT_VALUE
        out.append(pnl)
        prev_mark = mark
    return out


def slice_window(dates, daily, trades, start, end):
    """[start, end) on YYYYMMDD."""
    idx = [i for i, d in enumerate(dates) if start <= d < end]
    if not idx:
        return [], [], []
    sub_dates = [dates[i] for i in idx]
    sub_daily = [daily[i] for i in idx]
    sub_tr = [t for t in trades if start <= t["d"] < end]
    return sub_dates, sub_daily, sub_tr


def attribution(dates, daily_usd, bh_daily_usd, trade_list, key_fn, label_fn):
    """Period rows. Equity compounds inside this window from START_EQ."""
    # map date -> index
    eq = START_EQ
    bh = START_EQ
    eod = {}
    bh_eod = {}
    for d, pnl, bp in zip(dates, daily_usd, bh_daily_usd):
        eq += pnl
        bh += bp
        eod[d] = eq
        bh_eod[d] = bh
    # group dates preserving order
    groups = []
    cur_k = None
    cur = []
    for d in dates:
        k = key_fn(d)
        if k != cur_k:
            if cur:
                groups.append((cur_k, cur))
            cur_k = k
            cur = [d]
        else:
            cur.append(d)
    if cur:
        groups.append((cur_k, cur))
    rows = []
    prev_eq = START_EQ
    prev_bh = START_EQ
    running_dates_before = 0
    all_dates = dates
    for k, ds in groups:
        start_eq = prev_eq
        start_bh = prev_bh
        end_eq = eod[ds[-1]]
        end_bh = bh_eod[ds[-1]]
        # daily rets inside period using equity at prior close
        rets = []
        eq_cursor = start_eq
        peak = start_eq
        max_dd = 0.0
        for d in ds:
            pnl = eod[d] - eq_cursor
            if eq_cursor > 0:
                rets.append(pnl / eq_cursor)
            eq_cursor = eod[d]
            if eq_cursor > peak:
                peak = eq_cursor
            if peak > 0:
                max_dd = max(max_dd, (peak - eq_cursor) / peak)
        sharpe, vol = _sharpe(rets, min_n=20)
        # vol for shorter samples still useful down to 5 sessions
        if vol is None:
            vol = _vol(rets)
        trs = [t for t in trade_list if t["d"] in set(ds)]
        # But trade_list points are base-mode if daily is base. Recompute from daily consistency:
        # win rate from trade points under the SAME mode used to build daily_usd.
        # Caller passes trades already filtered; we need mode. Infer from daily vs trades is messy.
        # We'll pass base-mode trades and base daily. Win rate uses base net.
        wins = 0
        for t in trs:
            if net_of(t["gross"], t["reason"], "base") > 1e-9:
                wins += 1
        n = len(trs)
        ret = (end_eq / start_eq - 1.0) if start_eq else 0.0
        bh_ret = (end_bh / start_bh - 1.0) if start_bh else 0.0
        y, q = quarter_of(ds[0])
        rows.append(
            {
                "label": label_fn(k, ds),
                "year": y,
                "quarter": q,
                "month": (ds[0] // 100) % 100,
                "start": ymd_to_iso(ds[0]),
                "end": ymd_to_iso(ds[-1]),
                "strategy_return_pct": round(ret * 100, 2),
                "bh_return_pct": round(bh_ret * 100, 2),
                "diff_pp": round((ret - bh_ret) * 100, 2),
                "start_equity": round(start_eq, 2),
                "end_equity": round(end_eq, 2),
                "bh_start_equity": round(start_bh, 2),
                "bh_end_equity": round(end_bh, 2),
                "trades": n,
                "win_rate": round(wins / n, 4) if n else None,
                "max_dd_pct": round(max_dd * 100, 2),
                "vol_ann_pct": None if vol is None else round(vol * 100, 2),
                "sharpe": None if sharpe is None else round(sharpe, 3),
                "net_usd": round(end_eq - start_eq, 2),
                "n_days": len(ds),
            }
        )
        prev_eq = end_eq
        prev_bh = end_bh
        running_dates_before += len(ds)
    _ = (all_dates, running_dates_before)
    return rows


def q_label(_k, ds):
    y, q = quarter_of(ds[0])
    return f"{y} Q{q}"


def y_label(k, _ds):
    return str(k)


def m_label(k, _ds):
    y = k // 100
    m = k % 100
    names = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{names[m - 1]} {y}"


def select_combo(cands: list[dict]) -> dict | None:
    """Max base-case Sharpe, min trades, deterministic tie break.

    Tie-break: higher net points, then shorter opening range, then tighter stop, then tighter target.
    """
    eligible = [c for c in cands if c["summary"]["trades"] >= MIN_TRADES and c["summary"]["sharpe"] is not None]
    if not eligible:
        eligible = [c for c in cands if c["summary"]["trades"] >= 15 and c["summary"]["sharpe"] is not None]
    if not eligible:
        return None
    eligible.sort(
        key=lambda c: (
            c["summary"]["sharpe"],
            c["summary"]["net_points"],
            -c["or"],
            -c["sl"],
            -c["tp"],
        ),
        reverse=True,
    )
    # reverse=True would invert the "smaller OR" preference because -or is larger when or is smaller.
    # With reverse=True: higher sharpe first (good), higher net points (good), higher -or means smaller or (good),
    # higher -sl means smaller sl (good), higher -tp means smaller tp (good). Correct.
    return eligible[0]


def pack_summary_pair(dates, trades):
    daily_base = build_daily(dates, trades, "base")
    daily_gross = build_daily(dates, trades, "gross")
    daily_stress = build_daily(dates, trades, "stress")
    base = summarize(dates, daily_base, trades, "base")
    gross = summarize(dates, daily_gross, trades, "gross")
    stress = summarize(dates, daily_stress, trades, "stress")
    base["gross_net_points"] = gross["net_points"]
    base["gross_return_pct"] = gross["return_pct"]
    base["gross_sharpe"] = gross["sharpe"]
    base["gross_max_dd_pct"] = gross["max_dd_pct"]
    base["stress_net_points"] = stress["net_points"]
    base["stress_return_pct"] = stress["return_pct"]
    base["stress_sharpe"] = stress["sharpe"]
    base["stress_max_dd_pct"] = stress["max_dd_pct"]
    return base, daily_base


def period_bundle(dates, daily_base, bh, trades):
    return {
        "quarterly": attribution(dates, daily_base, bh, trades, quarter_of, q_label),
        "annual": attribution(dates, daily_base, bh, trades, lambda d: d // 10000, y_label),
        "monthly": attribution(dates, daily_base, bh, trades, month_of, m_label),
    }


def main() -> None:
    self_check()
    sessions = load_sessions()
    dates = [s["d"] for s in sessions]
    close_of = {s["d"]: s["close"] for s in sessions}
    open_of = {s["d"]: s["open"] for s in sessions}
    trades, diag = all_trades(sessions)
    diag["large_rth_gaps"] = rth_gaps(sessions)
    diag["first_day"] = ymd_to_iso(dates[0])
    diag["last_day"] = ymd_to_iso(dates[-1])

    bh_full = bh_daily(dates, close_of, open_of)
    bh_summary = summarize(dates, bh_full, [], "base")
    # BH summarize with empty trades is fine for equity path; decorate label fields
    bh_summary["note"] = "1 NQ held from the first RTH open, marked on each RTH close. Not an ORB."

    grid = []
    daily_by_key = {}
    for n in ORS:
        for tp in TPS:
            for sl in SLS:
                key = (n, tp, sl)
                tr = trades[key]
                full, daily = pack_summary_pair(dates, tr)
                is_dates = [d for d in dates if d < SPLIT]
                oos_dates = [d for d in dates if d >= SPLIT]
                is_tr = [t for t in tr if t["d"] < SPLIT]
                oos_tr = [t for t in tr if t["d"] >= SPLIT]
                is_sum, _ = pack_summary_pair(is_dates, is_tr)
                oos_sum, _ = pack_summary_pair(oos_dates, oos_tr)
                bundle = period_bundle(dates, daily, bh_full, tr)
                grid.append(
                    {
                        "or": n,
                        "tp": tp,
                        "sl": sl,
                        "full": full,
                        "is": is_sum,
                        "oos": oos_sum,
                        "quarterly": bundle["quarterly"],
                        "annual": bundle["annual"],
                        "monthly": bundle["monthly"],
                    }
                )
                daily_by_key[key] = daily
                print(
                    f"OR{n} tp{tp} sl{sl} full {full['net_points']} pts sharpe {full['sharpe']} "
                    f"oos {oos_sum['net_points']} sharpe {oos_sum['sharpe']}"
                )

    def _audit_cell(g: dict) -> None:
        name = f"OR {g['or']}/{g['tp']}/{g['sl']}"
        full = g["full"]
        net = round(sum(r["net_usd"] for r in g["annual"]), 2)
        if abs(net - full["net_usd"]) > 0.05:
            raise SystemExit(f"{name} annual P&L {net} != full {full['net_usd']}")
        if abs(g["annual"][-1]["end_equity"] - full["end_equity"]) > 0.05:
            raise SystemExit(f"{name} annual end equity != full end equity")
        worst_year = max(r["max_dd_pct"] for r in g["annual"])
        if full["max_dd_pct"] + 0.02 < worst_year:
            raise SystemExit(f"{name} full DD {full['max_dd_pct']} < year DD {worst_year}")

    for g in grid:
        _audit_cell(g)
    print("attribution reconciles to every grid cell")

    # Walk-forward
    def run_wf(mode: str):
        """mode expanding | rolling. Returns book trades in OOS, folds, daily pnl aligned to oos dates."""
        oos_start = FOLDS[0][0]
        oos_dates = [d for d in dates if d >= oos_start]
        chosen_by_day = {}
        folds = []
        book = []
        for a, b in FOLDS:
            if mode == "expanding":
                is_dates = [d for d in dates if d < a]
            else:
                prior = [d for d in dates if d < a]
                is_dates = prior[-ROLL_DAYS:]
            cands = []
            is_set = set(is_dates)
            for n in ORS:
                for tp in TPS:
                    for sl in SLS:
                        tr = [t for t in trades[(n, tp, sl)] if t["d"] in is_set]
                        summary, _ = pack_summary_pair(is_dates, tr)
                        cands.append({"or": n, "tp": tp, "sl": sl, "summary": summary})
            pick = select_combo(cands)
            fold = {
                "oos_start": ymd_to_iso(a),
                "oos_end": ymd_to_iso(min(b - 1, dates[-1])) if any(a <= d < b for d in dates) else ymd_to_iso(a),
                "is_sessions": len(is_dates),
                "is_start": ymd_to_iso(is_dates[0]) if is_dates else None,
                "is_end": ymd_to_iso(is_dates[-1]) if is_dates else None,
            }
            if pick is None:
                fold["chosen"] = None
            else:
                fold["chosen"] = {
                    "or": pick["or"],
                    "tp": pick["tp"],
                    "sl": pick["sl"],
                    "is_sharpe": pick["summary"]["sharpe"],
                    "is_net_points": pick["summary"]["net_points"],
                    "is_trades": pick["summary"]["trades"],
                    "is_win_rate": pick["summary"]["win_rate"],
                    "is_max_dd_pct": pick["summary"]["max_dd_pct"],
                    "is_return_pct": pick["summary"]["return_pct"],
                }
                oos_tr = [t for t in trades[(pick["or"], pick["tp"], pick["sl"])] if a <= t["d"] < b]
                for t in oos_tr:
                    book.append({**t, "or": pick["or"], "tp": pick["tp"], "sl": pick["sl"]})
                for d in dates:
                    if a <= d < b:
                        chosen_by_day[d] = (pick["or"], pick["tp"], pick["sl"])
            # standalone fold result filled after book built per fold
            fold_tr = [t for t in book if a <= t["d"] < b]
            fold_dates = [d for d in dates if a <= d < b]
            if fold_dates:
                sm, _ = pack_summary_pair(fold_dates, fold_tr)
                bh = bh_daily(fold_dates, close_of, open_of)
                bh_sm = summarize(fold_dates, bh, [], "base")
                fold["oos"] = {
                    "trades": sm["trades"],
                    "win_rate": sm["win_rate"],
                    "net_points": sm["net_points"],
                    "return_pct": sm["return_pct"],
                    "sharpe": sm["sharpe"],
                    "max_dd_pct": sm["max_dd_pct"],
                    "end_equity": sm["end_equity"],
                    "bh_return_pct": bh_sm["return_pct"],
                    "profit_factor": sm["profit_factor"],
                }
            folds.append(fold)
        return oos_dates, book, folds

    def pack_wf(mode: str):
        oos_dates, book, folds = run_wf(mode)
        # BH over the OOS window only, restarting at 100k — comparable track
        bh = bh_daily(oos_dates, close_of, open_of)
        summary, daily = pack_summary_pair(oos_dates, book)
        bh_sm = summarize(oos_dates, bh, [], "base")
        # terminal exit tick on BH so the hold is a round turn
        if bh:
            bh[-1] -= TICK * POINT_VALUE
            bh_sm = summarize(oos_dates, bh, [], "base")
        bundle = period_bundle(oos_dates, daily, bh, book)
        equity = []
        eq = START_EQ
        bhe = START_EQ
        for d, pnl, bp in zip(oos_dates, daily, bh):
            eq += pnl
            bhe += bp
            equity.append({"d": ymd_to_iso(d), "eq": round(eq, 2), "bh": round(bhe, 2)})
        return {
            "summary": summary,
            "bh_summary": bh_sm,
            "folds": folds,
            "quarterly": bundle["quarterly"],
            "annual": bundle["annual"],
            "monthly": bundle["monthly"],
            "equity": equity,
        }

    print("walk-forward expanding...")
    wf_exp = pack_wf("expanding")
    print("walk-forward rolling...")
    wf_roll = pack_wf("rolling")
    print("WF expanding", wf_exp["summary"]["net_points"], "sharpe", wf_exp["summary"]["sharpe"])
    print("WF rolling", wf_roll["summary"]["net_points"], "sharpe", wf_roll["summary"]["sharpe"])

    # BH full sample equity for the chart of a selected combo is rebuilt client-side
    # from monthly tables. Also store a full-sample BH equity sampled for the benchmark line
    # on the combo chart — daily is ~750 points, fine.
    bh_eq = []
    eq = START_EQ
    for d, pnl in zip(dates, bh_full):
        eq += pnl
        bh_eq.append(round(eq, 2))
    # charge exit tick at the end for the stored path's last point
    if bh_eq:
        bh_eq[-1] = round(bh_eq[-1] - TICK * POINT_VALUE, 2)

    payload = {
        "meta": {
            "symbol": "NQ",
            "name": "CME E-mini Nasdaq-100 futures",
            "point_value": POINT_VALUE,
            "contracts": 1,
            "starting_equity": START_EQ,
            "tick": TICK,
            "commission_rt_usd": 4.50,
            "stress_commission_rt_usd": 8.00,
            "first_day": diag["first_day"],
            "last_day": diag["last_day"],
            "sessions": diag["sessions"],
            "complete_sessions": diag["complete_sessions"],
            "early_close_sessions": diag["early_close_sessions"],
            "split_date": ymd_to_iso(SPLIT),
            "wf_first_oos": ymd_to_iso(FOLDS[0][0]),
            "min_is_trades": MIN_TRADES,
            "selection": "Highest in-sample daily Sharpe (rf=0, sqrt(252)), minimum 40 trades. Ties: higher net points, then shorter opening range, then tighter stop, then tighter target.",
            "rules": [
                "Session is RTH, 09:30–16:00 America/New_York. Bars are minute stamps. The 16:00 stamp is the closing bar (volume spike) and is flatten-only — no new entries.",
                "Opening range is the first N minute bars starting 09:30. N is 5, 15, 30, or 60. A missing minute inside the range skips that day for that N.",
                "After the range completes, the first touch of the range high is a long and the first touch of the range low is a short. One trade a day. If one bar trades through both sides, the day is skipped.",
                "A stop order fills at the range level, or at the open if the bar gaps through. Target and stop are measured in index points from that signal price.",
                "Stops and targets are live from the next bar. On the breakout bar itself, a target is filled when the bar trades through it and does not also trade through the stop. If that bar reaches both, the stop is assumed. The pre-breakout wick is not a stop. Gap-through entries are managed on the entry bar from the open, and a bar that opens through the stop fills at the open.",
                "Still-open trades are flattened on the last RTH bar of that session, including early closes (13:00 / 13:15).",
                "No width filter, no trend filter, no VWAP filter, no re-entry, no overnight hold.",
                "Base friction: 1 tick adverse on entry, 1 tick adverse on stops and time exits, none on targets (limit), plus $4.50 round-turn commission. Stress: 2 ticks and $8. Gross: no friction. The price path does not depend on the cost assumption.",
                "Position is a constant 1 NQ ($20 per point) on a $100,000 capital base. Percent returns are return on that capital, not on notional. The same 1-lot base is used for buy-and-hold, which buys the first RTH open of the window and marks each RTH close.",
                "Closed-trade drawdown only, as the largest percentage decline from a prior equity peak on the daily marked curve. The dollar figure is the loss at that same peak-to-trough, not the largest dollar loss if it happened at a smaller percentage. A single year cannot be deeper, in percent, than the full sample of the same path.",
                "Sharpe uses daily simple returns on marked equity, including flat sessions, rf = 0, sample standard deviation, sqrt(252). It is omitted when a window has fewer than 20 sessions. Volatility is omitted under 5 sessions.",
                "Full-sample and the 2024-07-01 holdout reset equity to $100,000. They are not tradable after seeing the whole grid. The walk-forward books reset only at the first out-of-sample day and then compound. Parameters for a quarter are chosen only on sessions before that quarter.",
            ],
        },
        "diagnostics": diag,
        "bh_full_summary": bh_summary,
        "dates": [ymd_to_iso(d) for d in dates],
        "bh_equity": bh_eq,
        "grid": grid,
        "wf_expanding": wf_exp,
        "wf_rolling": wf_roll,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print("wrote", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
