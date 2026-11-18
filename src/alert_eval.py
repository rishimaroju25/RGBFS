"""False alarms and period-by-period scoring for stress alerts (Week 6).

Owner: Salah (Evaluation / backtesting support). Notebook: notebooks/14_false_alarms_by_period.ipynb

Any signal can be scored here once it is turned into a daily True/False alert
series (for example VIX >= 30, a forecast above 25%, a model probability above
a threshold, or Rishi's anomaly flags).

Definitions
-----------
alert event   consecutive alert days, merging gaps of up to `merge_gap` trading
              days, so one long alert counts once. Dated by its first day.
warning       an alert event that starts on a calm day (not inside a labeled
              stress episode). Events that start during an episode are counted
              separately as `during_stress`: they confirm stress, they don't warn.
true warning  a warning followed by the start of a stress episode within
              `horizon` trading days after the event starts. An alert that
              stays on for months before an episode does not count as a
              warning for it: it is a false alarm.
false alarm   a warning with no episode start in that window. `after_stress`
              marks false alarms that start within `horizon` days after an
              episode ends (the signal is still elevated from the last episode).
caught        an episode is caught if at least one true warning points to it.
lead          trading days from the first true warning's start to the episode start.
quiet day     a calm day with no episode starting in the next `horizon` days.
              day_false_positive_rate = share of quiet days with an alert.

The labels are evaluation targets and look ahead by design; nothing here feeds
back into a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON = 20
MERGE_GAP = 5


@dataclass
class AlertResult:
    events: pd.DataFrame     # one row per alert event
    days: pd.DataFrame       # one row per trading day: alert, calm, quiet
    episodes: pd.DataFrame   # one row per episode start: caught, lead


def episode_starts(stress: pd.Series) -> pd.DatetimeIndex:
    """First day of each run of stress days (stress is 0/1 with NaN where unlabeled)."""
    s = stress.fillna(0).astype(int)
    return s.index[(s == 1) & (s.shift(1, fill_value=0) == 0)]


def score_alerts(alert: pd.Series, stress: pd.Series, horizon: int = HORIZON,
                 merge_gap: int = MERGE_GAP) -> AlertResult:
    """Score a daily alert series against the stress labels. Both share a trading-day index."""
    idx = alert.index.intersection(stress.dropna().index).sort_values()
    a = alert.reindex(idx).fillna(False).astype(bool).to_numpy()
    st = stress.reindex(idx).astype(int).to_numpy()
    n = len(idx)
    starts = np.flatnonzero((st == 1) & (np.r_[0, st[:-1]] == 0))
    ends = np.flatnonzero((st == 1) & (np.r_[st[1:], 0] == 0))

    # alert events
    on = np.flatnonzero(a)
    events = []
    if len(on):
        s = e = on[0]
        for p in list(on[1:]) + [None]:
            if p is not None and p - e <= merge_gap + 1:
                e = p
                continue
            events.append((s, e))
            if p is not None:
                s = e = p

    rows = []
    for s, e in events:
        during = bool(st[s] == 1)
        hit = starts[(starts > s) & (starts <= s + horizon)]
        target = hit[0] if (len(hit) and not during) else None
        recent = ends[(ends < s) & (ends >= s - horizon)]
        rows.append({
            "start": idx[s], "end": idx[e], "alert_days": int(a[s:e + 1].sum()),
            "during_stress": during,
            "true_warning": target is not None,
            "false_alarm": (not during) and target is None,
            "after_stress": (not during) and target is None and len(recent) > 0,
            "episode_start": idx[target] if target is not None else pd.NaT,
            "lead": int(target - s) if target is not None else np.nan,
        })
    ev = pd.DataFrame(rows, columns=["start", "end", "alert_days", "during_stress", "true_warning",
                                     "false_alarm", "after_stress", "episode_start", "lead"])

    tw = ev[ev["true_warning"]]
    ep = pd.DataFrame({"start": idx[starts]})
    first = tw.groupby("episode_start")["lead"].max() if len(tw) else pd.Series(dtype=float)
    ep["caught"] = ep["start"].isin(first.index)
    ep["lead"] = ep["start"].map(first)

    soon = np.zeros(n, dtype=bool)
    for p in starts:
        soon[max(0, p - horizon):p] = True
    days = pd.DataFrame({"alert": a, "calm": st == 0, "quiet": (st == 0) & ~soon}, index=idx)
    return AlertResult(ev, days, ep)


def summarize(r: AlertResult, start=None, end=None) -> dict:
    """Headline numbers for events and episodes that start in [start, end], and days in that range."""
    lo = pd.Timestamp(start) if start is not None else r.days.index[0]
    hi = pd.Timestamp(end) if end is not None else r.days.index[-1]
    ev = r.events[(r.events["start"] >= lo) & (r.events["start"] <= hi)]
    ep = r.episodes[(r.episodes["start"] >= lo) & (r.episodes["start"] <= hi)]
    d = r.days.loc[lo:hi]
    warn = ev[~ev["during_stress"]]
    years = len(d) / 252
    return {
        "alert_day_share": d["alert"].mean(),
        "warnings": len(warn),
        "true_warnings": int(warn["true_warning"].sum()),
        "false_alarms": int(warn["false_alarm"].sum()),
        "false_alarms_after_stress": int(warn["after_stress"].sum()),
        "false_alarm_share": warn["false_alarm"].mean() if len(warn) else np.nan,
        "false_alarms_per_year": warn["false_alarm"].sum() / years if years else np.nan,
        "day_false_positive_rate": d.loc[d["quiet"], "alert"].mean() if d["quiet"].any() else np.nan,
        "episodes": len(ep),
        "episodes_caught": int(ep["caught"].sum()),
        "median_lead": ep["lead"].median() if ep["caught"].any() else np.nan,
        "alerts_during_stress": int(ev["during_stress"].sum()),
    }
