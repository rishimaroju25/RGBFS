"""Turning signals into alerts and scoring them against the labeled stress episodes.

Owner: Rishi. Week 5 (anomaly flags in a model) and Week 6 (alert lead time).

Project target (team plan): on trading day t, does a stress episode START within
the next `horizon` (20) trading days? Days already inside an episode are left
out of the target, because the question is early warning, not "are we in
stress now".

Episode starts come from docs/stress_periods.csv. Salah's labels give two start
dates: `start` (centered volatility window, the early bound) and
`start_trailing` (trailing window, the late bound). Lead times should be
reported for both (see src/stress_labels.py).

Everything here works on a trading-day calendar, so "20 days" always means 20
trading days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 20       # trading days ahead for the classification target
LOOKBACK = 60      # trading days before an episode start in which an alert counts as a warning
LATE = 10          # trading days after the start in which a first alert counts as "late"
MERGE_GAP = 5      # flagged days separated by fewer quiet days than this are one alert event


def load_episodes(path) -> pd.DataFrame:
    ep = pd.read_csv(path, parse_dates=["start", "end", "start_trailing"])
    return ep.sort_values("start").reset_index(drop=True)


def _positions(calendar: pd.DatetimeIndex, dates) -> np.ndarray:
    """Calendar position of each date (next trading day if the date is not a trading day)."""
    return calendar.searchsorted(pd.DatetimeIndex(dates))


def episode_masks(calendar: pd.DatetimeIndex, episodes: pd.DataFrame, horizon: int = HORIZON,
                  start_col: str = "start", label_end=None) -> pd.DataFrame:
    """Daily `in_episode`, `target` (an episode starts in (t, t+horizon]) and `eligible`.

    `eligible` is 1 on days that should be used to train or score an early-warning
    model: outside every episode, and with the whole horizon on or before
    `label_end` (the last day the stress labels are known; default: last calendar day).
    """
    calendar = pd.DatetimeIndex(calendar)
    n = len(calendar)
    in_ep = np.zeros(n, dtype=bool)
    target = np.zeros(n, dtype=bool)
    starts = _positions(calendar, episodes[start_col])
    ends = _positions(calendar, episodes["end"])
    for s, e in zip(starts, ends):
        in_ep[min(s, n):min(e + 1, n)] = True
        target[max(0, s - horizon):min(s, n)] = True
    last = n - 1 if label_end is None else min(n - 1, calendar.searchsorted(pd.Timestamp(label_end), side="right") - 1)
    eligible = ~in_ep & (np.arange(n) + horizon <= last)
    return pd.DataFrame({"in_episode": in_ep.astype(int), "target": target.astype(int),
                         "eligible": eligible.astype(int)}, index=calendar)


def alert_events(flag: pd.Series, merge_gap: int = MERGE_GAP) -> pd.DataFrame:
    """Group flagged days into events. Gaps shorter than `merge_gap` quiet days are bridged."""
    f = flag.fillna(0).astype(bool)
    days = f.index[f.to_numpy()]
    if len(days) == 0:
        return pd.DataFrame(columns=["start", "end", "n_flag_days"])
    pos = f.index.get_indexer(days)
    groups = np.r_[0, np.cumsum(np.diff(pos) > merge_gap)]
    ev = pd.DataFrame({"day": days, "g": groups}).groupby("g")["day"].agg(["min", "max", "count"])
    return ev.rename(columns={"min": "start", "max": "end", "count": "n_flag_days"}).reset_index(drop=True)


def lead_times(flag: pd.Series, episodes: pd.DataFrame, start_col: str = "start",
               lookback: int = LOOKBACK, late: int = LATE) -> pd.DataFrame:
    """For each episode, the first flagged day from `lookback` days before its start to `late` days after.

    lead_days > 0: warned that many trading days early. lead_days <= 0: first
    flag came on or after the start (late). NaN: missed. The window never
    reaches back past the end of the previous episode. `already_on` is True
    when the flag was on at the first day of the window, which means the lead
    time is capped by the window, not a clean early warning.
    """
    cal = flag.index
    f = flag.reindex(cal).fillna(0).astype(bool).to_numpy()
    rows = []
    prev_end = -1
    for _, ep in episodes.iterrows():
        s = _positions(cal, [ep[start_col]])[0]
        lo = max(0, s - lookback, prev_end + 1)
        hi = min(len(cal) - 1, s + late)
        hits = np.flatnonzero(f[lo:hi + 1]) + lo if lo <= hi else np.array([], dtype=int)
        first = hits[0] if len(hits) else None
        rows.append({
            "episode_id": ep.get("episode_id"),
            "name": ep.get("name"),
            "start": cal[s] if s < len(cal) else pd.NaT,
            "window_start": cal[lo] if lo < len(cal) else pd.NaT,
            "first_alert": cal[first] if first is not None else pd.NaT,
            "lead_days": float(s - first) if first is not None else np.nan,
            "already_on": bool(first is not None and first == lo and lo > 0 and f[lo - 1]),
            "has_data": bool(lo <= hi and flag.reindex(cal[lo:hi + 1]).notna().any()),
        })
        prev_end = _positions(cal, [ep["end"]])[0]
    out = pd.DataFrame(rows)
    out["result"] = np.select(
        [~out.has_data, out.lead_days.isna(), out.lead_days > 0],
        ["no data", "missed", "early"], default="late")
    return out


def score_events(flag: pd.Series, episodes: pd.DataFrame, start_col: str = "start",
                 lookback: int = LOOKBACK, merge_gap: int = MERGE_GAP) -> pd.DataFrame:
    """Label each alert event: 'warning' if an episode starts within `lookback` days after
    the event starts, 'during stress' if it starts inside an episode, else 'false alarm'."""
    ev = alert_events(flag, merge_gap)
    cal = flag.index
    masks = episode_masks(cal, episodes, horizon=lookback, start_col=start_col)
    kind = []
    for s in ev["start"]:
        if masks.at[s, "in_episode"]:
            kind.append("during stress")
        elif masks.at[s, "target"]:
            kind.append("warning")
        else:
            kind.append("false alarm")
    ev["kind"] = kind
    return ev


def day_metrics(flag: pd.Series, masks: pd.DataFrame) -> dict:
    """Day-level precision, recall and false-alarm rate on eligible days with a flag value."""
    d = pd.concat([flag.rename("flag"), masks], axis=1, join="inner")
    d = d[d.eligible.eq(1) & d.flag.notna()]
    tp = int(((d.flag == 1) & (d.target == 1)).sum())
    fp = int(((d.flag == 1) & (d.target == 0)).sum())
    fn = int(((d.flag == 0) & (d.target == 1)).sum())
    tn = int(((d.flag == 0) & (d.target == 0)).sum())
    return {
        "days": len(d), "flag_rate": (tp + fp) / len(d) if len(d) else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
    }


def summarize_signal(name: str, flag: pd.Series, episodes: pd.DataFrame,
                     start_col: str = "start", horizon: int = HORIZON,
                     lookback: int = LOOKBACK) -> dict:
    """One row comparing a signal: episodes warned, median lead, false alarms per year, day metrics."""
    lt = lead_times(flag, episodes, start_col, lookback)
    lt = lt[lt.has_data]
    ev = score_events(flag, episodes, start_col, lookback)
    covered = flag.dropna()
    years = (covered.index[-1] - covered.index[0]).days / 365.25 if len(covered) > 1 else np.nan
    m = day_metrics(flag, episode_masks(flag.index, episodes, horizon, start_col))
    return {
        "signal": name, "start_col": start_col,
        "episodes_scored": len(lt),
        "warned_early": int((lt.result == "early").sum()),
        "late": int((lt.result == "late").sum()),
        "missed": int((lt.result == "missed").sum()),
        "median_lead_days": lt.loc[lt.result == "early", "lead_days"].median(),
        "alert_events": len(ev),
        "false_alarm_events": int((ev.kind == "false alarm").sum()),
        "false_alarms_per_year": (ev.kind == "false alarm").sum() / years if years else np.nan,
        **{f"day_{k}": v for k, v in m.items()},
    }


def walk_forward_classifier(data: pd.DataFrame, features: list[str], target: str = "target",
                            eligible: str = "eligible", horizon: int = HORIZON,
                            first_test_year: int = 2010, alert_quantile: float = 0.90,
                            C: float = 1.0) -> pd.DataFrame:
    """Logistic regression refitted at the start of every calendar year.

    Training uses only eligible rows whose target is fully known before the
    refit date: row t is used only if t + horizon trading days < refit date
    (purge). Rows with missing features are skipped. The alert threshold is the
    `alert_quantile` of predicted probabilities on the training rows, so it is
    also fixed before the test year starts.

    Returns one row per eligible test day: prob, alert, threshold, fold year, train size.
    """
    d = data.copy()
    pos = pd.Series(np.arange(len(d)), index=d.index)
    preds = []
    for year in range(first_test_year, d.index.max().year + 1):
        origin = d.index[d.index.year == year]
        if len(origin) == 0:
            continue
        origin_pos = pos[origin[0]]
        train = d[(pos + horizon < origin_pos) & d[eligible].eq(1)].dropna(subset=features + [target])
        test = d.loc[origin][d.loc[origin, eligible].eq(1)].dropna(subset=features)
        if train[target].nunique() < 2 or test.empty:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
        model.fit(train[features], train[target].astype(int))
        thr = float(np.quantile(model.predict_proba(train[features])[:, 1], alert_quantile))
        p = model.predict_proba(test[features])[:, 1]
        preds.append(pd.DataFrame({"prob": p, "alert": (p > thr).astype(int), "threshold": thr,
                                   "fold": year, "train_n": len(train),
                                   "train_last": train.index.max(), target: test[target]},
                                  index=test.index))
    return pd.concat(preds) if preds else pd.DataFrame()


def classifier_metrics(pred: pd.DataFrame, target: str = "target") -> dict:
    y = pred[target].astype(int)
    two = y.nunique() == 2
    tp = int(((pred.alert == 1) & (y == 1)).sum())
    fp = int(((pred.alert == 1) & (y == 0)).sum())
    fn = int(((pred.alert == 0) & (y == 1)).sum())
    tn = int(((pred.alert == 0) & (y == 0)).sum())
    return {
        "test_days": len(pred), "positive_days": int(y.sum()),
        "roc_auc": roc_auc_score(y, pred.prob) if two else np.nan,
        "pr_auc": average_precision_score(y, pred.prob) if two else np.nan,
        "base_rate": y.mean(),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
    }
