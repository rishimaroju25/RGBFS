# Week 3 code reviews

**Reviewer:** Salah · **Date:** 2026-11-13 · **Repo state:** `main` at commit `d02da90`

Files reviewed:

- Brandon: `05_network_analysis.ipynb`
- Grace: `notebooks/06_clustering_analysis.ipynb`, `notebooks/07_clustering_over_time.ipynb`
- Fatima: `notebooks/04_sentiment_full_corpus.ipynb`

Issues are listed most important first.

## Brandon: network code

1. **Still the Week 2 prototype.** The notebook builds one network, for the 60-day window ending 2008-10-15. Week 3's goal is network features across the full history, and there are none yet.
2. **The 0.55 threshold makes stress periods one blob.** On 2008-10-15, 63% of all stock pairs are linked, so the network mostly repeats the average correlation already in `features/connectedness_60d.parquet`. `src/clustering.py` documents this and uses a fixed top 5% of links instead. Pick one network definition and use it everywhere. Salah's Week 4 centrality metrics depend on this choice.
3. **Density is computed after removing isolated stocks.** The number of connected stocks changes between windows, so density is not comparable over time. Compute it over all stocks in the window.
4. **Location and paths.** The notebook is in the repo root and reads `data/processed/...` with a relative path, so it breaks when run from `notebooks/`. Move it to `notebooks/` and use the same `ROOT` setup as the other notebooks.

## Grace: clustering code

1. **`07_clustering_over_time` uses the old setting** (|corr| > 0.55, greedy modularity). `notebooks/08_cluster_stability.ipynb` showed that this setting mostly tracks density: rank correlation −0.92 with the number of communities and −0.95 with modularity. Its clusters also reshuffle between windows. The shared `features/clusters_market_adjusted_20d.parquet` uses the newer setting, so the team now has two conflicting cluster tables. Rerun 07 with `src.clustering.RECOMMENDED`.
2. **Graph code is copied into each notebook** instead of importing `src/clustering.py`, so fixes don't carry over. Import the shared functions.
3. **`sector_mapping.csv` is not in the repo or the shared Drive**, so nobody else can rerun the sector comparison. It covers 351 of 977 stocks and applies one current-day sector label to every date (noted in the notebook). Upload it to the shared Drive and document its source in `docs/data_dictionary.md`.
4. **`density` in the shared cluster table is about 0.05 in every window** by design (top 5% of links). It carries no information and should not be used as a model feature.

## Fatima: NLP code

1. **The Apple tone change is an artifact.** Apple's 10-Qs have 1 Risk Factors paragraph each (they refer to the 10-K), then the 10-K has 109. "Tone change" jumps to −0.55 just from switching document type. Compare like with like (10-K to the previous 10-K, 10-Q to the previous 10-Q), or require a minimum paragraph count.
2. **Risk Factors are still scored raw.** 829 of 1,211 Risk Factors paragraphs come out negative. The Week 2 spot-check fix, scoring only new or changed risk paragraphs, was not applied (`docs/week2_spotcheckresults.md`).
3. **Coverage is 5 tech companies in 2022 only.** That is one year of a 21-year history, so sentiment features can't be tested against 12 of the 13 stress episodes. This is a team decision, not only a code fix.
4. **Done well:** Risk Factors and MD&A are split, paragraphs under 15 words are dropped, and features are delayed one trading day to avoid lookahead.

## For the team

- Decide which network definition is official (threshold vs top 5% of links) before Week 4 network features are built on it.
- Decide which cluster feature table is official: Grace's `notebooks/outputs/grace_week3/cluster_features.csv` or `features/clusters_market_adjusted_20d.parquet`.
- Decide whether sentiment coverage can be extended beyond 2022 before Week 5 modeling.
