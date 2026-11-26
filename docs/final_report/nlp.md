# Natural Language Processing (NLP) and Sentiment Analysis

## 1. Data Collection and Preprocessing

The NLP component of our Market Stress Early-Warning System uses financial disclosures from SEC EDGAR to examine whether changes in corporate sentiment can provide early indications of market stress.

The initial prototype focused on five companies: Apple, Amazon, Alphabet, Microsoft, and NVIDIA during 2022. The pipeline was later expanded to support 50 S&P 500 companies across all 11 sectors, covering the period from 2006 to 2026.

We focused on two sections of annual 10-K and quarterly 10-Q filings:

- Management's Discussion and Analysis (MD&A)
- Risk Factors

Paragraphs containing fewer than 15 words were excluded. Repeated Risk Factors paragraphs were removed using a 90% text similarity threshold. In the expanded pipeline, paragraphs are compared with the company's previous four filings to reduce processing time while retaining relevant new disclosures.

## 2. FinBERT Sentiment Analysis

We used FinBERT, a language model designed for financial text, to classify each retained paragraph into three sentiment categories:

- Positive: +1
- Neutral: 0
- Negative: -1

The paragraph scores were aggregated by company, filing, and section to calculate average sentiment and the proportion of negative paragraphs. Tone change was calculated by comparing average sentiment with the previous filing of the same company, section, and filing type.

The expanded pipeline also introduced a reusable sentiment scoring cache to improve processing efficiency and avoid repeatedly scoring identical text.

## 3. Daily Sentiment Features

Filing dates were aligned with trading days, and sentiment features were carried forward until the next applicable filing. An additional one-trading-day delay was applied to reduce look-ahead bias.

The expanded pipeline also calculates market-wide sentiment indicators by averaging available company-level features. Filing information older than 130 trading days is excluded from these market-wide calculations.

These features allow sentiment information to be combined with market indicators, network features, and volatility measurements.

## 4. Sentiment Analysis Results

The initial 2022 sentiment dataset produced the following results:

| Metric | Result |
|---|---:|
| Companies analyzed | 5 |
| Trading days | 251 |
| Filing sections | 2 |
| Total daily records | 2,510 |
| Available sentiment scores | 2,052 |
| Missing sentiment scores | 458 |
| Sentiment data coverage | 81.8% |
| Unique filing-section observations | 37 |
| Available filing-to-filing sentiment changes | 27 |

Risk Factors generally contained more negative sentiment than MD&A sections. Risk Factors also showed greater variability in filing-to-filing sentiment changes.

Although the dataset contained 2,510 daily records, many observations represented carried-forward filing scores rather than newly released information. Therefore, anomaly detection was performed using unique filing-section observations to avoid overrepresenting repeated daily values.

## 5. Sentiment Anomaly Detection

Sentiment anomaly detection was performed using modified Z-scores to identify unusual changes in corporate filing tone. Separate baselines were used for MD&A and Risk Factors because their sentiment distributions and variability differed.

A primary absolute modified Z-score threshold of 3.5 was selected.

Of the 37 unique filing-section observations, 27 contained usable sentiment changes. No observations exceeded the primary anomaly threshold.

The largest absolute modified Z-score was observed in NVIDIA's MD&A filing on August 31, 2022, with a score of approximately -3.29 and a sentiment change of -0.2576.

Although this represented a noticeable decline in sentiment, it did not exceed the primary anomaly threshold.

### Threshold Sensitivity

| Modified Z-score threshold | Detected anomalies |
|---|---:|
| 1.5 | 4 |
| 2.0 | 2 |
| 2.5 | 1 |
| 3.0 | 1 |
| 3.5 | 0 |
| 4.0 | 0 |

Lower thresholds identified more potential anomalies. However, without ground-truth sentiment anomaly labels, these additional flags cannot be confirmed as accurate early-warning signals.

The threshold of 3.5 was retained as the primary rule, while lower thresholds were treated as exploratory alternatives.

## 6. Cluster-Level Sentiment Aggregation

To integrate NLP features with network analysis, sentiment data was merged with network community assignments using trading dates and stock tickers.

The analysis loaded 2,510 sentiment records and 95,601 cluster records. Sentiment scores were aggregated by date and community to calculate mean sentiment and paragraph counts at the cluster level.

The initial results included:

| Date | Cluster ID | Mean sentiment |
|---|---:|---:|
| January 31, 2022 | 2 | -0.1378 |
| March 1, 2022 | 1 | -0.1870 |
| March 29, 2022 | 1 | -0.2164 |
| April 27, 2022 | 1 | -0.2395 |
| May 25, 2022 | 1 | -0.3475 |

The observations assigned to cluster 1 showed increasingly negative mean sentiment between March and May 2022. However, changes in community membership may also have influenced these results.

This approach demonstrates how company-level financial sentiment can be integrated with network-based features to examine collective financial conditions.

## 7. Dashboard Visualization

An interactive sentiment panel was developed using Streamlit to make the NLP findings accessible through the project's dashboard.

The panel includes:

- Company-level sentiment trends.
- Filing-section filters.
- Sentiment and tone-change visualizations.
- Data coverage metrics.
- Historical sentiment observations.

The updated dashboard also supports market-wide sentiment visualization and company-level exploration when the expanded sentiment dataset is available.

## 8. Expanded Sentiment Pipeline

The original FinBERT methodology was subsequently extended to support 50 large S&P 500 companies across all 11 sectors from 2006 to 2026.

The expanded pipeline includes incremental SEC filing downloads, reusable FinBERT scores, improved repeated-text filtering, filing-level sentiment aggregation, and market-wide daily sentiment features.

The expanded sentiment features are also designed to support the project's anomaly detection models and early-warning lead-time analysis.

However, the completed numerical results from the expanded corpus were not available for verification at the time of this write-up. Therefore, the numerical sentiment findings reported above are based on the original five-company 2022 dataset.

## 9. Limitations

Several limitations affect the interpretation of the NLP results:

- SEC filings are released periodically rather than daily, limiting the frequency of new sentiment information.
- Carried-forward sentiment scores do not represent independent daily observations.
- The original five-company dataset is concentrated in large technology companies and may not represent the broader market.
- The expanded selection of large surviving companies may introduce survivorship bias.
- Missing disclosures and repeated Risk Factors can affect sentiment coverage.
- FinBERT may misclassify complex financial language.
- The limited number of filing-level observations reduces the reliability of anomaly detection.
- The absence of detected anomalies at the primary threshold does not establish that unusual sentiment changes did not occur.

Although sentiment provides additional information about corporate financial conditions, the available results do not establish that it consistently provides earlier warnings of market stress than traditional financial indicators.

Further evaluation using the expanded historical corpus and out-of-sample testing is necessary to determine its predictive value.

## 10. Supporting Project Files

- `notebooks/02_sec_filings.ipynb`
- `notebooks/03_sentiment_prototype.ipynb`
- `notebooks/04_sentiment_2022_sample_original.ipynb`
- `notebooks/04_sentiment_full_corpus.ipynb`
- `notebooks/08_sentiment_anomaly_detection.ipynb`
- `notebooks/09_cluster_sentiment_aggregation.ipynb`
- `notebooks/sentiment_spot_check.ipynb`
- `src/sentiment.py`
- `scripts/build_sentiment.py`
- `tests/test_sentiment.py`
- `dashboard/panels/sentiment_panel.py`