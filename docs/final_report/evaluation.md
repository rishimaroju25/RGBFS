# Evaluation and Backtesting

**Author:** Fatima 

## 1. Evaluation Objective

The objective of the evaluation was to determine whether combining market indicators, network features, sentiment information, and anomaly detection methods improves the identification and early prediction of U.S. equity market stress compared with traditional financial indicators.

The evaluation considered classification performance, robustness, early-warning lead times, false alarms, and volatility forecasting accuracy.

## 2. Evaluation Metrics

The following metrics were used to evaluate model performance:

- **Accuracy:** Proportion of correctly classified observations.
- **Precision:** Proportion of predicted stress observations that were actual stress observations.
- **Recall:** Proportion of actual stress observations correctly identified.
- **F1-score:** Balance between precision and recall.
- **ROC-AUC:** Ability to distinguish between stress and non-stress observations.
- **PR-AUC:** Precision-recall performance, particularly useful for imbalanced datasets.
- **False Alarm Rate:** Proportion of non-stress observations incorrectly identified as stress.
- **Lead Time:** Number of days an alert occurs before the beginning of a stress episode.
- **RMSE:** Measures the magnitude of volatility forecasting errors.

Recall, PR-AUC, lead time, and false alarms were particularly important because market stress events are relatively uncommon.

## 3. Initial Model Evaluation

The initial evaluation compared a VIX-only baseline model with a combined model incorporating additional market indicators.

| Metric | VIX-only baseline | Combined model |
|---|---:|---:|
| Accuracy | 0.9196 | 0.8876 |
| Precision | 0.5075 | 0.4145 |
| Recall | 0.7556 | 0.8889 |
| F1-score | 0.6071 | 0.5654 |
| ROC-AUC | 0.9479 | 0.9577 |
| PR-AUC | 0.7241 | 0.7506 |
| False alarm rate | 0.0657 | 0.1125 |

The combined model improved recall from 75.56% to 88.89%, indicating that it identified more stress observations.

ROC-AUC and PR-AUC also improved. However, precision, accuracy, and F1-score decreased, while the false alarm rate increased.

These results demonstrate a trade-off between detecting more stress observations and generating additional false alarms.

The initial evaluation primarily measured retrospective stress classification rather than genuine advance prediction.

## 4. Sentiment Feature Evaluation

An exploratory evaluation examined whether adding FinBERT sentiment features improved market stress classification.

The original sentiment dataset covered five companies during 2022. After merging sentiment features with market indicators and removing incomplete observations, 172 observations remained.

The dataset was divided chronologically into 137 training observations and 35 testing observations.

Adding sentiment features produced modest improvements in recall and PR-AUC in this experiment. However, the small testing sample and limited company coverage prevent strong conclusions about the predictive value of sentiment.

The expanded 50-company sentiment pipeline provides a foundation for broader evaluation, but its numerical results were not available for verification at the time of this write-up.

## 5. Robustness Checks

Robustness checks were conducted using different realized volatility windows and prediction horizons to examine the sensitivity of model performance.

The volatility windows included:

- 10 trading days.
- 21 trading days.
- 42 trading days.

Stress labels showed approximately 95% agreement across the tested volatility-window definitions.

For the 21-day volatility-window experiment, the following results were obtained:

| Metric | VIX-only model | Combined model |
|---|---:|---:|
| Recall | 0.8125 | 0.8906 |
| F1-score | 0.6209 | 0.6404 |
| ROC-AUC | 0.9451 | 0.9615 |
| PR-AUC | 0.6953 | 0.7590 |
| False alarm rate | 0.0846 | 0.0936 |

The combined model generally achieved higher recall, F1-score, and PR-AUC across the tested volatility windows.

Prediction horizons of 1, 5, and 10 trading days were also examined. The combined model generally achieved higher recall and PR-AUC, but longer prediction horizons introduced additional false alarms and reduced F1-score relative to the baseline in some experiments.

These findings suggest that additional market indicators may improve stress classification, although performance depends on the stress definition and prediction horizon.

The robustness experiments used threshold-based proxy stress labels rather than the original episode-adjusted labels. Therefore, these results should be interpreted as sensitivity checks rather than directly equivalent evaluations.

## 6. Anomaly Detection Model Evaluation

Additional experiments evaluated whether network anomaly features improved early-warning classification.

Three model configurations were compared:

- Baseline model using VIX and volatility.
- Baseline model with Z-score anomaly features.
- Baseline model with Z-score and Isolation Forest features.

The evaluation included 3,573 test days, of which 180 were positive observations. The positive class represented approximately 5.04% of the testing dataset.

| Metric | Baseline | + Z-scores | + Z-scores and Isolation Forest |
|---|---:|---:|---:|
| ROC-AUC | 0.6243 | 0.6016 | 0.4910 |
| PR-AUC | 0.0897 | 0.0639 | 0.0451 |
| Precision | 0.1404 | 0.0625 | 0.0506 |
| Recall | 0.1833 | 0.1278 | 0.1778 |
| False alarm rate | 0.0595 | 0.1017 | 0.1768 |

The baseline model achieved the strongest performance across the reported metrics.

Adding Z-score and Isolation Forest features did not improve predictive performance in this experiment. The false alarm rate increased substantially when additional anomaly features were included.

These findings demonstrate that increasing model complexity does not necessarily improve early-warning performance.

## 7. Early-Warning Lead-Time Analysis

Network anomaly signals were compared with traditional financial stress indicators to determine how early they identified historical market stress episodes.

Using the trailing stress episode definition, the following results were obtained:

| Signal | Early warnings | Median lead days | False alarms per year |
|---|---:|---:|---:|
| Network anomaly (combined) | 12/13 | 24.5 | 0.67 |
| Network Z-score only | 12/13 | 24.5 | 0.67 |
| VIX > 25 | 13/13 | 18.0 | 1.38 |
| VIX Z-score >= 2 | 11/13 | 23.0 | 1.24 |
| STLFSI4 > 0 (lagged) | 10/13 | 35.5 | 1.19 |
| STLFSI4 > 1 (lagged) | 2/13 | 60.0 | 0.29 |
| Combined model alert | 5/9 | 37.0 | 1.97 |
| Baseline model alert | 5/9 | 38.0 | 1.08 |

Network anomaly detection provided early warnings for 12 of the 13 evaluated stress episodes, representing approximately 92.3%.

Compared with the VIX > 25 threshold, the network anomaly signal provided a longer median lead time and approximately 52% fewer false alarm events per year.

However, VIX > 25 successfully warned before all 13 episodes.

The combined network anomaly signal produced identical results to the Z-score-only signal, indicating that Isolation Forest did not provide additional warning benefits in this configuration.

The combined model alert also generated more false alarms than the baseline model alert.

These results suggest that network indicators can provide useful supplementary warnings, although no single signal performed best across every evaluation criterion.

The model-based alerts were evaluated on fewer episodes than the individual network and benchmark signals, so their results should not be compared without considering differences in evaluation coverage.

## 8. Walk-Forward Validation

Walk-forward validation was conducted to evaluate performance under more realistic chronological testing conditions.

The analysis also examined the effect of accounting for publication delays in FRED financial indicators.

| Evaluation setup | Model | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Original evaluation | VIX only | 0.9479 | 0.7241 |
| Original evaluation | Combined | 0.9577 | 0.7506 |
| FRED publication lag | VIX only | 0.9479 | 0.7241 |
| FRED publication lag | Combined | 0.9512 | 0.7206 |
| Walk-forward validation | VIX only | 0.9434 | 0.7607 |
| Walk-forward validation | Combined | 0.9180 | 0.7112 |
| Early-warning walk-forward | VIX only | 0.5711 | 0.0677 |
| Early-warning walk-forward | Combined | 0.4902 | 0.0662 |

The original evaluation suggested that combining market indicators improved classification performance.

However, after accounting for publication delays and applying walk-forward validation, the improvement became less consistent.

The VIX-only model achieved stronger performance than the combined model under both walk-forward configurations.

Early-warning performance was substantially weaker than retrospective stress classification performance. This highlights the difficulty of predicting future market stress rather than identifying stress that is already occurring.

## 9. Volatility Forecasting Evaluation

Three models were evaluated for 20-day volatility forecasting:

- Persistence baseline.
- HAR-style volatility model.
- Volatility model with network cluster features.

Forecasting accuracy was compared across four historical market periods using RMSE, where lower values indicate better performance.

| Market period | Persistence | HAR-style | Volatility + Clusters |
|---|---:|---:|---:|
| 2010–2012: Euro crisis | 0.0722 | 0.0659 | 0.0635 |
| 2013–2019: Bull market | 0.0610 | 0.0543 | 0.0748 |
| 2020–2021: COVID-19 | 0.1696 | 0.1414 | 0.1479 |
| 2022–2026: Rate hikes and tariffs | 0.0740 | 0.0651 | 0.0663 |

The HAR-style model outperformed the persistence baseline across all four market periods and achieved the lowest forecasting error in three of the four periods.

Adding network cluster features improved forecasting accuracy during the 2010–2012 crisis period. However, the cluster-enhanced model did not consistently outperform the HAR-style model during other periods.

These results indicate that network features may provide forecasting benefits under certain market conditions, but their effectiveness varies across market regimes.

## 10. Historical Stress Episode Evaluation

The project evaluated warning performance across multiple historical market stress episodes.

Some indicators provided advance warnings during the 2022 bear market and the 2025 tariff-related selloff. However, several other episodes, including the COVID-19 market crash and the 2018 selloffs, were not successfully predicted in the separate forecasting-based warning evaluation.

The variation in results demonstrates that market stress events can develop differently depending on economic conditions, market structure, and the speed of emerging shocks.

A combination of indicators may provide complementary information, but the results do not establish that combining additional features consistently improves advance prediction.

## 11. Evaluation Limitations

Several limitations should be considered when interpreting the evaluation results:

- Market stress observations represent a small proportion of the historical dataset, creating class imbalance.
- The original centered realized volatility stress labels incorporate future observations, limiting their interpretation as real-time warning targets.
- Publication delays in financial stress indicators can affect the availability of information.
- Retrospective stress classification is different from predicting future stress episodes.
- The original sentiment evaluation was limited to five companies and one year of data.
- Different stress definitions and evaluation periods affect the comparability of results.
- Additional model features can increase false alarms without improving predictive performance.
- Network and forecasting performance may vary across market regimes.

Chronological validation, trailing stress definitions, and publication-lag adjustments provide more realistic evaluation conditions, although they do not eliminate all limitations.

## 12. Evaluation Summary

The evaluation produced mixed but informative results.

The initial combined market-indicator model improved recall and ranking performance compared with the VIX-only baseline. However, these improvements were not consistently maintained under more realistic walk-forward and early-warning evaluations.

Network anomaly signals demonstrated promising episode-level warning performance, providing advance warnings for 12 of 13 stress episodes under the trailing definition while generating fewer false alarms per year than the VIX > 25 threshold.

The HAR-style volatility model also demonstrated consistent forecasting improvements over the persistence baseline.

Overall, the findings suggest that network features, sentiment information, and volatility forecasting can provide complementary perspectives on market conditions. However, the available evidence does not establish that a combined system consistently predicts market stress earlier or more accurately than traditional financial indicators.

## Supporting Project Files

- `notebooks/10_evaluation_backtesting.ipynb`
- `notebooks/11_robustness_checks.ipynb`
- `notebooks/outputs/rishi_week5/model_metrics.csv`
- `notebooks/outputs/rishi_week6/signal_summary.csv`
- `walk_forward_comparison.csv`
- `forecast_accuracy_by_period.csv`
- `warnings_by_episode.csv`