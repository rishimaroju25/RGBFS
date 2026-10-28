
# Financial News Data Source Comparison

## Week 1 – Market Stress Early-Warning System

### Objective

The purpose of this document is to evaluate financial news data sources for the Market Stress Early-Warning System for U.S. equity markets.

The comparison focuses on historical data availability for 2022, API access limitations, cost, and suitability for financial sentiment analysis.


### News Data Source Comparison

| Data Source | Historical 2022 Data | Free Access | Limitations | Suitability |
|---|---|---|---|---|
| NewsAPI | Not available on the free plan | Limited | Free plan has 100 requests per day, restricted historical access, and no full article text | Low |
| GDELT | Historical coverage includes 2022 | Yes | Article metadata and links are available, but full article text is not guaranteed | High for initial exploration |
| Alpha Vantage | Historical date filtering supported | Limited | News Sentiment endpoint requires premium access; free API usage is restricted | Medium if paid access is available |

### Initial Findings

GDELT appears to be the most practical option for exploring historical financial news during the 2022 market stress period because it provides access to historical news coverage.

NewsAPI's free plan is not suitable for retrieving news from 2022 because of its historical access restrictions.

Alpha Vantage provides financial news sentiment functionality, but access and pricing limitations make it less practical for a project relying on free data.

These findings are preliminary. Historical query results, coverage quality, and access restrictions should be verified before selecting a final news source.



### References and Official Documentation

1. **NewsAPI – Pricing and Access Limits**
   https://newsapi.org/pricing

2. **GDELT – DOC 2.0 API Documentation**
   https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

3. **GDELT – Historical Search Coverage**
   https://blog.gdeltproject.org/doc-2-0-updates-1-5-year-searching-and-updated-mobile-interface/

4. **Alpha Vantage – API Documentation**
   https://www.alphavantage.co/documentation/



### GDELT API Access Test

A test request was made to the GDELT DOC API to retrieve news articles related to the stock market from June 1–7, 2022.

**Test Results:**
- Requested period: June 1–7, 2022
- Search term: stock market
- API response: Rate-limit warning
- Reported limit: One request every 5 seconds
- Retry result: Same rate-limit warning
- Historical article retrieval: Not verified

**Observation:**

The GDELT API did not return article data during testing because the requests were rate-limited.

This test does not establish whether historical 2022 articles are available through the endpoint.

**Conclusion:**

GDELT remains a potential historical news source, but its access restrictions and historical query capabilities require further investigation before it can be recommended for the project.


### Final Text Source Recommendation

For the initial version of the Market Stress Early-Warning System, SEC EDGAR filings are recommended as the primary source of financial text data.

The first pass retrieved 2022 Form 10-K and 10-Q filings for five large technology companies (NVIDIA, Apple, Alphabet, Microsoft and Amazon) from EDGAR without access problems. Its counts were not final: Microsoft's paragraph extraction failed with 404 errors (the download URL used the filing agent's ID instead of Microsoft's), the Alphabet counts disagreed between steps, and the text still contained XBRL codes and cover-page checkboxes. Those issues are fixed in `src/sec_filings.py`; `notebooks/02_sec_filings.ipynb` reports the corrected filing and paragraph counts.

SEC filings provide an accessible and reproducible starting point for financial text analysis.

Financial news could improve the system by capturing market developments more frequently. However, the evaluated news providers have access limitations, and the GDELT historical API test did not successfully return articles.

Therefore, the recommended approach is:

- **Week 1 prototype:** Use SEC filings as the primary text source.
- **Future development:** Investigate GDELT historical datasets or other accessible financial-news archives.
- **Long-term goal:** Combine SEC filings with financial news if reliable historical coverage can be obtained.

This recommendation prioritizes data availability and reproducibility while recognizing that periodic SEC filings alone are not sufficient for real-time market stress detection.

