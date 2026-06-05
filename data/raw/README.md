# Raw source files (not committed)

These vendor files are **git-ignored** and not redistributed (some, e.g. the Bloomberg
exports, are licensed). To rebuild the dataset from scratch with `src/regression_analysis.py`,
download them from the sources in the main README (§2 / acknowledgements) and drop them here:

| File | Series | Source |
|---|---|---|
| `grid1_tsx3lw0h.xlsx` | USD/INR monthly (INR REGN) | Bloomberg |
| `brent_crude_price.xlsx` | Brent crude (CO1) | Bloomberg |
| `gold_price.xlsx` | Gold (XAU) | Bloomberg |
| `monthly_vix.xlsx` | VIX | Bloomberg |
| `india_10_year_bond_yields.xlsx` | India 10Y (GIND10YR) | Bloomberg |
| `us_10_year_bond_yields.xlsx` | US 10Y (USGG10YR) | Bloomberg |
| `India_Trade_balance.xlsx` | Merch. trade balance (INMTBAL$) | Bloomberg |
| `India_Forex_reserves.xlsx` | FX reserves | Bloomberg |
| `CPIndex_Jan13ToDec25.xls` | India CPI General (base 2012) | MOSPI |
| `CPIndex_Jan11ToDec14.xls` | India CPI General (base 2010) | MOSPI |
| `CPIAUCSL.xlsx` | US CPI all-items | FRED / BLS |
| `FPI_Monthly_Totals_20022026.xlsx` | FPI net flows (USD mn) | NSDL |
| `DTWEXBGS.xlsx` | Broad US dollar index | FRED |
| `data_gpr_daily_recent_1.xls` | Geopolitical Risk index | Caldara–Iacoviello |

The derived monthly series are embedded in the notebook and `usdinr_regression_colab.py`,
so the analysis is fully reproducible **without** these raw files.
