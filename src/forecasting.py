#!/usr/bin/env python3
"""
One-month-ahead forecasting of USD/INR returns — honest out-of-sample evaluation.

Key principle: a forecast may use ONLY information available before the month being
predicted, so every predictor is lagged one month. Models are compared OUT-OF-SAMPLE
(expanding-window walk-forward) against the RANDOM-WALK benchmark (next-month return = 0,
i.e. "no change in the level"), the standard exchange-rate yardstick (Meese-Rogoff, 1983).

Models:
  RW            random walk (predict 0)                      <- benchmark
  RW_drift      historical mean return
  AR1           AR(1) on returns
  ARIMA(1,0,1)  ARMA on returns
  OLS_lag2      return_t ~ Broad_USD_Return_{t-1} + FPI_Flow_{t-1} + return_{t-1}
  OLS_lagAll    return_t ~ all 9 regressors_{t-1} + return_{t-1}
  Ridge         ridge on all lagged features (standardised)
  RandomForest  RF on all lagged features

Metrics (out-of-sample): RMSE, MAE, directional accuracy, RMSE ratio vs RW (Theil),
Diebold-Mariano test vs RW. Also a level-forecast RMSE (in rupees) and a next-month forecast.
"""
import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
DATA, RESULTS, CH = (os.path.join(ROOT, d) for d in ("data", "results", "charts"))
for d in (RESULTS, CH): os.makedirs(d, exist_ok=True)
NAVY="#1f3b57"; RUST="#c1442e"; TEAL="#2a7f7f"; GOLD="#c79a3a"; GREY="#9aa0a6"
plt.rcParams.update({"figure.dpi":120,"font.size":10,"axes.grid":True,"grid.alpha":0.3,
                     "axes.spines.top":False,"axes.spines.right":False})

# ---------------------------------------------------------------- data
data = pd.read_csv(f"{DATA}/regression_data.csv", index_col=0); data.index = pd.PeriodIndex(data.index, freq="M")
lvl = pd.read_csv(f"{DATA}/usdinr_full.csv", index_col=0); lvl.index = pd.PeriodIndex(lvl.index, freq="M")
DEP = "USDINR_Return"
FEATURES = ["Oil_Return","Gold_Return","Inflation_Differential","Interest_Rate_Differential",
            "FPI_Flow","Trade_Balance","FX_Reserves_Change","VIX","Broad_USD_Return"]

# design: target return_t predicted from t-1 information (all predictors lagged 1 month)
df = pd.DataFrame({DEP: data[DEP]})
for f in FEATURES: df[f+"_lag"] = data[f].shift(1)
df["AR1"] = data[DEP].shift(1)
df = df.dropna()
LAGCOLS = [f+"_lag" for f in FEATURES] + ["AR1"]
PARSI   = ["Broad_USD_Return_lag", "FPI_Flow_lag", "AR1"]

INIT = 60                      # initial training window (months) -> OOS starts month 61
idx = df.index
print(f"Design rows: {len(df)} ({idx.min()}..{idx.max()})  |  initial train = {INIT}  |  OOS = {len(df)-INIT} months "
      f"({idx[INIT]}..{idx[-1]})")

def dm_test(err_model, err_bench, h=1):
    """Diebold-Mariano on squared-error loss differential (h=1: no HAC needed). +stat => model worse."""
    d = err_model**2 - err_bench**2
    dbar = d.mean(); T = len(d)
    var = np.var(d, ddof=0) / T
    if var <= 0: return np.nan, np.nan
    stat = dbar / np.sqrt(var)
    return stat, 2*(1-norm.cdf(abs(stat)))

def arima_fc(y, order):
    """One-step ARIMA forecast, warnings suppressed; falls back to the mean if it fails to converge."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(ARIMA(y, order=order).fit().forecast(1)[0])
        except Exception:
            return float(np.mean(y))

models = ["RW","RW_drift","AR1","ARIMA(1,0,1)","OLS_lag2","OLS_lagAll","Ridge","RandomForest"]
preds = {m: [] for m in models}
actual = []

for i in range(INIT, len(df)):
    tr = df.iloc[:i]; te = df.iloc[i]
    yv = tr[DEP].values
    actual.append(te[DEP])
    preds["RW"].append(0.0)
    preds["RW_drift"].append(yv.mean())
    # AR(1) and ARMA on the return series
    preds["AR1"].append(arima_fc(yv, (1,0,0)))
    preds["ARIMA(1,0,1)"].append(arima_fc(yv, (1,0,1)))
    # lagged-predictor OLS
    for name, cols in [("OLS_lag2", PARSI), ("OLS_lagAll", LAGCOLS)]:
        b = sm.OLS(tr[DEP], sm.add_constant(tr[cols])).fit()
        preds[name].append(b.params["const"] + float(np.dot(b.params[cols].values, te[cols].values)))
    # ML on all lagged features (standardise within the training window)
    mu, sd = tr[LAGCOLS].mean(), tr[LAGCOLS].std().replace(0,1)
    Xtr = (tr[LAGCOLS]-mu)/sd; xte = (te[LAGCOLS]-mu)/sd
    preds["Ridge"].append(float(Ridge(alpha=10.0).fit(Xtr, tr[DEP]).predict(xte.values.reshape(1,-1))[0]))
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
    preds["RandomForest"].append(float(rf.fit(tr[LAGCOLS], tr[DEP]).predict(te[LAGCOLS].values.reshape(1,-1))[0]))

actual = np.array(actual)
oos_idx = idx[INIT:]
err = {m: np.array(preds[m]) - actual for m in models}

# ---------------------------------------------------------------- metrics
rmse_rw = np.sqrt(np.mean(err["RW"]**2))
rows = []
for m in models:
    e = err[m]; rmse = np.sqrt(np.mean(e**2)); mae = np.mean(np.abs(e))
    p = np.array(preds[m])
    da = np.mean(np.sign(p) == np.sign(actual)) if np.any(p != 0) else np.nan   # RW (all 0) -> n/a
    dm, dmp = dm_test(e, err["RW"]) if m != "RW" else (np.nan, np.nan)
    rows.append({"model": m, "RMSE": round(rmse,4), "MAE": round(mae,4),
                 "Dir_Acc_%": round(100*da,1) if not np.isnan(da) else np.nan,
                 "RMSE_ratio_vs_RW": round(rmse/rmse_rw,4),
                 "DM_stat_vs_RW": round(dm,3) if not np.isnan(dm) else np.nan,
                 "DM_p_vs_RW": round(dmp,4) if not np.isnan(dmp) else np.nan})
mt = pd.DataFrame(rows)
mt.to_csv(f"{RESULTS}/forecast_metrics.csv", index=False)
pd.DataFrame({"date":[str(p) for p in oos_idx], "actual":actual, **{m:np.array(preds[m]) for m in models}}).to_csv(
    f"{RESULTS}/forecast_oos_predictions.csv", index=False)
print("\n===== OUT-OF-SAMPLE FORECAST METRICS (1-month-ahead) =====")
print(mt.to_string(index=False))
print(f"\nRandom-walk RMSE (benchmark): {rmse_rw:.4f} %   |   base depreciation rate: {np.mean(actual>0)*100:.1f}% of months")

# ---------------------------------------------------------------- level-forecast RMSE (rupees)
L = lvl["USDINR"].reindex(idx).values            # level aligned to design index
Lprev = lvl["USDINR"].reindex(idx - 1).values    # level one month earlier
Lact = L[INIT:]; Lprev_oos = Lprev[INIT:]
lvl_rows=[]
for m in models:
    Lf = Lprev_oos * np.exp(np.array(preds[m])/100.0)
    lvl_rows.append({"model":m,"level_RMSE_INR":round(np.sqrt(np.mean((Lf-Lact)**2)),3)})
lvl_tbl=pd.DataFrame(lvl_rows); lvl_tbl.to_csv(f"{RESULTS}/forecast_level_rmse.csv",index=False)
print("\nLevel-forecast RMSE (INR per USD):")
print(lvl_tbl.to_string(index=False))

# ---------------------------------------------------------------- next-month forecast (full sample)
last = df.index[-1]; nxt = last + 1
full = df
ret_fc = {}
ret_fc["RW"] = 0.0
ret_fc["RW_drift"] = full[DEP].mean()
ret_fc["AR1"] = arima_fc(full[DEP].values, (1,0,0))
b2 = sm.OLS(full[DEP], sm.add_constant(full[PARSI])).fit()
xvals = [data[c.replace("_lag","")].iloc[-1] if c!="AR1" else data[DEP].iloc[-1] for c in PARSI]
ret_fc["OLS_lag2"] = b2.params["const"] + float(np.dot(b2.params[PARSI].values, np.array(xvals)))
L_last = float(lvl["USDINR"].loc[last])   # level at the last in-sample month (forecast origin)
band = rmse_rw  # ~1-sigma monthly forecast error from the benchmark
print(f"\n===== NEXT-MONTH FORECAST ({nxt}) =====")
print(f"  last observed level ({last}): {L_last:.3f} INR/USD")
for m in ["RW","RW_drift","AR1","OLS_lag2"]:
    lf = L_last*np.exp(ret_fc[m]/100.0)
    print(f"  {m:<9} return={ret_fc[m]:+.3f}%  ->  level {lf:.2f}  (~68% band +/-{band:.2f}% => {L_last*np.exp((ret_fc[m]-band)/100):.2f} to {L_last*np.exp((ret_fc[m]+band)/100):.2f})")

# ---------------------------------------------------------------- CHARTS
t = oos_idx.to_timestamp()
best_nonrw = min([m for m in models if m!="RW"], key=lambda m: np.sqrt(np.mean(err[m]**2)))

# fcst1: OOS actual vs predicted (RW reference + AR1 + best)
fig,ax=plt.subplots(figsize=(10,3.8))
ax.bar(t, actual, width=22, color=GREY, alpha=0.55, label="Actual return")
ax.axhline(0, color=NAVY, lw=1.4, label="Random walk (predict 0)")
ax.plot(t, preds["AR1"], color=RUST, lw=1.2, label="AR(1)")
if best_nonrw not in ("AR1",): ax.plot(t, preds[best_nonrw], color=TEAL, lw=1.2, ls="--", label=f"{best_nonrw} (best non-naive)")
ax.set_title("One-month-ahead out-of-sample forecasts vs actual USD/INR return", fontweight="bold")
ax.set_ylabel("% return"); ax.legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(f"{CH}/fcst1_oos_actual_vs_pred.png", bbox_inches="tight"); plt.close()

# fcst2: RMSE ratio vs RW
order = mt.sort_values("RMSE_ratio_vs_RW")
fig,ax=plt.subplots(figsize=(8,4))
cols=[TEAL if r<1 else RUST for r in order["RMSE_ratio_vs_RW"]]
ax.barh(order["model"], order["RMSE_ratio_vs_RW"], color=cols)
ax.axvline(1.0, color=NAVY, ls="--", lw=1.3, label="random walk (=1.0)")
ax.set_title("Out-of-sample RMSE relative to the random walk\n(<1 beats the benchmark)", fontweight="bold")
ax.set_xlabel("RMSE ratio vs RW"); ax.legend(frameon=False)
for i,(m,v) in enumerate(zip(order["model"],order["RMSE_ratio_vs_RW"])): ax.text(v+0.002,i,f"{v:.3f}",va="center",fontsize=8)
fig.tight_layout(); fig.savefig(f"{CH}/fcst2_rmse_ratio_bar.png", bbox_inches="tight"); plt.close()

# fcst3: directional accuracy
da_tbl = mt.dropna(subset=["Dir_Acc_%"]).sort_values("Dir_Acc_%")
fig,ax=plt.subplots(figsize=(8,4))
ax.barh(da_tbl["model"], da_tbl["Dir_Acc_%"], color=NAVY)
ax.axvline(50, color=RUST, ls="--", lw=1.3, label="coin flip (50%)")
ax.set_title("Directional accuracy (% of months sign predicted correctly)", fontweight="bold")
ax.set_xlabel("%"); ax.legend(frameon=False); ax.set_xlim(40, max(62, da_tbl["Dir_Acc_%"].max()+4))
for i,(m,v) in enumerate(zip(da_tbl["model"],da_tbl["Dir_Acc_%"])): ax.text(v+0.3,i,f"{v:.0f}%",va="center",fontsize=8)
fig.tight_layout(); fig.savefig(f"{CH}/fcst3_directional_accuracy_bar.png", bbox_inches="tight"); plt.close()

# fcst4: level forecast path (RW vs best) vs actual
fig,ax=plt.subplots(figsize=(10,3.8))
ax.plot(t, Lact, color=NAVY, lw=1.6, label="Actual level")
ax.plot(t, Lprev_oos*np.exp(np.array(preds["RW"])/100), color=GREY, lw=1.2, ls="--", label="Random-walk forecast")
ax.plot(t, Lprev_oos*np.exp(np.array(preds[best_nonrw])/100), color=RUST, lw=1.2, ls=":", label=f"{best_nonrw} forecast")
ax.set_title("One-month-ahead level forecast vs actual (INR per USD)", fontweight="bold")
ax.set_ylabel("INR per USD"); ax.legend(frameon=False, ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(f"{CH}/fcst4_level_forecast.png", bbox_inches="tight"); plt.close()

json.dump({"rw_rmse":round(rmse_rw,4),"best_nonrw":best_nonrw,
           "best_nonrw_ratio":float(order.iloc[0]["RMSE_ratio_vs_RW"]),
           "best_model_overall":order.iloc[0]["model"]},
          open(f"{RESULTS}/forecast_summary.json","w"), indent=2)
print("\nDONE — forecast metrics, predictions, and 4 charts (fcst1-4) saved.")
