#!/usr/bin/env python3
"""
USD/INR monthly log-return regression — full pipeline.
Builds the master dataset from raw files, runs OLS with diagnostics,
eliminates spurious/insignificant regressors, and saves all artefacts.

Run from the repository root (or from src/); paths are resolved automatically:
    python src/regression_analysis.py
Raw source files are expected in  data/raw/ .
"""
import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey, linear_reset
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller

# ----- portable paths (work from repo root, from src/, or in a notebook) -----
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
RAW     = os.path.join(ROOT, "data", "raw")
DATA    = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
for d in (DATA, RESULTS): os.makedirs(d, exist_ok=True)

MONNAMES = ['January','February','March','April','May','June','July','August','September','October','November','December']
MON = {m:i for i,m in enumerate(MONNAMES,1)}

def _need(fname):
    p = os.path.join(RAW, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing raw file: {p}\n  -> place the 14 source files in {RAW}/")
    return p

# ----------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------
def bbg(fname):
    raw = pd.read_excel(_need(fname), header=None, engine="openpyxl")
    h = raw.index[raw.iloc[:,0]=="Date"][0]
    df = raw.iloc[h+1:].copy(); df.columns = raw.iloc[h].tolist()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    col = "PX_LAST" if pd.to_numeric(df.get("PX_LAST"), errors="coerce").notna().any() else "PX_MID"
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=df["Date"].dt.to_period("M")).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]

def india_cpi():
    t = pd.read_html(_need("CPIndex_Jan13ToDec25.xls"))[0]; t.columns=[str(c).strip() for c in t.columns]
    t["Year"]=pd.to_numeric(t["Year"],errors="coerce").astype("Int64"); t["mo"]=t["Month"].map(MON)
    t=t.dropna(subset=["Year","mo"])
    idx=pd.to_datetime(dict(year=t.Year.astype(int),month=t.mo.astype(int),day=1)).dt.to_period("M")
    return pd.Series(pd.to_numeric(t["Combined"],errors="coerce").values,index=idx.values).sort_index()

def us_cpi():
    d=pd.read_excel(_need("CPIAUCSL.xlsx"),sheet_name="Monthly"); d.columns=["date","cpi"]
    d["date"]=pd.to_datetime(d["date"])
    return pd.Series(pd.to_numeric(d["cpi"],errors="coerce").values,index=d["date"].dt.to_period("M")).sort_index()

def fpi_flows():
    f=pd.read_excel(_need("FPI_Monthly_Totals_20022026.xlsx"),header=2)
    m=f.melt(id_vars=["Year"],value_vars=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
             var_name="mon",value_name="fpi")
    short={"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    m=m[pd.to_numeric(m["Year"],errors="coerce").notna()].copy()
    idx=pd.to_datetime(dict(year=m.Year.astype(int),month=m.mon.map(short),day=1)).dt.to_period("M")
    return pd.Series(pd.to_numeric(m["fpi"],errors="coerce").values,index=idx.values).dropna().sort_index()

def dtwexbgs():
    d=pd.read_excel(_need("DTWEXBGS.xlsx"),sheet_name="Daily"); d.columns=["date","v"]
    d["date"]=pd.to_datetime(d["date"]); d["v"]=pd.to_numeric(d["v"],errors="coerce")
    return d.dropna().set_index("date")["v"].resample("ME").last().to_period("M")

def gpr():
    g=pd.read_excel(_need("data_gpr_daily_recent_1.xls"),engine="xlrd")[["date","GPRD"]]
    g["date"]=pd.to_datetime(g["date"]); g["GPRD"]=pd.to_numeric(g["GPRD"],errors="coerce")
    return g.dropna().set_index("date")["GPRD"].resample("ME").mean().to_period("M")

# ----------------------------------------------------------------------
# BUILD
# ----------------------------------------------------------------------
usdinr   = bbg("grid1_tsx3lw0h.xlsx")
brent    = bbg("brent_crude_price.xlsx")
gold     = bbg("gold_price.xlsx")
vix      = bbg("monthly_vix.xlsx")
ind10    = bbg("india_10_year_bond_yields.xlsx")
us10     = bbg("us_10_year_bond_yields.xlsx")
tbal     = bbg("India_Trade_balance.xlsx")
fxres    = bbg("India_Forex_reserves.xlsx")
icpi     = india_cpi()
ucpi     = us_cpi()
fpi      = fpi_flows()
dxy      = dtwexbgs()
try:                       # GPR is supplementary (not used in the core model); keep optional
    gp = gpr()
except Exception as e:
    print(f"(GPR optional load skipped: {e})")
    gp = pd.Series(dtype=float)

# full USD/INR level history (for the context chart) — saved before truncation
usdinr.rename("USDINR").to_frame().to_csv(f"{DATA}/usdinr_full.csv")

# --- patch the two known gaps on the underlying levels (flagged in audit) ---
p = pd.Period("2024-05","M")
if p not in brent.index:
    a,b = brent[pd.Period("2024-04")], brent[pd.Period("2024-06")]
    brent.loc[p]= np.sqrt(a*b); brent=brent.sort_index()
p = pd.Period("2025-10","M")
if pd.isna(ucpi.get(p, np.nan)):
    a,b = ucpi[pd.Period("2025-09")], ucpi[pd.Period("2025-11")]
    ucpi.loc[p]= np.sqrt(a*b); ucpi=ucpi.sort_index()

def logret(s): return np.log(s).diff()*100
def yoy(s):    return (s/s.shift(12)-1)*100

m = pd.DataFrame({
    "USDINR": usdinr, "USDINR_Return": logret(usdinr),
    "Oil_Return": logret(brent), "Gold_Return": logret(gold),
    "Inflation_Differential": yoy(icpi)-yoy(ucpi),
    "Interest_Rate_Differential": ind10-us10,
    "FPI_Flow": fpi, "Trade_Balance": tbal,
    "FX_Reserves_Change": fxres.diff(), "VIX": vix,
    "Broad_USD_Return": logret(dxy), "GPR": gp,
}).sort_index()
m = m[(m.index>=pd.Period("2014-01")) & (m.index<=pd.Period("2025-12"))]

DEP = "USDINR_Return"
REGRESSORS = ["Oil_Return","Gold_Return","Inflation_Differential","Interest_Rate_Differential",
              "FPI_Flow","Trade_Balance","FX_Reserves_Change","VIX","Broad_USD_Return"]
data = m[[DEP]+REGRESSORS].dropna()
print(f"SAMPLE: {data.index.min()} .. {data.index.max()}  N = {len(data)}")
m.to_csv(f"{DATA}/usdinr_master_dataset.csv")
data.to_csv(f"{DATA}/regression_data.csv")

# 1. DESCRIPTIVES
desc = data.describe().T; desc["skew"]=data.skew(); desc["kurtosis"]=data.kurtosis()
desc.round(3).to_csv(f"{RESULTS}/descriptives.csv")
print("\n===== DESCRIPTIVE STATISTICS ====="); print(desc.round(3).to_string())

# 2. ADF STATIONARITY
print("\n===== ADF STATIONARITY TESTS =====")
adf=pd.DataFrame([{"variable":c,**dict(zip(["adf_stat","p_value"],adfuller(data[c].dropna(),autolag="AIC")[:2])),
                   "stationary_5pct":"YES" if adfuller(data[c].dropna(),autolag="AIC")[1]<0.05 else "NO"}
                  for c in [DEP]+REGRESSORS]).round(4)
adf.to_csv(f"{RESULTS}/adf_tests.csv",index=False); print(adf.to_string(index=False))

# 3. CORRELATION & COVARIANCE
data.corr().round(3).to_csv(f"{RESULTS}/correlation_matrix.csv")
data.cov().round(2).to_csv(f"{RESULTS}/covariance_matrix.csv")
print("\n===== CORRELATION WITH DEPENDENT =====")
print(data.corr()[DEP].drop(DEP).sort_values(key=abs,ascending=False).round(3).to_string())

# 4. VIF
def vif_table(cols):
    X=sm.add_constant(data[cols])
    return pd.DataFrame({"variable":cols,"VIF":[round(variance_inflation_factor(X.values,i+1),2) for i in range(len(cols))]})
vif_full=vif_table(REGRESSORS); vif_full.to_csv(f"{RESULTS}/vif_full.csv",index=False)
print("\n===== VIF (FULL MODEL) ====="); print(vif_full.to_string(index=False))

# 5. FULL OLS
y=data[DEP]; X=sm.add_constant(data[REGRESSORS])
full=sm.OLS(y,X).fit(); full_hac=sm.OLS(y,X).fit(cov_type="HAC",cov_kwds={"maxlags":6})
open(f"{RESULTS}/ols_full_summary.txt","w").write(
    "FULL MODEL — classical SE\n"+str(full.summary())+
    "\n\n\nFULL MODEL — HAC (Newey-West, 6 lags) robust SE\n"+str(full_hac.summary()))
print("\n===== FULL OLS MODEL (classical) ====="); print(full.summary())
full_cmp=pd.DataFrame({"coef":full.params,"p_classical":full.pvalues,"p_HAC":full_hac.pvalues}).round(4)
full_cmp.to_csv(f"{RESULTS}/full_classical_vs_hac.csv")
print("\n===== FULL MODEL: classical vs HAC p-values ====="); print(full_cmp.to_string())

# 6. BACKWARD ELIMINATION
print("\n===== BACKWARD ELIMINATION =====")
keep=REGRESSORS.copy()
while True:
    fit=sm.OLS(y,sm.add_constant(data[keep])).fit(); pv=fit.pvalues.drop("const")
    if pv.max()>0.05 and len(keep)>1:
        print(f"  drop {pv.idxmax():<28} (p={pv.max():.4f})  -> remaining {len(keep)-1}"); keep.remove(pv.idxmax())
    else: break
print("  FINAL regressors:",keep)
Xr=sm.add_constant(data[keep]); red=sm.OLS(y,Xr).fit()
red_hac=sm.OLS(y,Xr).fit(cov_type="HAC",cov_kwds={"maxlags":6})
vif_table(keep).to_csv(f"{RESULTS}/vif_reduced.csv",index=False)
open(f"{RESULTS}/ols_reduced_summary.txt","w").write(
    "REDUCED (PARSIMONIOUS) MODEL — classical SE\n"+str(red.summary())+
    "\n\n\nREDUCED MODEL — HAC robust SE\n"+str(red_hac.summary()))
print("\n===== REDUCED MODEL (classical) ====="); print(red.summary())
pd.DataFrame({"coef":red.params,"std_err_classical":red.bse,"t":red.tvalues,"p_classical":red.pvalues,
              "std_err_HAC":red_hac.bse,"p_HAC":red_hac.pvalues,
              "ci_low":red.conf_int()[0],"ci_high":red.conf_int()[1]}).round(6).to_csv(f"{RESULTS}/reduced_coefficients.csv")

# 7. DIAGNOSTICS
print("\n===== RESIDUAL DIAGNOSTICS (reduced model) =====")
bg=acorr_breusch_godfrey(red,nlags=12); bp=het_breuschpagan(red.resid,Xr)
wh=het_white(red.resid,Xr); jb=jarque_bera(red.resid); reset=linear_reset(red,power=2,use_f=True)
diag={"Durbin-Watson":round(durbin_watson(red.resid),3),"Breusch-Godfrey_p":round(bg[1],4),
 "Breusch-Pagan_p":round(bp[1],4),"White_p":round(wh[1],4),"Jarque-Bera_p":round(jb[1],4),
 "Ramsey_RESET_p":round(reset.pvalue,4),"R2":round(red.rsquared,4),"Adj_R2":round(red.rsquared_adj,4),
 "F_pvalue":float(f"{red.f_pvalue:.3e}"),"AIC":round(red.aic,2),"BIC":round(red.bic,2),"N":int(red.nobs),
 "Full_R2":round(full.rsquared,4),"Full_Adj_R2":round(full.rsquared_adj,4),
 "Full_AIC":round(full.aic,2),"Full_BIC":round(full.bic,2)}
for k,v in diag.items(): print(f"  {k}: {v}")
json.dump(diag,open(f"{RESULTS}/diagnostics.json","w"),indent=2)
pd.DataFrame({"actual":y,"fitted":red.fittedvalues,"resid":red.resid}).to_csv(f"{RESULTS}/fitted_resid.csv")
print("\nDONE — data/ and results/ written under", ROOT)
