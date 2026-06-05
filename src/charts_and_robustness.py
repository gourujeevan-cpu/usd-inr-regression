#!/usr/bin/env python3
"""
Charts + robustness for the USD/INR regression.
Reads data/ (built by regression_analysis.py) and writes charts/ + results/.
Run from the repo root (or src/), or in a notebook whose working dir is the repo root:
    python src/charts_and_robustness.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

# ----- portable paths -----
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
DATA    = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
CH      = os.path.join(ROOT, "charts")
for d in (RESULTS, CH): os.makedirs(d, exist_ok=True)

plt.rcParams.update({"figure.dpi":120,"savefig.dpi":120,"font.size":10,"axes.grid":True,
                     "grid.alpha":0.3,"axes.spines.top":False,"axes.spines.right":False})
NAVY="#1f3b57"; RUST="#c1442e"; TEAL="#2a7f7f"; GOLD="#c79a3a"; GREY="#9aa0a6"

data=pd.read_csv(f"{DATA}/regression_data.csv",index_col=0)
data.index=pd.PeriodIndex(data.index,freq="M"); t=data.index.to_timestamp()

DEP="USDINR_Return"
REG=["Oil_Return","Gold_Return","Inflation_Differential","Interest_Rate_Differential",
     "FPI_Flow","Trade_Balance","FX_Reserves_Change","VIX","Broad_USD_Return"]
FINAL=["FPI_Flow","Broad_USD_Return"]
y=data[DEP]
full=sm.OLS(y,sm.add_constant(data[REG])).fit()
red =sm.OLS(y,sm.add_constant(data[FINAL])).fit()
red_hac=sm.OLS(y,sm.add_constant(data[FINAL])).fit(cov_type="HAC",cov_kwds={"maxlags":6})

# VIF (computed inline — no dependency on results/)
Xf=sm.add_constant(data[REG])
vif=pd.DataFrame({"variable":REG,"VIF":[round(variance_inflation_factor(Xf.values,i+1),2) for i in range(len(REG))]})

# standardized betas (full model)
z=(data[[DEP]+REG]-data[[DEP]+REG].mean())/data[[DEP]+REG].std()
stdbeta=sm.OLS(z[DEP],sm.add_constant(z[REG])).fit().params.drop("const")

# robustness: difference the 3 non-stationary vars, add to final
rob=data.copy()
for c in ["Inflation_Differential","Interest_Rate_Differential","Trade_Balance"]: rob["d_"+c]=rob[c].diff()
rob=rob.dropna()
robfit=sm.OLS(rob[DEP],sm.add_constant(rob[["FPI_Flow","Broad_USD_Return","d_Inflation_Differential",
              "d_Interest_Rate_Differential","d_Trade_Balance"]])).fit()

print("===== REDUCED MODEL: classical vs HAC =====")
cmp=pd.DataFrame({"coef":red.params,"p_classical":red.pvalues,"p_HAC":red_hac.pvalues}).round(4)
print(cmp.to_string()); cmp.to_csv(f"{RESULTS}/reduced_classical_vs_hac.csv")
print("\n===== STANDARDIZED BETAS (full model) ====="); print(stdbeta.round(3).sort_values(key=abs,ascending=False).to_string())
print("\n===== ROBUSTNESS: differenced non-stationary vars added =====")
print(pd.DataFrame({"coef":robfit.params,"p":robfit.pvalues}).round(4).to_string())
print("Robust R2:",round(robfit.rsquared,4),"adjR2:",round(robfit.rsquared_adj,4))

LAB={"Oil_Return":"Oil ret","Gold_Return":"Gold ret","Inflation_Differential":"Infl. diff",
 "Interest_Rate_Differential":"Rate diff","FPI_Flow":"FPI flow","Trade_Balance":"Trade bal",
 "FX_Reserves_Change":"FX res \u0394","VIX":"VIX","Broad_USD_Return":"Broad USD ret","USDINR_Return":"USDINR ret"}

# ---------- FIG 1: USD/INR level (full history if available) ----------
fig,ax=plt.subplots(figsize=(9,3.6))
full_path=f"{DATA}/usdinr_full.csv"
if os.path.exists(full_path):
    lv=pd.read_csv(full_path,index_col=0); lv.index=pd.PeriodIndex(lv.index,freq="M")
    ax.plot(lv.index.to_timestamp(),lv["USDINR"].values,color=NAVY,lw=1.2)
    title="USD/INR exchange rate level (INR per USD 1), full history"
else:
    m=pd.read_csv(f"{DATA}/usdinr_master_dataset.csv",index_col=0); m.index=pd.PeriodIndex(m.index,freq="M")
    ax.plot(m.index.to_timestamp(),m["USDINR"].values,color=NAVY,lw=1.4); title="USD/INR exchange rate level (INR per USD 1)"
ax.axvspan(pd.Timestamp("2014-01-01"),pd.Timestamp("2025-12-31"),color=GOLD,alpha=0.15,label="regression sample (2014-2025)")
ax.set_title(title,fontweight="bold"); ax.set_ylabel("INR per USD 1"); ax.legend(loc="upper left",frameon=False)
fig.tight_layout(); fig.savefig(f"{CH}/fig1_usdinr_level.png"); plt.close()

# ---------- FIG 2: dependent variable ----------
fig,ax=plt.subplots(figsize=(9,3.4))
ax.bar(t,y.values,width=22,color=[RUST if v>0 else TEAL for v in y.values]); ax.axhline(0,color="k",lw=0.7)
ax.set_title("Dependent variable: monthly USD/INR log return (%)  -  red = depreciation",fontweight="bold")
ax.set_ylabel("% log return"); fig.tight_layout(); fig.savefig(f"{CH}/fig2_dependent_return.png"); plt.close()

# ---------- FIG 3: drivers ----------
fig,axes=plt.subplots(3,3,figsize=(11,8)); axes=axes.ravel()
for i,c in enumerate(REG):
    axes[i].plot(t,data[c].values,color=NAVY,lw=1.1); axes[i].set_title(LAB[c],fontsize=10,fontweight="bold")
    axes[i].tick_params(labelsize=8)
    if c=="Trade_Balance": axes[i].axhline(0,color=GREY,lw=0.6)
fig.suptitle("Explanatory variables over the sample (Jan 2014 - Dec 2025)",fontweight="bold",y=1.01)
fig.tight_layout(); fig.savefig(f"{CH}/fig3_drivers_timeseries.png",bbox_inches="tight"); plt.close()

# ---------- FIG 4: heatmap ----------
corr=data.corr(); fig,ax=plt.subplots(figsize=(8,6.6)); im=ax.imshow(corr.values,cmap="RdBu_r",vmin=-1,vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels([LAB[c] for c in corr.columns],rotation=45,ha="right",fontsize=8)
ax.set_yticklabels([LAB[c] for c in corr.columns],fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=7,
                color="white" if abs(corr.values[i,j])>0.55 else "black")
ax.set_title("Correlation matrix (dependent + 9 regressors)",fontweight="bold")
fig.colorbar(im,fraction=0.046,pad=0.04); fig.tight_layout(); fig.savefig(f"{CH}/fig4_corr_heatmap.png"); plt.close()

# ---------- FIG 5: corr with dependent ----------
cd=corr[DEP].drop(DEP).sort_values(); fig,ax=plt.subplots(figsize=(8,4))
ax.barh([LAB[c] for c in cd.index],cd.values,color=[RUST if v>0 else TEAL for v in cd.values]); ax.axvline(0,color="k",lw=0.7)
ax.set_title("Correlation of each regressor with USD/INR return",fontweight="bold"); ax.set_xlabel("Pearson r")
for i,v in enumerate(cd.values): ax.text(v+(0.01 if v>=0 else -0.01),i,f"{v:.2f}",va="center",ha="left" if v>=0 else "right",fontsize=8)
fig.tight_layout(); fig.savefig(f"{CH}/fig5_corr_with_dep_bar.png"); plt.close()

# ---------- FIG 6: VIF ----------
fig,ax=plt.subplots(figsize=(8,4)); ax.bar([LAB[c] for c in vif["variable"]],vif["VIF"],color=NAVY)
ax.axhline(5,color=RUST,ls="--",lw=1.2,label="VIF = 5 (caution)")
ax.set_title("Variance Inflation Factors (multicollinearity check)",fontweight="bold"); ax.set_ylabel("VIF")
ax.legend(frameon=False); plt.xticks(rotation=45,ha="right",fontsize=8)
for i,v in enumerate(vif["VIF"]): ax.text(i,v+0.05,f"{v:.2f}",ha="center",fontsize=8)
fig.tight_layout(); fig.savefig(f"{CH}/fig6_vif_bar.png"); plt.close()

# ---------- FIG 7: full-model t-stats ----------
tv=full.tvalues.drop("const"); order=tv.abs().sort_values().index; fig,ax=plt.subplots(figsize=(8,4.2))
ax.barh([LAB[c] for c in order],[tv[c] for c in order],color=[GOLD if abs(tv[c])>=1.96 else GREY for c in order])
ax.axvline(1.96,color=RUST,ls="--",lw=1); ax.axvline(-1.96,color=RUST,ls="--",lw=1); ax.axvline(0,color="k",lw=0.7)
ax.set_title("Full-model t-statistics (gold = significant at 5%, |t|>=1.96)",fontweight="bold"); ax.set_xlabel("t-statistic")
fig.tight_layout(); fig.savefig(f"{CH}/fig7_full_tstats_bar.png"); plt.close()

# ---------- FIG 8: actual vs fitted ----------
fig,ax=plt.subplots(figsize=(9,3.8)); ax.plot(t,y.values,color=NAVY,lw=1.3,label="Actual")
ax.plot(t,red.fittedvalues.values,color=RUST,lw=1.5,ls="--",label="Fitted (FPI + Broad USD)"); ax.axhline(0,color=GREY,lw=0.6)
ax.set_title(f"Actual vs fitted USD/INR return - parsimonious model (R2={red.rsquared:.2f})",fontweight="bold")
ax.set_ylabel("% log return"); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(f"{CH}/fig8_actual_vs_fitted.png"); plt.close()

# ---------- FIG 9: residual diagnostics ----------
res=red.resid; fit=red.fittedvalues; fig=plt.figure(figsize=(10,7)); gs=GridSpec(2,2,figure=fig)
a1=fig.add_subplot(gs[0,0]); a1.plot(t,res.values,color=NAVY,lw=1); a1.axhline(0,color=RUST,lw=0.8); a1.set_title("Residuals over time",fontweight="bold")
a2=fig.add_subplot(gs[0,1]); a2.scatter(fit.values,res.values,s=14,color=TEAL,alpha=0.7); a2.axhline(0,color=RUST,lw=0.8)
a2.set_title("Residuals vs fitted",fontweight="bold"); a2.set_xlabel("fitted")
a3=fig.add_subplot(gs[1,0]); a3.hist(res.values,bins=24,color=NAVY,alpha=0.8,density=True)
xs=np.linspace(res.min(),res.max(),100); a3.plot(xs,stats.norm.pdf(xs,res.mean(),res.std()),color=RUST,lw=1.6); a3.set_title("Residual distribution vs normal",fontweight="bold")
a4=fig.add_subplot(gs[1,1]); stats.probplot(res.values,dist="norm",plot=a4)
a4.get_lines()[0].set_color(TEAL); a4.get_lines()[0].set_markersize(4); a4.get_lines()[1].set_color(RUST); a4.set_title("Normal Q-Q plot",fontweight="bold")
fig.suptitle("Residual diagnostics - parsimonious model",fontweight="bold",y=1.01); fig.tight_layout(); fig.savefig(f"{CH}/fig9_residual_diagnostics.png",bbox_inches="tight"); plt.close()

# ---------- FIG 10: reduced coefficients CI ----------
ci=red.conf_int(); params=red.params.drop("const"); fig,ax=plt.subplots(figsize=(7,3)); yp=range(len(params))
ax.errorbar(params.values,list(yp),xerr=[(params-ci[0].drop("const")).values,(ci[1].drop("const")-params).values],
            fmt="o",color=NAVY,ecolor=RUST,capsize=4,ms=7); ax.axvline(0,color="k",lw=0.7)
ax.set_yticks(list(yp)); ax.set_yticklabels([LAB[c] for c in params.index])
ax.set_title("Parsimonious-model coefficients (95% CI)",fontweight="bold"); ax.set_xlabel("coefficient")
fig.tight_layout(); fig.savefig(f"{CH}/fig10_reduced_coef_ci.png"); plt.close()

print("\nCharts written to", CH, "->", sorted(os.listdir(CH)))
