"""
Project 1: Customer Retention & Churn-Propensity Engine for a Digital Consumer Bank
-----------------------------------------------------------------------------------
Simulated 60K-customer retail-bank dataset (current account + savings + round-ups).
Pipeline: SQL (SQLite) feature engineering -> leakage audit -> interpretable logistic
baseline vs gradient boosting -> out-of-time evaluation -> calibration -> drift (PSI)
-> profit-based targeting policy (unit economics).

ALL DATA IS SYNTHETIC. Economic parameters (offer cost, save rate, customer value)
are explicit assumptions, not real bank figures.
"""
import sqlite3, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
sig = lambda x: 1 / (1 + np.exp(-x))

# ------------------------------------------------------------------ 1. SIMULATE
N, T = 60_000, 24                       # customers, months of history (obs month = 23)
signup = rng.integers(0, T, N)          # month of onboarding
tenure = T - signup                     # months on book at observation (1..24)
newness = signup / (T - 1)              # 0 = oldest cohort, 1 = newest
# acquisition mix drifts toward paid/partner for newer cohorts (creates real drift)
p_paid = 0.18 + 0.22 * newness
p_ref = 0.25 - 0.08 * newness
p_part = 0.07 + 0.08 * newness
p_org = 1 - p_paid - p_ref - p_part
channel = np.array([rng.choice(["organic", "referral", "paid", "partner"],
                               p=[p_org[i], p_ref[i], p_paid[i], p_part[i]]) for i in range(N)])
age = np.clip(rng.normal(32, 9, N), 18, 70).round()
salary = rng.random(N) < sig(-0.6 + 0.04 * (age - 30) - 0.4 * (channel == "paid") - 0.3 * (channel == "partner"))
eng = rng.normal(0, 1, N) + 0.6 * salary - 0.3 * (channel == "paid")
roundups = rng.random(N) < sig(-0.3 + 0.5 * eng)
saver = rng.random(N) < sig(-0.2 + 0.4 * eng + 0.3 * roundups)
support = rng.poisson(np.exp(-1.3 - 0.2 * eng), N)
rate_sens = rng.random(N) < sig(-0.9 + 0.7 * saver)          # chases best savings rate

logit = (-1.75 - 0.85 * eng - 0.55 * salary - 0.30 * np.log(tenure) + 0.35 * (channel == "paid")
         + 0.25 * (channel == "partner") - 0.25 * (channel == "referral") + 0.30 * rate_sens * (~roundups)
         + 0.18 * np.minimum(support, 4) - 0.20 * roundups + rng.normal(0, 0.35, N))
churn = rng.random(N) < sig(logit)                          # dormant in months 24-26
visible = rng.random(N) < 0.70                              # 30% churn "suddenly" w/o early warning

# monthly activity panel
rows = []
base_txn = np.exp(2.3 + 0.45 * eng + 0.5 * salary)
base_ses = np.exp(2.0 + 0.40 * eng)
base_bal = np.exp(6.0 + 0.5 * eng + 0.8 * salary + 0.6 * saver)
for m in range(T):
    live = signup <= m
    k = m - (T - 1)                                         # -2,-1,0 for months 21..23
    decay = np.ones(N)
    if k >= -2:
        decay = np.where(churn & visible, [0.92, 0.78, 0.60][k + 2], 1.0)
    txn = rng.poisson(base_txn * decay * rng.lognormal(0, 0.30, N))
    ses = rng.poisson(base_ses * decay * rng.lognormal(0, 0.30, N))
    spend = txn * rng.gamma(4, 9, N)
    bal = base_bal * (decay ** 0.7) * rng.lognormal(0, 0.25, N)
    idx = np.where(live)[0]
    rows.append(pd.DataFrame(dict(customer_id=idx, month_idx=m, txn_count=txn[idx], app_sessions=ses[idx],
                                  spend=spend[idx].round(2), balance=bal[idx].round(2),
                                  salary_credit=(salary[idx] & (rng.random(len(idx)) < 0.93)).astype(int))))
panel = pd.concat(rows, ignore_index=True)

cust = pd.DataFrame(dict(customer_id=np.arange(N), signup_month=signup, age=age, channel=channel,
                         roundups_on=roundups.astype(int), saver_user=saver.astype(int),
                         support_contacts_90d=support, rate_sensitive=rate_sens.astype(int),
                         # LEAKY field: closure request is logged AFTER the observation date
                         closure_request_flag=((churn) & (rng.random(N) < 0.55)).astype(int),
                         churn_90d=churn.astype(int)))

con = sqlite3.connect(":memory:")
cust.to_sql("customers", con, index=False)
panel.to_sql("monthly_activity", con, index=False)
con.execute("CREATE INDEX ix ON monthly_activity(customer_id, month_idx)")

# ------------------------------------------------------------------ 2. SQL FEATURES
SQL = """
WITH w AS (
  SELECT customer_id, month_idx, txn_count, app_sessions, spend, balance, salary_credit,
         AVG(txn_count)    OVER (PARTITION BY customer_id ORDER BY month_idx ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS txn_avg_3m,
         AVG(txn_count)    OVER (PARTITION BY customer_id ORDER BY month_idx ROWS BETWEEN 5 PRECEDING AND 3 PRECEDING) AS txn_avg_prev3m,
         AVG(app_sessions) OVER (PARTITION BY customer_id ORDER BY month_idx ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ses_avg_3m,
         AVG(app_sessions) OVER (PARTITION BY customer_id ORDER BY month_idx ROWS BETWEEN 5 PRECEDING AND 3 PRECEDING) AS ses_avg_prev3m,
         SUM(spend)        OVER (PARTITION BY customer_id ORDER BY month_idx ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS spend_3m,
         LAG(balance, 3)   OVER (PARTITION BY customer_id ORDER BY month_idx) AS balance_3m_ago,
         SUM(salary_credit) OVER (PARTITION BY customer_id ORDER BY month_idx ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS salary_months_6m,
         COUNT(*)          OVER (PARTITION BY customer_id ORDER BY month_idx ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS months_obs_6m
  FROM monthly_activity
)
SELECT c.customer_id, c.signup_month, c.age, c.channel, c.roundups_on, c.saver_user,
       c.support_contacts_90d, c.rate_sensitive, c.closure_request_flag, c.churn_90d,
       23 - c.signup_month + 1 AS tenure_months,
       w.txn_count AS txn_m0, w.txn_avg_3m,
       COALESCE(w.txn_avg_3m * 1.0 / NULLIF(w.txn_avg_prev3m, 0), 1.0)  AS txn_trend,
       w.ses_avg_3m,
       COALESCE(w.ses_avg_3m * 1.0 / NULLIF(w.ses_avg_prev3m, 0), 1.0)  AS ses_trend,
       w.spend_3m, w.balance AS balance_m0,
       COALESCE(w.balance * 1.0 / NULLIF(w.balance_3m_ago, 0), 1.0)     AS balance_ratio_3m,
       w.salary_months_6m * 1.0 / w.months_obs_6m                       AS salary_regularity
FROM customers c JOIN w ON w.customer_id = c.customer_id AND w.month_idx = 23
"""
df = pd.read_sql(SQL, con)
df = pd.get_dummies(df, columns=["channel"], dtype=int)

LEAK = ["closure_request_flag"]
ID = ["customer_id", "signup_month", "churn_90d"]
HONEST = [c for c in df.columns if c not in ID + LEAK]

# stratified random split 65/15/20 (cohort split would separate tenure perfectly and force extrapolation)
from sklearn.model_selection import train_test_split
tr0, te = train_test_split(df, test_size=0.20, stratify=df.churn_90d, random_state=1)
tr, ca = train_test_split(tr0, test_size=0.1875, stratify=tr0.churn_90d, random_state=2)
ytr, yca, yte = tr.churn_90d, ca.churn_90d, te.churn_90d

res = {"n_customers": N, "base_rate": round(float(df.churn_90d.mean()), 4),
       "n_train": len(tr), "n_calib": len(ca), "n_test": len(te)}

# ------------------------------------------------------------------ 3. LEAKAGE AUDIT
gb_leak = HistGradientBoostingClassifier(random_state=0).fit(tr[HONEST + LEAK], ytr)
res["auc_with_leaky_feature"] = round(roc_auc_score(yte, gb_leak.predict_proba(te[HONEST + LEAK])[:, 1]), 4)

# ------------------------------------------------------------------ 4. MODELS
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(tr[HONEST], ytr)
gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0,
                                    random_state=0).fit(tr[HONEST], ytr)
p_lr, p_gb = lr.predict_proba(te[HONEST])[:, 1], gb.predict_proba(te[HONEST])[:, 1]

def ks(y, p):
    f, t, _ = roc_curve(y, p); return float(np.max(t - f))
def lift_at(y, p, frac):
    k = int(len(y) * frac); o = np.argsort(-p)[:k]; return float(np.mean(np.asarray(y)[o]) / np.mean(y))
for nm, p in [("logistic", p_lr), ("gbm", p_gb)]:
    res[f"{nm}_auc"] = round(roc_auc_score(yte, p), 4)
    res[f"{nm}_pr_auc"] = round(average_precision_score(yte, p), 4)
    res[f"{nm}_ks"] = round(ks(yte, p), 4)
    res[f"{nm}_top_decile_lift"] = round(lift_at(yte, p, 0.10), 2)
res["leakage_inflation_auc_pts"] = round(res["auc_with_leaky_feature"] - res["gbm_auc"], 4)

coefs = pd.Series(lr[-1].coef_[0], index=HONEST).sort_values(key=abs, ascending=False)
res["top_logit_drivers_std_coef"] = {k: round(float(v), 3) for k, v in coefs.head(6).items()}

# ------------------------------------------------------------------ 5. CALIBRATION
p_ca = gb.predict_proba(ca[HONEST])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca, yca)
p_cal = iso.predict(p_gb)
res["gbm_brier_raw"] = round(brier_score_loss(yte, p_gb), 5)
res["gbm_brier_calibrated"] = round(brier_score_loss(yte, p_cal), 5)
def ece(y, p, b=10):
    q = pd.qcut(p, b, duplicates="drop"); g = pd.DataFrame({"y": np.asarray(y), "p": p, "q": q}).groupby("q", observed=True)
    return float(np.sum(g.size() / len(p) * (g.p.mean() - g.y.mean()).abs()))
res["gbm_ece_raw"], res["gbm_ece_calibrated"] = round(ece(yte, p_gb), 4), round(ece(yte, p_cal), 4)

# ------------------------------------------------------------------ 6. STABILITY / DRIFT BY ONBOARDING COHORT
def psi(a, b, bins=10):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6; pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))
old_c, new_c = df[df.signup_month <= 11], df[df.signup_month >= 12]      # older vs newer 12-month cohorts
drift = {c: round(psi(old_c[c].values, new_c[c].values), 3) for c in ["txn_avg_3m", "ses_avg_3m", "balance_m0", "salary_regularity", "age"]}
drift["paid_channel_share_old_vs_new"] = f"{old_c.channel_paid.mean():.1%} -> {new_c.channel_paid.mean():.1%}"
s_old, s_new = gb.predict_proba(old_c[HONEST])[:, 1], gb.predict_proba(new_c[HONEST])[:, 1]
drift["score_psi_old_vs_new"] = round(psi(s_old, s_new), 3)
drift["auc_old_cohorts_holdout"] = round(roc_auc_score(te[te.signup_month <= 11].churn_90d, gb.predict_proba(te[te.signup_month <= 11][HONEST])[:, 1]), 4)
drift["auc_new_cohorts_holdout"] = round(roc_auc_score(te[te.signup_month >= 12].churn_90d, gb.predict_proba(te[te.signup_month >= 12][HONEST])[:, 1]), 4)
res["drift_and_stability"] = drift

# ------------------------------------------------------------------ 7. TARGETING ECONOMICS (assumptions!)
V, COST, SAVE = 120.0, 6.0, 0.15      # annual margin per retained customer (GBP), offer cost, save-rate on would-be churners
thr = COST / (SAVE * V)               # break-even probability
def net(mask, y): return float(np.sum(np.where(mask, y * SAVE * V - COST, 0.0)))
y = yte.values; n_te = len(y)
pol = {"target_all": net(np.ones(n_te, bool), y)}
rand = np.zeros(n_te, bool); rand[rng.choice(n_te, int(n_te * 0.2), replace=False)] = True
pol["random_20pct"] = net(rand, y)
k20 = np.zeros(n_te, bool); k20[np.argsort(-p_cal)[:int(n_te * .2)]] = True
pol["model_top20pct"] = net(k20, y)
opt = p_cal >= thr
pol["model_breakeven_threshold"] = net(opt, y)
res["economics"] = {"assumptions": dict(value_per_saved_customer_gbp=V, offer_cost_gbp=COST, save_rate=SAVE,
                                        breakeven_prob=round(thr, 3)),
                    "net_value_gbp_test_set": {k: round(v) for k, v in pol.items()},
                    "customers_targeted_at_breakeven": int(opt.sum()), "share_targeted": round(float(opt.mean()), 3),
                    "precision_among_targeted": round(float(y[opt].mean()), 3),
                    "recall_of_churners_captured": round(float(y[opt].sum() / y.sum()), 3)}
res["economics"]["uplift_vs_target_all_gbp"] = round(pol["model_breakeven_threshold"] - pol["target_all"])
res["economics"]["uplift_vs_random20_gbp"] = round(pol["model_breakeven_threshold"] - pol["random_20pct"])

# decile table
dec = pd.DataFrame({"y": y, "p": p_cal}); dec["decile"] = pd.qcut(dec.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
dt = dec.groupby("decile", observed=True).agg(customers=("y", "size"), actual_churn=("y", "mean"), mean_pred=("p", "mean")).round(3)
dt["lift"] = (dt.actual_churn / y.mean()).round(2)
dt.to_csv("outputs/decile_table.csv")

# ------------------------------------------------------------------ 8. FIGURES
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for nm, p in [("Logistic", p_lr), ("GBM", p_gb)]:
    f, t, _ = roc_curve(yte, p); ax[0].plot(f, t, label=f"{nm} (AUC {roc_auc_score(yte, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=.7); ax[0].set(title="ROC - held-out customers", xlabel="FPR", ylabel="TPR"); ax[0].legend()
ax[1].bar(dt.index.astype(int), dt.lift, color="#1f4e79"); ax[1].axhline(1, c="k", lw=.7)
ax[1].set(title="Lift by risk decile (calibrated GBM)", xlabel="Decile (1 = highest risk)", ylabel="Lift vs base rate")
ax[2].plot([0, .6], [0, .6], "k--", lw=.7)
for nm, p in [("Raw", p_gb), ("Isotonic", p_cal)]:
    q = pd.qcut(pd.Series(p).rank(method="first"), 10); g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).mean()
    ax[2].plot(g.p, g.y, "o-", label=nm)
ax[2].set(title="Calibration (deciles)", xlabel="Mean predicted", ylabel="Observed churn"); ax[2].legend()
plt.tight_layout(); plt.savefig("outputs/retention_model_diagnostics.png", dpi=130)

json.dump(res, open("outputs/results_project1.json", "w"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str)); print(dt)
