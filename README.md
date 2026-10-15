# Project 1: Customer Retention & Churn-Propensity Engine for a Digital Bank
Target role: Data Scientist Associate, JPMorganChase (Chase UK, Business Analytics), Job ID 210788748

All data is **synthetic** (generated in the script, seeded for reproducibility). Economic parameters
(customer value, offer cost, save rate) are stated assumptions, not real bank figures. Say "simulated" in interviews.

**Stack:** Python, SQL (SQLite window functions/CTEs), scikit-learn (logistic regression, gradient boosting), pandas, matplotlib

## Resume bullets
- Simulated a 60K-customer digital-bank dataset (current account, savings pot, round-ups; 10.3% 90-day dormancy rate) and engineered ~20 behavioural features in SQL using window functions, LAG and COALESCE/NULLIF guards (3-month rolling activity, trend ratios, balance drawdown, salary regularity).
- Built an interpretable logistic baseline (AUC 0.846) and a gradient-boosted model (AUC 0.850, PR-AUC 0.49 vs 0.10 base rate, KS 0.55); top risk decile churns at 50.8% vs 10.3% average (4.9x lift).
- Ran a leakage audit: a post-observation account-closure flag inflated AUC from 0.850 to 0.928; removed it from the feature set. Verified calibration (ECE 0.4%) and monitored cohort drift (score PSI 0.11 as paid-channel mix rose 22.9% to 35.0%) with stable AUC across onboarding cohorts (0.845 vs 0.849).
- Converted scores into a profit-based targeting policy (break-even probability 0.33 under assumed £6 offer cost, 15% save rate, £120 customer value): targeting 7.2% of customers captured 40% of churners at 56.5% precision, +£3.6K net on a 12K-customer holdout vs -£10.2K for random 20% targeting and -£49.8K for contacting everyone.

## Run
```
pip install pandas numpy scikit-learn matplotlib
mkdir -p outputs && python3 retention_model.py
```

## Files
- `retention_model.py` - full pipeline (simulation, SQL features, leakage audit, models, calibration, drift, targeting economics)
- `outputs/results_project1.json` - all metrics
- `outputs/decile_table.csv` - lift by risk decile
- `outputs/retention_model_diagnostics.png` - ROC, lift, calibration plots

## Caveats (know these for interviews)
- Results reflect the simulation's signal strength, not real-world performance.
- Raw GBM scores were already well calibrated; isotonic recalibration gave no gain, so say "verified calibration", not "improved".
