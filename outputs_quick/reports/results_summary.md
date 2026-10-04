# Results summary

- Weather source: **synthetic**; event labels: **simulated**
- **Event labels are simulated.** Scores show that the pipeline works end to end. They are not evidence of real-world skill.
- Best model per hazard (selected on validation PR-AUC): landslide: **XGBoost**, flood: **XGBoost**

## Test-set metrics

| hazard | horizon | model | positives | roc_auc | pr_auc | brier_skill | ece | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| flood | 1 | Logistic Regression | 33 | 0.993 | 0.055 | -0.031 | 0.000 | 0.058 | 0.242 | 0.094 |
| flood | 6 | Logistic Regression | 198 | 0.979 | 0.133 | 0.068 | 0.001 | 0.169 | 0.429 | 0.243 |
| flood | 24 | Logistic Regression | 792 | 0.952 | 0.178 | 0.101 | 0.004 | 0.206 | 0.593 | 0.306 |
| landslide | 1 | Logistic Regression | 25 | 0.897 | 0.014 | -0.053 | 0.000 | 0.000 | 0.000 | 0.000 |
| landslide | 6 | Logistic Regression | 150 | 0.881 | 0.040 | 0.009 | 0.001 | 0.106 | 0.067 | 0.082 |
| landslide | 24 | Logistic Regression | 600 | 0.805 | 0.034 | 0.008 | 0.010 | 0.000 | 0.000 | 0.000 |
| flood | 1 | Random Forest | 33 | 0.980 | 0.163 | 0.074 | 0.000 | 0.200 | 0.242 | 0.219 |
| flood | 6 | Random Forest | 198 | 0.983 | 0.356 | 0.174 | 0.001 | 0.278 | 0.460 | 0.347 |
| flood | 24 | Random Forest | 792 | 0.949 | 0.280 | 0.152 | 0.005 | 0.417 | 0.124 | 0.191 |
| landslide | 1 | Random Forest | 25 | 0.951 | 0.102 | 0.049 | 0.000 | 0.172 | 0.200 | 0.185 |
| landslide | 6 | Random Forest | 150 | 0.979 | 0.185 | 0.030 | 0.001 | 0.227 | 0.267 | 0.245 |
| landslide | 24 | Random Forest | 600 | 0.963 | 0.317 | 0.133 | 0.005 | 0.270 | 0.557 | 0.364 |
| flood | 1 | XGBoost | 33 | 0.976 | 0.243 | 0.114 | 0.000 | 0.308 | 0.121 | 0.174 |
| flood | 6 | XGBoost | 198 | 0.979 | 0.331 | 0.187 | 0.002 | 0.254 | 0.525 | 0.343 |
| flood | 24 | XGBoost | 792 | 0.939 | 0.234 | 0.131 | 0.008 | 0.323 | 0.152 | 0.206 |
| landslide | 1 | XGBoost | 25 | 0.928 | 0.089 | -0.023 | 0.000 | 0.079 | 0.200 | 0.114 |
| landslide | 6 | XGBoost | 150 | 0.978 | 0.212 | -0.026 | 0.001 | 0.210 | 0.280 | 0.240 |
| landslide | 24 | XGBoost | 600 | 0.961 | 0.351 | 0.157 | 0.004 | 0.350 | 0.455 | 0.396 |
| flood | 1 | LightGBM | 33 | 0.739 | 0.070 | 0.049 | 0.000 | 0.075 | 0.485 | 0.130 |
| flood | 6 | LightGBM | 198 | 0.968 | 0.250 | 0.149 | 0.002 | 0.292 | 0.379 | 0.330 |
| flood | 24 | LightGBM | 792 | 0.940 | 0.232 | 0.120 | 0.006 | 0.315 | 0.323 | 0.319 |
| landslide | 1 | LightGBM | 25 | 0.690 | 0.004 | -0.001 | 0.001 | 0.010 | 0.280 | 0.020 |
| landslide | 6 | LightGBM | 150 | 0.982 | 0.229 | -0.083 | 0.001 | 0.198 | 0.327 | 0.247 |
| landslide | 24 | LightGBM | 600 | 0.961 | 0.307 | 0.109 | 0.005 | 0.380 | 0.418 | 0.398 |
| flood | 1 | LSTM | 33 | 0.946 | 0.013 | 0.005 | 0.000 | 0.028 | 0.121 | 0.046 |
| flood | 6 | LSTM | 198 | 0.930 | 0.037 | 0.020 | 0.001 | 0.049 | 0.081 | 0.061 |
| flood | 24 | LSTM | 792 | 0.905 | 0.101 | 0.052 | 0.004 | 0.101 | 0.139 | 0.117 |
| landslide | 1 | LSTM | 25 | 0.953 | 0.006 | 0.003 | 0.000 | 0.005 | 0.080 | 0.010 |
| landslide | 6 | LSTM | 150 | 0.954 | 0.039 | 0.025 | 0.000 | 0.068 | 0.107 | 0.083 |
| landslide | 24 | LSTM | 600 | 0.950 | 0.134 | 0.089 | 0.001 | 0.149 | 0.447 | 0.223 |
| flood | 1 | GRU | 33 | 0.949 | 0.008 | -0.007 | 0.000 | 0.008 | 0.212 | 0.016 |
| flood | 6 | GRU | 198 | 0.957 | 0.048 | 0.009 | 0.002 | 0.045 | 0.227 | 0.075 |
| flood | 24 | GRU | 792 | 0.928 | 0.109 | 0.068 | 0.006 | 0.126 | 0.383 | 0.190 |
| landslide | 1 | GRU | 25 | 0.969 | 0.015 | 0.007 | 0.000 | 0.013 | 0.080 | 0.022 |
| landslide | 6 | GRU | 150 | 0.974 | 0.070 | 0.054 | 0.001 | 0.096 | 0.500 | 0.161 |
| landslide | 24 | GRU | 600 | 0.972 | 0.263 | 0.180 | 0.003 | 0.272 | 0.677 | 0.388 |


## Warning lead time (24 h models)

| hazard | model | rule | threshold | events | detected | detection_rate | mean_lead_hours | false_alarm_episodes_per_region_month |
|---|---|---|---|---|---|---|---|---|
| flood | Logistic Regression | alert_level | 0.500 | 33 | 1 | 0.030 | 1.000 | 0.097 |
| flood | Logistic Regression | best_f1 | 0.095 | 33 | 31 | 0.939 | 17.613 | 1.972 |
| flood | Random Forest | alert_level | 0.500 | 33 | 2 | 0.061 | 10.500 | 0.014 |
| flood | Random Forest | best_f1 | 0.238 | 33 | 10 | 0.303 | 15.000 | 0.361 |
| flood | XGBoost | alert_level | 0.500 | 33 | 0 | 0.000 | 0.000 | 0.000 |
| flood | XGBoost | best_f1 | 0.318 | 33 | 12 | 0.364 | 13.000 | 0.431 |
| flood | LightGBM | alert_level | 0.500 | 33 | 7 | 0.212 | 8.857 | 0.278 |
| flood | LightGBM | best_f1 | 0.144 | 33 | 25 | 0.758 | 13.760 | 1.306 |
| flood | LSTM | alert_level | 0.500 | 33 | 0 | 0.000 | 0.000 | 0.000 |
| flood | LSTM | best_f1 | 0.119 | 33 | 13 | 0.394 | 11.385 | 1.028 |
| flood | GRU | alert_level | 0.500 | 33 | 0 | 0.000 | 0.000 | 0.000 |
| flood | GRU | best_f1 | 0.074 | 33 | 33 | 1.000 | 13.182 | 1.903 |
| landslide | Logistic Regression | alert_level | 0.500 | 25 | 0 | 0.000 | 0.000 | 0.000 |
| landslide | Logistic Regression | best_f1 | 0.063 | 25 | 0 | 0.000 | 0.000 | 0.083 |
| landslide | Random Forest | alert_level | 0.500 | 25 | 13 | 0.520 | 16.154 | 0.222 |
| landslide | Random Forest | best_f1 | 0.187 | 25 | 17 | 0.680 | 22.882 | 1.361 |
| landslide | XGBoost | alert_level | 0.500 | 25 | 15 | 0.600 | 16.667 | 0.389 |
| landslide | XGBoost | best_f1 | 0.317 | 25 | 17 | 0.680 | 19.000 | 0.778 |
| landslide | LightGBM | alert_level | 0.500 | 25 | 15 | 0.600 | 14.467 | 0.389 |
| landslide | LightGBM | best_f1 | 0.280 | 25 | 17 | 0.680 | 18.235 | 0.847 |
| landslide | LSTM | alert_level | 0.500 | 25 | 0 | 0.000 | 0.000 | 0.000 |
| landslide | LSTM | best_f1 | 0.109 | 25 | 18 | 0.720 | 17.111 | 1.264 |
| landslide | GRU | alert_level | 0.500 | 25 | 0 | 0.000 | 0.000 | 0.000 |
| landslide | GRU | best_f1 | 0.205 | 25 | 22 | 0.880 | 19.864 | 0.722 |


## Risk levels vs. observed event rate (best model)

| target | level | hours | events | observed_rate | share_of_events |
|---|---|---|---|---|---|
| y_flood_1h | Low | 52389 | 27 | 0.001 | 0.818 |
| y_flood_1h | Medium | 27 | 6 | 0.222 | 0.182 |
| y_flood_1h | High | 0 | 0 | 0.000 | 0.000 |
| y_flood_1h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_flood_6h | Low | 52278 | 139 | 0.003 | 0.702 |
| y_flood_6h | Medium | 138 | 59 | 0.428 | 0.298 |
| y_flood_6h | High | 0 | 0 | 0.000 | 0.000 |
| y_flood_6h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_flood_24h | Low | 51861 | 617 | 0.012 | 0.779 |
| y_flood_24h | Medium | 555 | 175 | 0.315 | 0.221 |
| y_flood_24h | High | 0 | 0 | 0.000 | 0.000 |
| y_flood_24h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_landslide_1h | Low | 52396 | 22 | 0.000 | 0.880 |
| y_landslide_1h | Medium | 14 | 2 | 0.143 | 0.080 |
| y_landslide_1h | High | 6 | 1 | 0.167 | 0.040 |
| y_landslide_1h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_landslide_6h | Low | 52260 | 110 | 0.002 | 0.733 |
| y_landslide_6h | Medium | 64 | 11 | 0.172 | 0.073 |
| y_landslide_6h | High | 39 | 9 | 0.231 | 0.060 |
| y_landslide_6h | Severe | 53 | 20 | 0.377 | 0.133 |
| y_landslide_24h | Low | 51488 | 296 | 0.006 | 0.493 |
| y_landslide_24h | Medium | 516 | 122 | 0.236 | 0.203 |
| y_landslide_24h | High | 236 | 87 | 0.369 | 0.145 |
| y_landslide_24h | Severe | 176 | 95 | 0.540 | 0.158 |


## Stability across regions (best model, 24 h)

| target | region_id | positives | roc_auc | pr_auc |
|---|---|---|---|---|
| y_flood_24h | ALP | 600 | 0.861 | 0.277 |
| y_flood_24h | IDK | 0 | nan | nan |
| y_flood_24h | KKD | 24 | 0.981 | 0.507 |
| y_flood_24h | KTM | 168 | 0.814 | 0.081 |
| y_flood_24h | PKD | 0 | nan | nan |
| y_flood_24h | TVM | 0 | nan | nan |
| y_landslide_24h | ALP | 0 | nan | nan |
| y_landslide_24h | IDK | 552 | 0.890 | 0.379 |
| y_landslide_24h | KKD | 48 | 0.763 | 0.013 |
| y_landslide_24h | KTM | 0 | nan | nan |
| y_landslide_24h | PKD | 0 | nan | nan |
| y_landslide_24h | TVM | 0 | nan | nan |

