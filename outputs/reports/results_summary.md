# Results summary

- Weather source: **synthetic**; event labels: **simulated**
- **Event labels are simulated.** Scores show that the pipeline works end to end. They are not evidence of real-world skill.
- Best model per hazard (selected on validation PR-AUC): flood: **XGBoost**, landslide: **XGBoost**

## Test-set metrics

| hazard | horizon | model | positives | roc_auc | pr_auc | brier_skill | ece | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| flood | 1 | Logistic Regression | 39 | 0.992 | 0.057 | -0.045 | 0.000 | 0.051 | 0.077 | 0.061 |
| flood | 6 | Logistic Regression | 234 | 0.984 | 0.134 | -0.030 | 0.000 | 0.186 | 0.321 | 0.235 |
| flood | 24 | Logistic Regression | 936 | 0.955 | 0.165 | 0.066 | 0.001 | 0.183 | 0.361 | 0.243 |
| landslide | 1 | Logistic Regression | 38 | 0.985 | 0.023 | -0.018 | 0.000 | 0.030 | 0.395 | 0.056 |
| landslide | 6 | Logistic Regression | 228 | 0.981 | 0.104 | 0.000 | 0.001 | 0.116 | 0.461 | 0.185 |
| landslide | 24 | Logistic Regression | 912 | 0.972 | 0.195 | 0.079 | 0.002 | 0.183 | 0.562 | 0.276 |
| flood | 1 | Random Forest | 39 | 0.976 | 0.131 | 0.045 | 0.000 | 0.278 | 0.256 | 0.267 |
| flood | 6 | Random Forest | 234 | 0.987 | 0.143 | 0.022 | 0.001 | 0.177 | 0.380 | 0.242 |
| flood | 24 | Random Forest | 936 | 0.952 | 0.102 | -0.092 | 0.003 | 0.140 | 0.341 | 0.199 |
| landslide | 1 | Random Forest | 38 | 0.964 | 0.036 | 0.009 | 0.000 | 0.038 | 0.053 | 0.044 |
| landslide | 6 | Random Forest | 228 | 0.965 | 0.171 | 0.105 | 0.000 | 0.208 | 0.329 | 0.255 |
| landslide | 24 | Random Forest | 912 | 0.968 | 0.257 | 0.162 | 0.000 | 0.330 | 0.367 | 0.348 |
| flood | 1 | XGBoost | 39 | 0.988 | 0.170 | 0.088 | 0.000 | 0.216 | 0.205 | 0.211 |
| flood | 6 | XGBoost | 234 | 0.983 | 0.194 | 0.055 | 0.001 | 0.192 | 0.380 | 0.255 |
| flood | 24 | XGBoost | 936 | 0.953 | 0.193 | 0.071 | 0.004 | 0.222 | 0.333 | 0.267 |
| landslide | 1 | XGBoost | 38 | 0.965 | 0.069 | 0.021 | 0.000 | 0.104 | 0.132 | 0.116 |
| landslide | 6 | XGBoost | 228 | 0.978 | 0.213 | 0.145 | 0.001 | 0.300 | 0.382 | 0.336 |
| landslide | 24 | XGBoost | 912 | 0.974 | 0.323 | 0.211 | 0.001 | 0.356 | 0.465 | 0.403 |
| flood | 1 | LightGBM | 39 | 0.817 | 0.021 | 0.017 | 0.000 | 0.033 | 0.590 | 0.062 |
| flood | 6 | LightGBM | 234 | 0.951 | 0.043 | 0.009 | 0.000 | 0.082 | 0.197 | 0.115 |
| flood | 24 | LightGBM | 936 | 0.945 | 0.134 | 0.039 | 0.003 | 0.140 | 0.293 | 0.189 |
| landslide | 1 | LightGBM | 38 | 0.574 | 0.008 | 0.005 | 0.000 | 0.014 | 0.500 | 0.028 |
| landslide | 6 | LightGBM | 228 | 0.948 | 0.073 | 0.035 | 0.000 | 0.113 | 0.175 | 0.137 |
| landslide | 24 | LightGBM | 912 | 0.976 | 0.258 | 0.165 | 0.001 | 0.313 | 0.406 | 0.353 |
| flood | 1 | LSTM | 39 | 0.993 | 0.072 | 0.029 | 0.000 | 0.103 | 0.282 | 0.151 |
| flood | 6 | LSTM | 234 | 0.988 | 0.127 | 0.087 | 0.000 | 0.162 | 0.235 | 0.192 |
| flood | 24 | LSTM | 936 | 0.956 | 0.165 | 0.093 | 0.002 | 0.180 | 0.434 | 0.255 |
| landslide | 1 | LSTM | 38 | 0.975 | 0.024 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 |
| landslide | 6 | LSTM | 228 | 0.972 | 0.068 | 0.052 | 0.000 | 0.094 | 0.561 | 0.161 |
| landslide | 24 | LSTM | 912 | 0.968 | 0.195 | 0.134 | 0.001 | 0.213 | 0.526 | 0.303 |
| flood | 1 | GRU | 39 | 0.992 | 0.059 | 0.036 | 0.000 | 0.053 | 0.026 | 0.034 |
| flood | 6 | GRU | 234 | 0.987 | 0.149 | 0.087 | 0.000 | 0.182 | 0.350 | 0.239 |
| flood | 24 | GRU | 936 | 0.957 | 0.200 | 0.108 | 0.002 | 0.155 | 0.544 | 0.241 |
| landslide | 1 | GRU | 38 | 0.967 | 0.047 | 0.012 | 0.000 | 0.021 | 0.132 | 0.036 |
| landslide | 6 | GRU | 228 | 0.967 | 0.092 | 0.049 | 0.000 | 0.126 | 0.338 | 0.184 |
| landslide | 24 | GRU | 912 | 0.964 | 0.213 | 0.113 | 0.001 | 0.271 | 0.352 | 0.306 |


## Warning lead time (24 h models)

| hazard | model | rule | threshold | events | detected | detection_rate | mean_lead_hours | false_alarm_episodes_per_region_month |
|---|---|---|---|---|---|---|---|---|
| flood | Logistic Regression | alert_level | 0.500 | 39 | 17 | 0.436 | 6.294 | 0.107 |
| flood | Logistic Regression | best_f1 | 0.108 | 39 | 35 | 0.897 | 13.057 | 1.423 |
| flood | Random Forest | alert_level | 0.500 | 39 | 7 | 0.179 | 11.714 | 0.363 |
| flood | Random Forest | best_f1 | 0.143 | 39 | 29 | 0.744 | 13.241 | 1.274 |
| flood | XGBoost | alert_level | 0.500 | 39 | 0 | 0.000 | 0.000 | 0.000 |
| flood | XGBoost | best_f1 | 0.243 | 39 | 31 | 0.795 | 13.613 | 2.310 |
| flood | LightGBM | alert_level | 0.500 | 39 | 0 | 0.000 | 0.000 | 0.012 |
| flood | LightGBM | best_f1 | 0.193 | 39 | 26 | 0.667 | 15.462 | 1.863 |
| flood | LSTM | alert_level | 0.500 | 39 | 0 | 0.000 | 0.000 | 0.000 |
| flood | LSTM | best_f1 | 0.163 | 39 | 29 | 0.744 | 16.552 | 1.012 |
| flood | GRU | alert_level | 0.500 | 39 | 3 | 0.077 | 7.333 | 0.012 |
| flood | GRU | best_f1 | 0.093 | 39 | 36 | 0.923 | 15.833 | 1.232 |
| landslide | Logistic Regression | alert_level | 0.500 | 38 | 12 | 0.316 | 14.167 | 0.149 |
| landslide | Logistic Regression | best_f1 | 0.096 | 38 | 34 | 0.895 | 19.176 | 1.167 |
| landslide | Random Forest | alert_level | 0.500 | 38 | 5 | 0.132 | 16.200 | 0.024 |
| landslide | Random Forest | best_f1 | 0.197 | 38 | 25 | 0.658 | 16.240 | 0.315 |
| landslide | XGBoost | alert_level | 0.500 | 38 | 10 | 0.263 | 17.400 | 0.054 |
| landslide | XGBoost | best_f1 | 0.202 | 38 | 30 | 0.789 | 16.667 | 0.571 |
| landslide | LightGBM | alert_level | 0.500 | 38 | 0 | 0.000 | 0.000 | 0.000 |
| landslide | LightGBM | best_f1 | 0.243 | 38 | 28 | 0.737 | 18.821 | 1.857 |
| landslide | LSTM | alert_level | 0.500 | 38 | 0 | 0.000 | 0.000 | 0.000 |
| landslide | LSTM | best_f1 | 0.149 | 38 | 29 | 0.763 | 19.103 | 0.655 |
| landslide | GRU | alert_level | 0.500 | 38 | 0 | 0.000 | 0.000 | 0.000 |
| landslide | GRU | best_f1 | 0.180 | 38 | 25 | 0.658 | 16.760 | 0.446 |


## Risk levels vs. observed event rate (best model)

| target | level | hours | events | observed_rate | share_of_events |
|---|---|---|---|---|---|
| y_flood_1h | Low | 122617 | 33 | 0.000 | 0.846 |
| y_flood_1h | Medium | 21 | 5 | 0.238 | 0.128 |
| y_flood_1h | High | 2 | 1 | 0.500 | 0.026 |
| y_flood_1h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_flood_6h | Low | 122273 | 154 | 0.001 | 0.658 |
| y_flood_6h | Medium | 318 | 64 | 0.201 | 0.274 |
| y_flood_6h | High | 49 | 16 | 0.327 | 0.068 |
| y_flood_6h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_flood_24h | Low | 121309 | 626 | 0.005 | 0.669 |
| y_flood_24h | Medium | 1331 | 310 | 0.233 | 0.331 |
| y_flood_24h | High | 0 | 0 | 0.000 | 0.000 |
| y_flood_24h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_landslide_1h | Low | 122629 | 37 | 0.000 | 0.974 |
| y_landslide_1h | Medium | 11 | 1 | 0.091 | 0.026 |
| y_landslide_1h | High | 0 | 0 | 0.000 | 0.000 |
| y_landslide_1h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_landslide_6h | Low | 122481 | 174 | 0.001 | 0.763 |
| y_landslide_6h | Medium | 159 | 54 | 0.340 | 0.237 |
| y_landslide_6h | High | 0 | 0 | 0.000 | 0.000 |
| y_landslide_6h | Severe | 0 | 0 | 0.000 | 0.000 |
| y_landslide_24h | Low | 121719 | 551 | 0.005 | 0.604 |
| y_landslide_24h | Medium | 698 | 228 | 0.327 | 0.250 |
| y_landslide_24h | High | 223 | 133 | 0.596 | 0.146 |
| y_landslide_24h | Severe | 0 | 0 | 0.000 | 0.000 |


## Stability across regions (best model, 24 h)

| target | region_id | positives | roc_auc | pr_auc |
|---|---|---|---|---|
| y_flood_24h | ALP | 408 | 0.877 | 0.291 |
| y_flood_24h | EKM | 192 | 0.940 | 0.208 |
| y_flood_24h | IDK | 0 | nan | nan |
| y_flood_24h | KKD | 24 | 0.927 | 0.362 |
| y_flood_24h | KLM | 24 | 0.968 | 0.106 |
| y_flood_24h | KNR | 72 | 0.917 | 0.195 |
| y_flood_24h | KSD | 0 | nan | nan |
| y_flood_24h | KTM | 72 | 0.914 | 0.095 |
| y_flood_24h | MLP | 48 | 0.958 | 0.365 |
| y_flood_24h | PKD | 0 | nan | nan |
| y_flood_24h | PTA | 24 | 0.980 | 0.209 |
| y_flood_24h | TSR | 24 | 0.804 | 0.052 |
| y_flood_24h | TVM | 48 | 0.966 | 0.253 |
| y_flood_24h | WYD | 0 | nan | nan |
| y_landslide_24h | ALP | 0 | nan | nan |
| y_landslide_24h | EKM | 0 | nan | nan |
| y_landslide_24h | IDK | 432 | 0.949 | 0.483 |
| y_landslide_24h | KKD | 48 | 0.907 | 0.035 |
| y_landslide_24h | KLM | 0 | nan | nan |
| y_landslide_24h | KNR | 120 | 0.966 | 0.246 |
| y_landslide_24h | KSD | 0 | nan | nan |
| y_landslide_24h | KTM | 0 | nan | nan |
| y_landslide_24h | MLP | 72 | 0.917 | 0.119 |
| y_landslide_24h | PKD | 0 | nan | nan |
| y_landslide_24h | PTA | 96 | 0.916 | 0.409 |
| y_landslide_24h | TSR | 0 | nan | nan |
| y_landslide_24h | TVM | 0 | nan | nan |
| y_landslide_24h | WYD | 144 | 0.924 | 0.188 |

