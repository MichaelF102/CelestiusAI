# Sample datasets

Eleven demonstration datasets for CelestiusAI. Load them from the Executive Overview gallery or Home.

`ecommerce_customer_churn.csv` is the original churn file with deliberate quality anomalies. The other ten are
synthetic and deterministic: regenerate them with `python scripts/make_sample_data.py`.

| File | Good for |
|---|---|
| hr_attrition | Classification, survival |
| house_prices | Regression, outliers |
| retail_sales_daily | Time series, forecasting |
| marketing_ab_test | A/B testing, causal |
| credit_default | Imbalanced classification |
| patient_readmission | Data quality, classification |
| iot_sensor_readings | Anomalies, time series |
| product_reviews | Text classification |
| student_performance | Regression, group tests |
| airline_delays | Regression, cleaning |

Supported upload formats: CSV, Excel (`.xlsx`, `.xls`), and Parquet.
