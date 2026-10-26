"""Regenerate the bundled sample CSVs in sample_data/ (deterministic)."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "sample_data"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _holes(df: pd.DataFrame, rng: np.random.Generator, cols: list[str], frac: float) -> None:
    for c in cols:
        df.loc[rng.random(len(df)) < frac, c] = np.nan


def hr_attrition() -> pd.DataFrame:
    r, n = _rng(1), 1500
    df = pd.DataFrame(
        {
            "employee_id": np.arange(1, n + 1),
            "age": r.integers(20, 62, n),
            "department": r.choice(["Sales", "Engineering", "Support", "HR", "Finance"], n, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
            "job_level": r.integers(1, 6, n),
            "monthly_income": 0,
            "years_at_company": r.exponential(5, n).round(1),
            "overtime": r.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "satisfaction": r.integers(1, 5, n),
            "distance_from_home_km": r.gamma(2, 8, n).round(1),
        }
    )
    df["monthly_income"] = (2500 + df.job_level * 1800 + df.age * 25 + r.normal(0, 600, n)).round(0)
    z = -1.2 + (df.overtime == "Yes") * 1.1 - (df.satisfaction - 2.5) * 0.5 - df.years_at_company * 0.05 + df.distance_from_home_km * 0.02
    df["left_company"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    _holes(df, r, ["satisfaction", "distance_from_home_km"], 0.03)
    return df


def house_prices() -> pd.DataFrame:
    r, n = _rng(2), 1200
    area = r.normal(140, 45, n).clip(40, 400).round(0)
    df = pd.DataFrame(
        {
            "listing_id": np.arange(5000, 5000 + n),
            "neighborhood": r.choice(["Downtown", "Riverside", "Hillcrest", "Old Town", "Suburbs"], n),
            "area_m2": area,
            "bedrooms": np.clip((area / 40 + r.normal(0, 0.7, n)).round(), 1, 7).astype(int),
            "bathrooms": r.integers(1, 4, n),
            "year_built": r.integers(1950, 2023, n),
            "garage": r.choice(["Yes", "No"], n, p=[0.6, 0.4]),
            "distance_to_center_km": r.gamma(2.5, 3, n).round(1),
        }
    )
    base = {"Downtown": 1.5, "Riverside": 1.25, "Hillcrest": 1.35, "Old Town": 1.1, "Suburbs": 0.9}
    df["price"] = (
        df.neighborhood.map(base) * (1800 * df.area_m2 + 15000 * df.bathrooms + 900 * (df.year_built - 1950) - 6000 * df.distance_to_center_km)
        * np.exp(r.normal(0, 0.08, n))
    ).round(-2)
    df.loc[r.choice(n, 6, replace=False), "price"] *= 4  # a few extreme listings
    _holes(df, r, ["year_built"], 0.02)
    return df


def retail_sales_daily() -> pd.DataFrame:
    r = _rng(3)
    days = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    t = np.arange(len(days))
    sales = 2000 + t * 1.2 + 350 * np.sin(2 * np.pi * t / 365.25) + 260 * (days.dayofweek >= 5) + r.normal(0, 120, len(days))
    promo = (r.random(len(days)) < 0.12).astype(int)
    sales = sales * (1 + 0.25 * promo)
    df = pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "store": "Main Street",
            "units_sold": (sales / 25).round(0),
            "revenue": sales.round(2),
            "promotion": promo,
            "temperature_c": (14 + 11 * np.sin(2 * np.pi * (t - 100) / 365.25) + r.normal(0, 3, len(days))).round(1),
        }
    )
    df.loc[r.choice(len(df), 4, replace=False), "revenue"] = np.nan
    return df


def marketing_ab_test() -> pd.DataFrame:
    r, n = _rng(4), 6000
    variant = r.choice(["control", "treatment"], n)
    device = r.choice(["mobile", "desktop", "tablet"], n, p=[0.6, 0.33, 0.07])
    channel = r.choice(["email", "search", "social", "direct"], n)
    p = 0.10 + 0.025 * (variant == "treatment") + 0.02 * (device == "desktop") + 0.015 * (channel == "email")
    conv = (r.random(n) < p).astype(int)
    df = pd.DataFrame(
        {
            "user_id": r.permutation(np.arange(100000, 100000 + n)),
            "variant": variant,
            "device": device,
            "channel": channel,
            "visit_date": (pd.Timestamp("2024-03-01") + pd.to_timedelta(r.integers(0, 28, n), unit="D")).strftime("%Y-%m-%d"),
            "pages_viewed": r.poisson(4, n) + 1,
            "converted": conv,
            "order_value": np.where(conv == 1, r.gamma(3, 22, n).round(2), 0.0),
        }
    )
    return df


def credit_default() -> pd.DataFrame:
    r, n = _rng(5), 4000
    df = pd.DataFrame(
        {
            "applicant_id": np.arange(1, n + 1),
            "age": r.integers(21, 70, n),
            "annual_income": r.lognormal(10.8, 0.5, n).round(-2),
            "loan_amount": r.lognormal(9.4, 0.7, n).round(-2),
            "credit_history_years": r.gamma(2.5, 4, n).round(1),
            "employment": r.choice(["salaried", "self-employed", "unemployed", "retired"], n, p=[0.6, 0.2, 0.08, 0.12]),
            "num_open_accounts": r.poisson(4, n),
            "late_payments_12m": r.poisson(0.5, n),
        }
    )
    dti = df.loan_amount / df.annual_income
    z = -4.2 + 2.0 * dti + 0.5 * df.late_payments_12m - 0.05 * df.credit_history_years + 0.8 * (df.employment == "unemployed")
    df["defaulted"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    _holes(df, r, ["annual_income", "credit_history_years"], 0.04)
    return df


def patient_readmission() -> pd.DataFrame:
    r, n = _rng(6), 2000
    df = pd.DataFrame(
        {
            "patient_id": np.arange(900000, 900000 + n),
            "age": np.clip(r.normal(63, 16, n), 18, 98).round(0),
            "sex": r.choice(["F", "M"], n),
            "primary_diagnosis": r.choice(["heart failure", "diabetes", "pneumonia", "COPD", "hip fracture"], n),
            "length_of_stay_days": np.clip(r.gamma(2, 2.5, n), 1, 40).round(0),
            "num_prior_admissions": r.poisson(1.2, n),
            "num_medications": r.poisson(8, n),
            "bmi": np.clip(r.normal(28, 5.5, n), 15, 55).round(1),
            "discharged_to": r.choice(["home", "nursing facility", "rehab"], n, p=[0.7, 0.2, 0.1]),
        }
    )
    z = -3 + 0.03 * df.age + 0.35 * df.num_prior_admissions + 0.05 * df.length_of_stay_days + 0.6 * (df.primary_diagnosis == "heart failure")
    df["readmitted_30d"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    _holes(df, r, ["bmi", "num_medications"], 0.12)
    return df


def iot_sensor_readings() -> pd.DataFrame:
    r = _rng(7)
    ts = pd.date_range("2024-05-01", periods=2880, freq="15min")
    t = np.arange(len(ts))
    temp = 55 + 6 * np.sin(2 * np.pi * t / 96) + 0.004 * t + r.normal(0, 0.8, len(ts))
    vib = 0.4 + 0.00008 * t + r.normal(0, 0.05, len(ts))
    spikes = r.choice(len(ts), 12, replace=False)
    temp[spikes] += r.uniform(15, 25, 12)
    df = pd.DataFrame(
        {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": r.choice(["M-01", "M-02", "M-03"], len(ts)),
            "temperature_c": temp.round(2),
            "vibration_mm_s": vib.round(3),
            "pressure_kpa": (101 + r.normal(0, 2.5, len(ts))).round(1),
            "status": np.where(temp > 72, "alarm", "ok"),
        }
    )
    df.loc[r.choice(len(df), 40, replace=False), "pressure_kpa"] = np.nan
    return df


def product_reviews() -> pd.DataFrame:
    r, n = _rng(8), 1800
    pos = ["great quality and fast shipping", "love it works perfectly", "excellent value highly recommend", "very happy with this purchase", "sturdy and well made"]
    neu = ["it is okay for the price", "average product nothing special", "does the job but instructions were unclear", "decent but a bit smaller than expected"]
    neg = ["broke after two days terrible", "poor quality would not buy again", "arrived late and damaged", "waste of money stopped working", "very disappointed cheap material"]
    rating = r.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.1, 0.15, 0.3, 0.35])
    text = [
        r.choice(neg if s <= 2 else neu if s == 3 else pos) + (" " + r.choice(["thanks", "again", "overall", "honestly", ""]) if r.random() < 0.5 else "")
        for s in rating
    ]
    df = pd.DataFrame(
        {
            "review_id": np.arange(1, n + 1),
            "product_category": r.choice(["electronics", "kitchen", "outdoors", "books", "toys"], n),
            "rating": rating,
            "review_text": [t.strip() for t in text],
            "verified_purchase": r.choice(["Yes", "No"], n, p=[0.85, 0.15]),
            "helpful_votes": r.poisson(2, n),
        }
    )
    df["recommended"] = (df.rating >= 4).astype(int)
    return df


def student_performance() -> pd.DataFrame:
    r, n = _rng(9), 1000
    df = pd.DataFrame(
        {
            "student_id": np.arange(1, n + 1),
            "school": r.choice(["North High", "South High", "East Academy"], n),
            "gender": r.choice(["F", "M"], n),
            "study_hours_week": np.clip(r.normal(10, 4, n), 0, 30).round(1),
            "attendance_pct": np.clip(r.normal(88, 9, n), 40, 100).round(0),
            "prior_gpa": np.clip(r.normal(2.9, 0.6, n), 0.5, 4).round(2),
            "tutoring": r.choice(["Yes", "No"], n, p=[0.25, 0.75]),
            "parent_education": r.choice(["high school", "college", "graduate"], n, p=[0.4, 0.4, 0.2]),
        }
    )
    df["final_score"] = np.clip(
        25 + 1.6 * df.study_hours_week + 0.25 * df.attendance_pct + 8 * df.prior_gpa + 4 * (df.tutoring == "Yes") + r.normal(0, 6, n), 0, 100
    ).round(1)
    _holes(df, r, ["study_hours_week"], 0.05)
    return df


def airline_delays() -> pd.DataFrame:
    r, n = _rng(10), 3000
    carriers = ["Skyline", "AeroNova", "BlueJet", "Meridian"]
    df = pd.DataFrame(
        {
            "flight_id": np.arange(1, n + 1),
            "carrier": r.choice(carriers, n),
            "origin": r.choice(["JFK", "ORD", "ATL", "DEN", "SEA", "LAX"], n),
            "distance_km": r.integers(300, 4500, n),
            "scheduled_hour": r.integers(5, 23, n),
            "day_of_week": r.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n),
            "weather_severity": r.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05]),
        }
    )
    delay = -3 + 0.9 * df.weather_severity * 12 + 0.9 * (df.scheduled_hour - 5) + 6 * (df.carrier == "BlueJet") + r.exponential(9, n)
    df["arrival_delay_min"] = delay.round(0)
    df["delayed_15"] = (df.arrival_delay_min > 15).astype(int)
    df.loc[r.random(n) < 0.02, "carrier"] = "skyline"  # inconsistent casing to clean
    return df


GENERATORS = {
    "hr_attrition": hr_attrition,
    "house_prices": house_prices,
    "retail_sales_daily": retail_sales_daily,
    "marketing_ab_test": marketing_ab_test,
    "credit_default": credit_default,
    "patient_readmission": patient_readmission,
    "iot_sensor_readings": iot_sensor_readings,
    "product_reviews": product_reviews,
    "student_performance": student_performance,
    "airline_delays": airline_delays,
}

if __name__ == "__main__":
    for name, fn in GENERATORS.items():
        frame = fn()
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(frame):,} rows x {frame.shape[1]} columns")
