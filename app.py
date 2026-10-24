"""
Household Energy Demand Forecasting – Streamlit web app
Loads the trained pipeline (household_power_model.pkl) and predicts
the household's average power (kW) for the next hour.
"""
import pickle
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Household Energy Forecast", page_icon="⚡", layout="wide")


@st.cache_resource
def load_model():
    with open("household_power_model.pkl", "rb") as f:
        return pickle.load(f)


package = load_model()
model = package["model"]
feature_names = package["feature_names"]

st.title("⚡ Household Energy Demand Forecasting")
st.write(
    "Predicts how much electricity a household will use **in the next hour**, "
    f"using a **{package['algorithm']}** model selected by 10-fold cross validation "
    "on the UCI *Individual Household Electric Power Consumption* dataset (2,075,259 minute readings)."
)

c1, c2, c3 = st.columns(3)
c1.metric("Model", package["algorithm"])
c2.metric("CV RMSE", f"{package['cv_rmse_mean']:.3f} kW", f"± {package['cv_rmse_std']:.3f}", delta_color="off")
c3.metric("Improvement over naive forecast",
          f"{(1 - package['cv_rmse_mean'] / package['baseline_rmse']) * 100:.0f}%")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("1. When?")
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    date = st.date_input("Date of the current hour", now.date())
    hour_now = st.slider("Current hour (0–23)", 0, 23, 19)
    current = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour_now)
    target_time = current + timedelta(hours=1)
    st.caption(f"Forecasting the hour starting **{target_time:%A %d %b %Y, %H:00}**")

    st.subheader("2. Recent consumption (kW)")
    power_now = st.number_input("Average power in the current hour", 0.0, 12.0, 2.10, 0.05)
    power_1h = st.number_input("Average power 1 hour ago", 0.0, 12.0, 1.60, 0.05)
    power_2h = st.number_input("Average power 2 hours ago", 0.0, 12.0, 1.20, 0.05)
    power_yday = st.number_input("Power in the forecast hour, yesterday", 0.0, 12.0, 2.40, 0.05)
    power_week = st.number_input("Power in the forecast hour, one week ago", 0.0, 12.0, 2.20, 0.05)
    avg_24h = st.number_input("Average power over the last 24 hours", 0.0, 12.0, 1.30, 0.05)

with right:
    st.subheader("3. Current meter readings")
    voltage = st.slider("Voltage (V)", 220.0, 255.0, 239.5, 0.5)
    reactive = st.slider("Reactive power (kW)", 0.0, 1.0, 0.12, 0.01)
    kitchen = st.slider("Kitchen energy this hour – dishwasher, oven, microwave (Wh)", 0, 3000, 0, 10)
    laundry = st.slider("Laundry energy this hour – washer, dryer, fridge (Wh)", 0, 3000, 30, 10)
    heater = st.slider("Water heater & air conditioner energy this hour (Wh)", 0, 1500, 600, 10)

features = {
    "power_now": power_now,
    "power_1h_ago": power_1h,
    "power_2h_ago": power_2h,
    "power_same_hour_yesterday": power_yday,
    "power_same_hour_last_week": power_week,
    "avg_power_last_24h": avg_24h,
    "voltage_now": voltage,
    "reactive_power_now": reactive,
    "kitchen_Wh_now": kitchen,
    "laundry_Wh_now": laundry,
    "heater_ac_Wh_now": heater,
    "hour": target_time.hour,
    "day_of_week": target_time.weekday(),
    "month": target_time.month,
    "is_weekend": int(target_time.weekday() >= 5),
}
X = pd.DataFrame([features])[feature_names]

st.divider()
if st.button("🔮 Predict next-hour consumption", type="primary", width="stretch"):
    prediction = float(model.predict(X)[0])
    rmse = package["cv_rmse_mean"]
    st.success(f"### Predicted average power for {target_time:%H:00}–{target_time + timedelta(hours=1):%H:00}: "
               f"**{prediction:.3f} kW**  (≈ {prediction:.3f} kWh)")
    st.write(f"Typical error of this model: about **± {rmse:.2f} kW**, so the likely range is "
             f"**{max(prediction - rmse, 0):.2f} – {prediction + rmse:.2f} kW**.")
    change = prediction - power_now
    st.info(("📈 Consumption is expected to **rise**" if change > 0 else "📉 Consumption is expected to **fall**")
            + f" by {abs(change):.2f} kW compared with the current hour.")

with st.expander("Model comparison (10-fold cross validation)"):
    st.dataframe(pd.DataFrame(package["cv_summary"]).set_index("Algorithm"), width="stretch")
    st.caption(f"Naive 'same as this hour' baseline RMSE: {package['baseline_rmse']:.3f} kW")

st.caption("AIC354 Machine Learning Fundamentals – Assignment 3 · COMSATS University Islamabad, Lahore Campus")
