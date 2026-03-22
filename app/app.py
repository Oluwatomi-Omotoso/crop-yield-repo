import streamlit as st
import joblib
import pandas as pd
import numpy as np
import torch
import sys, os
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ML.pipelines.train_and_evaluate import CropYieldModel
from ML.pipelines.train_and_evaluate import calc_r2_score

input_cols = [
    "area",
    "year",
    "average_rain_fall_mm_per_year",
    "crops",
    "pesticide_tonnes",
    "avg_temp",
]

output_cols = ["crop_yields"]

input_size = len(input_cols)


# Load models and artificats
@st.cache_resource()
def load_assets():

    # Load NN assets
    
    nn_model = CropYieldModel(input_size, output_size=1)
    model_path = os.path.abspath(os.path.join(BASE_DIR, "ML/models/NN_Models/neural_net.pth"))
    nn_model.load_state_dict(torch.load(model_path))
    
    nn_model.eval()

    scaler = joblib.load(os.path.join(BASE_DIR,"ML/models/NN_Models/scaler.pkl"))
    cat_mappings = joblib.load(os.path.join(BASE_DIR,"ML/models/NN_Models/category_mappings.pkl"))

    # load pipeline model(Random_Forest)
    pipeline_model = joblib.load(os.path.join(BASE_DIR,"ML/models/final-model.pkl"))

    return nn_model, scaler, cat_mappings, pipeline_model


NN_model, Scaler, categorical_mappings, pipeline_model = load_assets()


# UI setup & ML
st.title("Agricultural Intelligence: Dual-Model Prediction")

df = pd.read_csv(os.path.join(BASE_DIR,"data/processed/Final_dataset.csv"))

locations = sorted(df["area"].unique())
crops = sorted(df["crops"].unique())

col1, col2 = st.columns(2)


with col1:
    area = st.selectbox("Location: ", locations)
    year = st.number_input("Year: ", 1990, 2030, 2026)
    crop = st.selectbox("Crop: ", crops)

with col2:
    pesticide = st.number_input("Pesticide (tonnes): ", 0, 200000, 0)
    temp = st.slider("Average Temperature: ", -30, 50, 20)
    rainfall = st.number_input("Average rainfall (mm/year): ", 0, 4000, 1000)


if st.button("Generate comparative predictions"):

    input_df = pd.DataFrame(
        {
            "area": [area],
            "year": [year],
            "average_rain_fall_mm_per_year": [rainfall],
            "crops": [crop],
            "pesticide_tonnes": [pesticide],
            "avg_temp": [temp],
        }
    )
    pipeline_pred = pipeline_model.predict(input_df)[0]

    # Encode categorical data
    area_code = categorical_mappings["area"].get(area, 0)
    crop_code = categorical_mappings["crops"].get(crop, 0)

    # Bulding array
    nn_raw = np.array(
        [
            [
                area_code,
                year,
                rainfall,
                crop_code,
                pesticide,
                temp,
            ]
        ],
        dtype=np.float32,
    )
    nn_scaled = Scaler.transform(nn_raw)
    nn_tensor = torch.Tensor(nn_scaled)

    # calc_r2_score(model=NN_model, X_scaled=nn_scaled, y_test=output_cols)

    with torch.no_grad():
        NN_pred = NN_model(nn_tensor).item()

    st.divider()
    res1, res2 = st.columns(2)

    res1.metric("Pipeline Model says", f"{pipeline_pred:.0f} hg/ha")
    res2.metric("Neural Network says", f"{NN_pred:.0f} hg/ha")

    # Comparative Insights
    diff = abs(pipeline_pred - NN_pred)
    st.info(f"The models have a variance of {diff:.0f} hg/ha")

    # Historical context comparison
    st.subheader("Historical Context")
    historical_data = df[(df["crops"] == crop) & (df["area"] == area)].sort_values(
        "year"
    )

    if not historical_data.empty:
        fig = px.line(
            historical_data,
            x="year",
            y="hg/ha_yield",
            title=f"Yield trends for {crop} in {area}",
            labels={"hg/ha_yield": "Yield (hg/ha)"},
        )

        fig.add_trace(
            go.Scatter(
                x=[year, year],
                y=[pipeline_pred, NN_pred],
                mode="markers+text",
                name="Current Predictions",
                text=["Pipeline", "NN"],
                textposition="top.center",
                marker=dict(color="red", size=12),
            )
        )

    else:
        st.info(
            "No historical data available for this specific crop/location combination."
        )

    # Model confidence (R squared Scores)
    from ML.pipelines.evaluate_algorithms import get_best_model_r2_score
    import pandas as pd

    comparison_df = pd.read_csv("../data/processed/algoritm_comparison_data.csv")
    pipeline_r2_score = get_best_model_r2_score(comparison_df)

    st.subheader("Model Confidence Metrics")

    confidence_data = pd.DataFrame(
        {
            "Model": ["Pipeline (Random Forest)", "Neural Network"],
            "Confidence (R2)": [pipeline_r2_score, 0.7057],
            "Error Margin (MAE)": ["Low", "Medium"],
        }
    )

    fig_conf = px.bar(
        confidence_data,
        x="Model",
        y="Confidence (R2)",
        color="Model",
        range_y=[0, 1],
        text_auto=".3f",
        title="Prediction Confidence (based on Training R2)",
    )

    st.plotly_chart(fig_conf)
