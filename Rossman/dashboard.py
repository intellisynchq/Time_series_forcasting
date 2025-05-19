import streamlit as st
import pandas as pd
from streamlit_utils import (
    plot_actual_vs_predicted,
    plot_outliers,
    identify_outliers,
    prediction,
    plot_store_time_series
)

st.title("Store Sales Prediction Dashboard")
store_id = st.number_input("Enter the store id", min_value=1, max_value=1000, value=1)

train_df = pd.read_csv('train.csv', low_memory=False)

st.write("Time series plot:")
time_series_fig = plot_store_time_series(store_id, train_df)
st.pyplot(time_series_fig)

st.write("Outliers plot:")
outliers_df = identify_outliers(train_df)
outlier_fig = plot_outliers(store_id, outliers_df)
st.pyplot(outlier_fig)

st.write("Actual vs Predicted plot (for all stores):")
st.image("actual_vs_predicted.png")

