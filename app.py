import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pickle
from helper import HelperClass

# load the models
model = pickle.load(open('model_dtr.pkl','rb'))


if __name__ == "__main__":
    # Predictive System
    st.title("Countries GDP Analysis Dashboard and Prediction System")
    features = st.text_input("Input features")
    if st.button("Predict GDP"):
        features = features.split(',')
        features = np.array([features])
        gdp_pred = model.predict(features).reshape(1, -1)
        st.write("Predict GDP Per Capita:", gdp_pred[0])

    # Side bar code
    st.sidebar.title("Data Uploader")
    file = st.sidebar.file_uploader('Upload CSV File', type=['csv'])
    if file is not None:
        # Reading data
        data = pd.read_csv(file)
        # Calling basic_counts function
        region, countries, countries_counts = HelperClass.basic_counts(data)
        st.sidebar.write("Total Region:", region)
        st.sidebar.write("Total Countries:", countries)
        st.sidebar.write("Total Countries Per Each Region:", countries_counts)

        if st.sidebar.button('Show Analysis Dashboard'):
            data = HelperClass.ConvertToFloatAndFillMissValues(data)
            st.subheader("Data View")
            st.write(data)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Average GDP, Literacy, Agriculture Per Region")
                result = HelperClass.AverageRegionsGDPLiteracyAgriculture(data)
                st.write(result)
            with col2:
                # Data aggregation
                data_agg = HelperClass.DataAgg(data)
                st.subheader("Data Aggregation Per Region")
                st.write(data_agg)

            # Top 15 Countries GDP per capita
            st.subheader("Top 15 Countries GDP Per Capita")
            HelperClass.plot_gdp_bar_chart(data)
            
            # Top 5 Asian Countries GDP, Literacy
            st.subheader("Top 5 Asian Countries GDP, Literacy")
            HelperClass.AsiaFiveRegionGDP(data)

            # Top five countries GDP per each Region
            st.subheader("Top Five Countries GDP Per Each Region")
            HelperClass.EachReginGDP(data)
