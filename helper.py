import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pickle

class HelperClass:
    # Helper functions
    @staticmethod
    def basic_counts(data):
        region = data['Region'].nunique()
        countries = data['Country'].nunique()
        countries_counts = data['Region'].value_counts()
        return region, countries, countries_counts

    @staticmethod
    def ConvertToFloatAndFillMissValues(data):
        # Conversion to float
        columns_to_keep_as_int = ['Population', 'Area (sq. mi.)']
        columns_to_skip = ['Region', 'Country'] + columns_to_keep_as_int

        for col in data.columns:
            if col not in columns_to_skip and data[col].dtype == 'O':
                data[col] = data[col].str.replace(',', '').astype(float)

        # Fill missing values
        for col in data.columns.values:
            if data[col].isnull().sum() == 0:
                continue
            if col == 'Climate':
                guess_values = data.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
            else:
                guess_values = data.groupby('Region')[col].median()
            for region in data['Region'].unique():
                data[col].loc[(data[col].isnull()) & (data['Region'] == region)] = guess_values[region]

        return data

    @staticmethod
    def AverageRegionsGDPLiteracyAgriculture(data):
        # Calculate the median for the specified columns by region
        result = data.groupby('Region')[['GDP ($ per capita)', 'Literacy (%)', 'Agriculture']].median()
        return result

    # Define a function to join all countries' data within each region
    @staticmethod
    def join_countries(data):
        return ', '.join(data.astype(str))

    @staticmethod
    def DataAgg(data):
        # Group the DataFrame by 'Region' and apply the join_countries function to aggregate country data
        region_data = data.groupby('Region').agg({
            'Country': HelperClass.join_countries,
            'Population': 'sum',
            'Area (sq. mi.)': 'sum',
            'Pop. Density (per sq. mi.)': 'mean',
            'Coastline (coast/area ratio)': 'mean',
            'Net migration': 'mean',
            'Infant mortality (per 1000 births)': 'mean',
            'GDP ($ per capita)': 'mean',
            'Literacy (%)': 'mean',
            'Phones (per 1000)': 'mean',
            'Arable (%)': 'mean',
            'Crops (%)': 'mean',
            'Other (%)': 'mean',
            'Climate': HelperClass.join_countries,
            'Birthrate': 'mean',
            'Deathrate': 'mean',
            'Agriculture': 'mean',
            'Industry': 'mean',
            'Service': 'mean'
        })

        # Reset the index to have 'Region' as a regular column
        region_data.reset_index(inplace=True)
        return region_data

    @staticmethod
    def plot_gdp_bar_chart(data):
        fig, ax = plt.subplots(figsize=(16, 6))
        top_gdp_countries = data.sort_values('GDP ($ per capita)', ascending=False).head(15)
        mean = pd.DataFrame({'Country': ['World mean'], 'GDP ($ per capita)': [data['GDP ($ per capita)'].mean()]})
        gdps = pd.concat([top_gdp_countries[['Country', 'GDP ($ per capita)']], mean], ignore_index=True)
        sns.barplot(x='Country', y='GDP ($ per capita)', data=gdps, palette='Set1')
        ax.set_xlabel(ax.get_xlabel(), labelpad=15)
        ax.set_ylabel(ax.get_ylabel(), labelpad=30)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)
        plt.xticks(rotation=45)

        # Display the plot in Streamlit
        st.pyplot(fig)

    @staticmethod
    def AsiaFiveRegionGDP(data):
        top_five_asia_countries_literacy = data[data['Region'].str.strip() == 'ASIA (EX. NEAR EAST)'].nlargest(5, 'Literacy (%)')
        top_five_asia_countries_literacy = top_five_asia_countries_literacy[['Country', 'Literacy (%)', 'GDP ($ per capita)']]
        labels = top_five_asia_countries_literacy['Country']
        literacy_rates = top_five_asia_countries_literacy['Literacy (%)']
        gdp_values = top_five_asia_countries_literacy['GDP ($ per capita)']

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Create a pie chart for literacy rates
        axes[0].pie(literacy_rates, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Literacy Rates')

        # Create a pie chart for GDP per capita
        axes[1].pie(gdp_values, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('GDP Per Capita')

        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def EachReginGDP(data):
        # Group the DataFrame by 'Region' and calculate the mean GDP for each region
        region_gdp = data.groupby('Region')['GDP ($ per capita)'].mean()

        # Get the regions and mean GDP values
        regions = region_gdp.index
        mean_gdp = region_gdp.values

        # Calculate the number of subplots needed
        num_subplots = len(regions)
        num_cols = 5  # Set the number of columns for the grid

        # Calculate the number of rows needed
        num_rows = (num_subplots - 1) // num_cols + 1

        # Create the grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows), constrained_layout=True)
        axes = axes.ravel()  # Flatten the 2D array of subplots

        # Customize parameters for better readability and spacing
        colors = plt.cm.tab20c(np.arange(20))
        autopct = '%1.1f%%'
        shadow = True

        for i in range(num_subplots):
            ax = axes[i]

            # Get the countries in the current region
            countries = data[data['Region'] == regions[i]]

            # Calculate the top 5 countries with the highest GDP in the current region
            top_countries = countries.nlargest(5, 'GDP ($ per capita)')

            # Get the top 5 countries and their mean GDP values
            country_names = top_countries['Country']
            country_gdp = top_countries['GDP ($ per capita)']

            # Generate colors for the top 5 countries
            region_colors = colors[:len(country_names)]

            # Define explode based on the number of countries in the region
            explode = [0] * len(country_names)

            ax.pie(country_gdp, labels=country_names, autopct=autopct, startangle=90, colors=region_colors, shadow=shadow, explode=explode)
            ax.set_aspect('equal')  # Ensure the pie is drawn as a circle
            ax.set_title(f'{regions[i]} Region', fontsize=14)

        # Hide any remaining empty subplots
        for i in range(num_subplots, num_cols * num_rows):
            fig.delaxes(axes[i])

        # Add some space between the plots
        plt.subplots_adjust(wspace=0.5)

        # Show the pie charts
        plt.suptitle("Top 5 GDP Distribution by Region", fontsize=16)
        st.pyplot(fig)

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
