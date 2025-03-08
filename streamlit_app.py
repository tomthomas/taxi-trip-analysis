import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Parquet file
parquet_file = 'data/yellow_tripdata_2024-01.parquet'
df = pd.read_parquet(parquet_file)

# Title and Introduction
st.title("🚕 Taxi Trip Analysis")
st.write("Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/).")

# Clean the Data
num_cols = df.select_dtypes(include=[np.number])

# Find outliers using IQR for each numeric column
Q1 = num_cols.quantile(0.25)
Q3 = num_cols.quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = (num_cols < lower_bound) | (num_cols > upper_bound)

# Find rows with outliers
outlier_rows = outliers.any(axis=1)

# Remove rows with outliers
df_cleaned = df[~outlier_rows]

# Remove rows with NaN values
df_cleaned = df_cleaned.dropna()

# Dropdown for selecting Pickup Location
pickup_location = st.selectbox('Select Pickup Location', df_cleaned['PULocationID'].unique())

# Filter data based on Pickup Location
filtered_data = df_cleaned[df_cleaned['PULocationID'] == pickup_location]

# Button for running analysis
if st.button('Analyze Data'):
    st.write(f"Showing data for Pickup Location: {pickup_location}")
    st.write(filtered_data.head())  # Show first few rows of the filtered data
    
    # 1. Trip Distance Distribution
    st.subheader("Trip Distance Distribution")
    plt.figure(figsize=(10, 5))
    sns.histplot(df_cleaned["trip_distance"], bins=10, kde=True)
    plt.title("Trip Distance Distribution")
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # 2. Distribution of Trip Distance for Filtered Data
    st.subheader("Filtered Trip Distance Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_data['trip_distance'], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Trip Distances")
    st.pyplot(fig)

    # 3. Average Fare Collected by Pickup Location
    st.subheader("Average Fare Collected by Pickup Location")
    df_grouped_locid = df_cleaned.groupby("PULocationID")["total_amount"].mean().reset_index()
    df_grouped_locid = df_grouped_locid.sort_values(by="total_amount", ascending=False)
    
    plt.figure(figsize=(30, 5))
    sns.barplot(data=df_grouped_locid, x="PULocationID", y="total_amount", color="skyblue", order=df_grouped_locid["PULocationID"])
    plt.xticks(rotation=90)
    plt.title("Average Fare Collected by Pickup Location")
    plt.xlabel("Pickup Location ID")
    plt.ylabel("Average Fare Collected ($)")
    st.pyplot(plt)

    # 4. Number of Trips Per Hour
    st.subheader("Number of Trips Per Hour")
    df_cleaned["pickup_hour"] = df_cleaned["tpep_pickup_datetime"].dt.hour
    df_hourly = df_cleaned.groupby("pickup_hour").size().reset_index(name="trip_count")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_hourly, x="pickup_hour", y="trip_count", marker="o", color="red")
    plt.title("Number of Trips Per Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Trips")
    plt.xticks(range(0, 24))
    st.pyplot(plt)

    # 5. Average Taxi Trips per Day of the Week
    st.subheader("Average Taxi Trips per Day of the Week")
    df_cleaned['trip_date'] = df_cleaned['tpep_pickup_datetime'].dt.date
    df_cleaned['day_of_week'] = df_cleaned['tpep_pickup_datetime'].dt.dayofweek
    trips_per_day = df_cleaned.groupby(['trip_date', 'day_of_week']).size().reset_index(name='trip_count')
    avg_trips_per_weekday = trips_per_day.groupby('day_of_week')['trip_count'].mean().reset_index()

    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_trips_per_weekday['day_of_week'] = avg_trips_per_weekday['day_of_week'].map(lambda x: day_labels[x])

    plt.figure(figsize=(10, 5))
    sns.barplot(data=avg_trips_per_weekday, x='day_of_week', y='trip_count', palette='Blues_r')
    plt.title("Average Taxi Trips per Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Average Number of Trips")
    plt.xticks(rotation=45)
    st.pyplot(plt)
