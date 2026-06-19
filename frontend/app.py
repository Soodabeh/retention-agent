import datetime
import requests
import streamlit as st
import pandas as pd

# --- UI Header ---
'''
# TaxiFareModel Front
'''
st.markdown('### Parameters of the ride')

# --- Inputs ---
d = st.date_input("When is your ride", datetime.date(2019, 7, 6))
t = st.time_input('What time is your ride', datetime.time(8, 45))
dt = datetime.datetime.combine(d, t).isoformat()

# Added default values to make testing easier
pickup_longitude = st.number_input('pickup_longitude', value=-73.950655)
pickup_latitude = st.number_input('pickup_latitude', value=40.783282)
dropoff_longitude = st.number_input('dropoff_longitude', value=-73.984365)
dropoff_latitude = st.number_input('dropoff_latitude', value=40.769802)
passenger_count = st.number_input('Passenger count', min_value=1, max_value=8, value=1)


# --- Map Visualization ---
st.markdown('### Ride Map')

# Create a DataFrame with the coordinates named specifically 'lat' and 'lon' for st.map()
map_data = pd.DataFrame({
    'lat': [pickup_latitude, dropoff_latitude],
    'lon': [pickup_longitude, dropoff_longitude]
})

# Render the map
st.map(map_data)

'''
## Retrieve Prediction
'''
url = 'https://taxifare.lewagon.ai/predict'

if url == 'https://taxifare.lewagon.ai/predict':
    st.markdown('*Note: Using the default Le Wagon API.*')

# --- API Call ---
# A button ensures the API is only called when the user is ready
if st.button("Get Fare Prediction"):

    INPUT_DICT = {
        "pickup_datetime": dt,
        "pickup_longitude": float(pickup_longitude),
        "pickup_latitude": float(pickup_latitude),
        "dropoff_longitude": float(dropoff_longitude),
        "dropoff_latitude": float(dropoff_latitude),
        "passenger_count": int(passenger_count)
    }

    try:
        # Use 'params' for GET requests to append data to the URL query string
        response = requests.get(url, params=INPUT_DICT)
        response.raise_for_status() # Catches HTTP errors (400, 500, etc.)

        prediction = response.json()

        # Displaying the prediction cleanly
        if 'fare' in prediction:
            st.success(f"Predicted Fare: ${prediction['fare']:.2f}")
        else:
            st.warning("Fare not found in the response.")

        # Optional: Hide raw JSON behind an expander to keep the UI clean
        with st.expander("View raw JSON response"):
            st.json(prediction)

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to retrieve prediction from API: {e}")
