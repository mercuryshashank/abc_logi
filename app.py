import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')

st.write("Enter the features below to predict delivery delay.")

# Define input fields for each feature based on x.columns
# 'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time'

delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.0, value=10.0)
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being highest)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-3, 1-Good, 2-Moderate, 3-Bad)', 1, 3, 2)
delivery_slot = st.slider('Delivery Slot (1-3, 1-Morning, 2-Afternoon, 3-Evening)', 1, 3, 2)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, value=3)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=5)
road_condition_score = st.slider('Road Condition Score (1-4, 4 being best)', 1, 4, 3)
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, value=5.0)
fuel_efficiency = st.number_input('Fuel Efficiency (km/L)', min_value=0.0, value=15.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60)

# Make prediction when button is clicked
if st.button('Predict Delivery Delay'):
    features = pd.DataFrame([[delivery_distance, traffic_congestion, weather_condition,
                              delivery_slot, driver_experience, num_stops, vehicle_age,
                              road_condition_score, package_weight, fuel_efficiency,
                              warehouse_processing_time]], 
                            columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                                     'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                                     'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                                     'Warehouse_Processing_Time'])
    
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    
    st.subheader('Prediction Result:')
    if prediction == 1:
        st.write("<p style='color:red;'><b>Delivery is likely to be delayed.</b></p>", unsafe_allow_html=True)
    else:
        st.write("<p style='color:green;'><b>Delivery is likely to be on time.</b></p>", unsafe_allow_html=True)
        
    st.write(f"Probability of No Delay (0): {prediction_proba[0]:.2f}")
    st.write(f"Probability of Delay (1): {prediction_proba[1]:.2f}")

