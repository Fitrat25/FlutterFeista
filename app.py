import streamlit as st
import pandas as pd
import joblib


model = joblib.load(open('planner_model.pkl', 'rb'))
encoder = joblib.load(open('encoder.pkl', 'rb'))
label_enc = joblib.load(open('label_encoder.pkl', 'rb'))

st.set_page_config(page_title="FlutterFeista", page_icon="🎈", layout="centered")

st.markdown("<h1 style='text-align: center; color: rgb(239 1 1);'> FlutterFeista: Event Food Category Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

with st.form("event_form"):
    st.subheader(" Event Details")

    col1, col2 = st.columns(2)
    with col1:
        event_type = st.selectbox("Event Type", ["Wedding", "Birthday", "Corporate", "Engagement", "Festival", "Anniversary"])
        guests = st.slider("Number of Guests", 10, 500, 50)
        budget = st.number_input("Total Budget (₹)", value=100000, step=10000)
        duration = st.slider("Event Duration (hrs)", 1, 10, 4)
    
    with col2:
        location = st.selectbox("Location Type", ["Indoor", "Outdoor", "Home"])
        season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Monsoon", "Autumn"])
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        food_type = st.selectbox("Food Type", ["Veg", "Non-Veg", "Mixed"])

    st.subheader(" Guest Preferences")
    cuisine = st.selectbox("Cuisine Preference", ["Indian", "North Indian", "South Indian", "Continental", "Asian", "Mixed"])
    age_group = st.selectbox("Guest Age Group", ["Children", "Adults", "Seniors", "Mixed"])
    service_style = st.selectbox("Service Style", ["Buffet", "Served", "Plated", "Live"])
    special_req = st.text_input("Any Special Requests?", placeholder="e.g. Live counters, dessert bar...")

    submitted = st.form_submit_button("✨ Predict Food Category")

if submitted:
    test_df = pd.DataFrame([{
        'EventType': event_type,
        'Guests': guests,
        'Budget': budget,
        'EventDuration': duration,
        'LocationType': location,
        'Season': season,
        'TimeOfDay': time_of_day,
        'FoodType': food_type,
        'CuisinePreference': cuisine,
        'AgeGroupOfGuests': age_group,
        'ServiceStyle': service_style,
        'SpecialRequests': special_req
    }])

    categorical_cols = ['EventType', 'LocationType', 'Season', 'TimeOfDay', 'FoodType', 'CuisinePreference', 'AgeGroupOfGuests', 'ServiceStyle', 'SpecialRequests']
    numerical_cols = ['Guests', 'Budget', 'EventDuration']

    cat_test = test_df[categorical_cols]
    num_test = test_df[numerical_cols]

    encoded_cat = encoder.transform(cat_test)
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

    X_test_final = pd.concat([encoded_cat_df, num_test.reset_index(drop=True)], axis=1)

    X_test_final = X_test_final.reindex(columns=model.feature_names_in_, fill_value=0)

    y_pred = model.predict(X_test_final)
    final_label = label_enc.inverse_transform(y_pred)[0]

    st.success(f"🔮 Recommended Food Category: **{final_label}**")

    if final_label == "Premium":
        st.balloons()
    elif final_label == "Street":
        st.snow()

    st.markdown("💡 *FlutterFeista combines tradition + taste to give you the perfect food plan!*")

