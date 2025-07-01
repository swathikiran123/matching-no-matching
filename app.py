import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Title of the app
st.title("Cluster Prediction App")

# Add a description
st.write("""
This app predicts whether your input matches a predefined group based on a machine learning model.
Fill out the following questions, and click "Predict" to see if your response is a match or not!
""")

# Load the pre-trained model
try:
    model = tf.keras.models.load_model('elite_28b_weight_knn.h5')
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define the categories for user input
ordinal_categories = {
    "How spontaneous are you?": ["Very planned", "Mostly planned", "Balanced", "Mostly spontaneous", "Very spontaneous"],
    "How open are you to trying new things?": ["Not open at all", "Slightly hesitant", "Sometimes open", "Mostly open", "Very adventurous"],
    "How much personal space do you need in a relationship?": ["Very little", "A little", "Moderate", "Quite a bit", "A lot"],
    "How emotionally expressive are you?": ["Very reserved", "Slightly reserved", "Balanced", "Mostly expressive", "Very expressive"],
    "How important is having similar long-term goals?": ["Not important", "Slightly important", "Neutral", "Important", "Very important"],
}

nominal_categories = {
    "Do you enjoy giving or receiving surprises?": ["Dislike it", "Hate it", "Like it", "Love it", "Neutral"],
    "How important is music taste compatibility?": ["Doesn't matter at all", "Important but not a deal-breaker", "Neutral", "Slightly important", "Very important, must match mine"],
    "What’s your preferred mode of communication?": ["Calling", "Face to Face", "Texting", "Video Calls"],
    "What’s your ideal time to hang out?": ["Afternoon", "Evening", "Late night", "Morning"],
    "What is your ideal weekend plan?": ["Outdoor Adventures", "Partying", "Staying Indoors/Chilling", "Watching Movies"]
}

# Initialize label encoders for each category
encoders = {}
for key, categories in {**ordinal_categories, **nominal_categories}.items():
    le = LabelEncoder()
    le.fit(categories)
    encoders[key] = le

# Section for user input
st.subheader("Please answer the following questions:")

user_input = {}

# Collect user input for each question
for key, categories in {**ordinal_categories, **nominal_categories}.items():
    user_input[key] = st.selectbox(key, categories)

# Add a section for predictions
st.subheader("Prediction")

if st.button("Predict"):
    try:
        # Encode user input based on pre-trained encoders
        encoded_input = {key: encoders[key].transform([user_input[key]])[0] for key in user_input}
        input_df = pd.DataFrame([encoded_input])

        # Predict using the model
        prediction = model.predict(input_df)[0][0]
        result = "Match" if prediction >= 0.5 else "No Match"

        # Show result
        st.write(f"**Prediction**: {result}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Styling for the app
st.markdown("""
    <style>
        .css-1v3fvcr { 
            font-size: 1.5em; 
        }
        .stButton>button {
            background-color: #0099ff;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)
