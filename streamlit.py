import streamlit as st
import requests
import json
import pandas as pd
import torch
import pickle
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Define preprocess_text function
def preprocess_text(df):
    df['heading'] = (df['heading'] + ' ' + df['section']).fillna('').astype(str)
    df['heading'] = df['heading'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).lower())
    df['heading'] = df['heading'].apply(word_tokenize)
    df['heading'] = df['heading'].apply(lambda tokens: ' '.join([word for word in tokens if word not in stop_words]))
    df = df.drop(['city', 'section'], axis=1)
    return df[['heading']]

# Load PyTorch model
class AdvancedNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, output_size)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_pytorch_model(model_path, input_size, output_size):
    model = AdvancedNN(input_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Load PyTorch model using the function
model_path = "best_model.pth"
pytorch_model = load_pytorch_model(model_path, input_size=12953, output_size=16)

# Load TfidfVectorizer from file
def load_tfidf_vectorizer(file_path):
    with open(file_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_vectorizer

# Load TfidfVectorizer
tfidf_vectorizer = load_tfidf_vectorizer('tfidf_vectorizer_model.pkl')

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Streamlit UI
st.title("Craigslist Post Category Prediction")

# Description
st.write("Craigslist Post Category Prediction aims to predict the category of each post published on the Craigslist classified ads platform.")

# Instructions for uploading JSON file
st.write("If you want to try the 'Upload JSON File' feature, you can directly download the sample JSON file, please click the link below:")
# Short link to the JSON file
json_file_link = "[Click here to download the JSON file](https://drive.google.com/uc?id=1KgYBe5XTwgMLQJjOP-jYfK5IzgJj-Xbv)"
st.markdown(json_file_link)

# Input method selection
input_method = st.radio("Select Input Method", ["Form Input", "Upload JSON File"])

# Function to make prediction
def make_prediction(data):
    # Combine 'heading' and 'section' columns
    combined_heading = (data['heading'] + ' ' + data['section']).strip()

    # Save the original heading before preprocessing
    original_heading = data['heading']

    # Apply text preprocessing
    preprocessed_data = preprocess_text(pd.DataFrame([data]))
    preprocessed_heading = preprocessed_data['heading'].iloc[0]

    # Transform data using TfidfVectorizer
    text_tfidf = tfidf_vectorizer.transform(preprocessed_data['heading'])
    X_new_tensor = torch.tensor(text_tfidf.toarray(), dtype=torch.float32)

    # Make predictions using PyTorch model
    pytorch_model.eval()
    with torch.no_grad():
        outputs = pytorch_model(X_new_tensor)
        _, predicted = torch.max(outputs, 1)

    # Mapping classes to the given context
    class_mapping = {0: "activities", 1: "appliances", 2: "artists", 3: "automotive", 4: "cell-phones",
                     5: "childcare", 6: "general", 7: "household-services", 8: "housing",
                     9: "photography", 10: "real-estate", 11: "shared", 12: "temporary",
                     13: "therapeutic", 14: "video-games", 15: "wanted-housing"}

    # Determine the prediction result based on the correct class
    if predicted.item() in class_mapping:
        category = class_mapping[predicted.item()]
    else:
        category = "Unknown Category"

    # Create the final result dictionary
    result = {
        "heading": original_heading,
        "city": data['city'],
        "section": data['section'],
        "category": category
    }

    return result

# Form input
if input_method == "Form Input":
    city = st.selectbox("Select City", ['dubai.en', 'kolkata.en', 'frankfurt.en', 'zurich.en', 'geneva.en', 'paris.en',
                                        'bangalore', 'singapore', 'delhi', 'mumbai', 'hyderabad', 'manchester',
                                        'london', 'chicago', 'seattle', 'newyork'])
    section = st.selectbox("Select Section", ['services', 'community', 'housing', 'for-sale'])
    heading = st.text_area("Enter Heading")

    if st.button("Predict"):
        data = {'city': city, 'section': section, 'heading': heading}
        result = make_prediction(data)

        st.write("Prediction Result:")
        st.write(result)

# Upload JSON file
elif input_method == "Upload JSON File":
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

    if uploaded_file is not None:
        file_contents = uploaded_file.getvalue()
        json_data = file_contents.decode("utf-8")

        if st.button("Predict"):
            try:
                data_list = [json.loads(entry) for entry in json_data.strip().split('\n')]
                results = []

                for data in data_list:
                    result = make_prediction(data)
                    results.append(result)

                st.write("Prediction Results:")
                for result in results:
                    st.write(result)

                output_filename = "prediction_results.json"
                with open(output_filename, "w") as output_file:
                    json.dump(results, output_file)

                st.markdown(f"**[Download Prediction Results JSON]({output_filename})**")

            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")
