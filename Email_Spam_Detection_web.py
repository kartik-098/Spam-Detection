import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the model and vectorizer
model = pickle.load(open("naive_bayes_model.pkl", "rb"))
cv = pickle.load(open("count_vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
    text = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)
    text = re.sub(r'£|\$', 'moneysymb', text)
    text = re.sub(r'\b(\+?\d[\d\s\-()]{7,})\b', 'phonenumbr', text)
    text = re.sub(r'\d+(\.\d+)?', 'numbr', text)
    text = re.sub(r'[^\w\d\s]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Streamlit UI
st.title("Email Spam Detector")

input_text = st.text_area("Paste your email content here:")

if st.button("Check Spam"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        vector = cv.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0]
        if prediction == 1:
            st.error("This email is SPAM!. Don't Trust.")
        else:
            st.success("This email is NOT SPAM.")