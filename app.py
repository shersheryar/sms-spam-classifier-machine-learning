import streamlit as st
import pickle 
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
ps = PorterStemmer()
# Download NLTK data with error handling
nltk.data.path.append('./nltk_data')
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
# except Exception as e:
#     st.error(f"Failed to download NLTK data: {e}")
# fucntions
def transform_text(text):
    text = text.lower()  # converting to lowercase
    text = nltk.word_tokenize(text)  # tokenizing the text
    text = [word for word in text if word.isalnum()]  # removing special characters
    text = [word for word in text if word not in stopwords.words("english") and word not in punctuation] 
    ps = PorterStemmer()  # creating an object of the PorterStemmer class
    text = [ps.stem(word) for word in text]  # stemming the words
    return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('spam_model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message here", key="input")

if st.button("Predict"):
    # preprocess the input
    transform_sms = transform_text(input_sms)
    # vectorize the input
    vector_input = tfidf.transform([transform_sms])
    # predict
    result = model.predict(vector_input)[0]
    # display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam") 


 
