import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

def configure_access():
    # This line - you can comment - This is fix
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS']="gen-lang-client-0973766600-1e5a6e73538e.json"

    # create .env file and copy paste key in the file like: GOOGLE_API_KEY=XXXXXXXXXXXXXXXXXXX
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    print("models list")
    for m in genai.list_models():
        print(m.name)

configure_access()
model=genai.GenerativeModel("gemini-1.0-pro-latest")

st.title("Gemini AI - GenAi  - Sentiment Analysis")
query=st.text_input("Enter the Review :")
button=st.button("Search")
if button:
    response = model.generate_content(["Write the answer for give given review  query - Classify into positive or  negative , write only one", query])
    print(response.text)
    if response.text=="Positive" or response.text=="positive":
        st.header('Sentiment Classification :green[Positive]')
    else:
        st.header('Sentiment Classification :red[Negative]')