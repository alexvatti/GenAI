import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
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
model=genai.GenerativeModel("gemini-1.5-flash")

st.title("Gemini AI - GenAi - Vision")
uploaded_file = st.file_uploader("Choose a image file",type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img=Image.open(uploaded_file)
    st.write(img)
    prompt=st.text_input("Enter what you want to find image:")
    button=st.button("Search")
    if button:
        response=model.generate_content([prompt,img])
        st.markdown(response.parts)

