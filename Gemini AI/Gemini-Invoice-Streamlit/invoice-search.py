import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

st.set_page_config(page_title="Gemini Invoice Demo")

@st.cache_resource
def configure_access():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro-vision')
    return model

def input_image_process(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def main_code(model):
    
    st.header("Gemini Invoice Extract Application")
    
    uploaded_file = st.file_uploader("Choose Invoice image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        input=st.text_input("Input Prompt: ",key="input")
        submit=st.button("Answer The Input Prompt")

        prompt = "You are an expert in understanding invoices. You will receive input images as invoices.you will have to answer questions based on the input image"
    

        ## If ask button is clicked
        if submit:
            image_data = input_image_process(uploaded_file)
            response = model.generate_content([input,image_data[0],prompt])

            st.subheader("The Response is")
            st.write(response.text)

if __name__=="__main__":
    model=configure_access()
    main_code(model)

