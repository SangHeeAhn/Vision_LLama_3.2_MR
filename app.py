import streamlit as st
import requests
from PIL import Image
import io
import base64

st.title("Medical Image Tumor Detection")

hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"

def query_huggingface(image_bytes, prompt):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": {
            "image": encoded_image,
            "text": prompt
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    # 1) 상태 코드 확인
    if response.status_code != 200:
        st.write(f"Status Code: {response.status_code}")
        st.write("Response Text:", response.text)
        return {"error": "Non-200 response received"}

    # 2) JSON 파싱 시도
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        st.write("Response was not valid JSON:")
        st.write(response.text)
        return {"error": "JSON decode failed"}

uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])
prompt = st.text_area(
    "Enter your prompt:",
    "You are a medical imaging expert trained to detect brain tumors in MRI scans..."
)

if uploaded_file and st.button("Run Model"):
    image = Image.open(uploaded_file).convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    result = query_huggingface(img_byte_arr, prompt)
    st.subheader("Model Response:")
    st.write(result)
