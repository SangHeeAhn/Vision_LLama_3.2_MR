import streamlit as st
import requests
from PIL import Image
import io
import base64

st.title("Medical Image Tumor Detection")

# Hugging Face API Token 입력
hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"

def query_huggingface(image_bytes, prompt, token):
    if not token:
        st.error("❌ API Token is missing! Please enter a valid Hugging Face API token.")
        return {"error": "API token missing"}

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "image": encoded_image
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 401:
            st.error("❌ Unauthorized! Invalid Hugging Face API token.")
            return {"error": "Invalid API token"}
        elif response.status_code != 200:
            st.error(f"⚠️ API Error {response.status_code}: {response.text}")
            return {"error": f"API Error {response.status_code}"}
        
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Request failed: {str(e)}")
        return {"error": "Request failed"}

uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])
prompt = st.text_area(
    "Enter your prompt:",
    "You are a medical imaging expert trained to detect brain tumors in MRI scans..."
)

if uploaded_file and st.button("Run Model"):
    # 파일 업로드
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # 바이트 배열로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 모델 호출
    result = query_huggingface(img_byte_arr, prompt, hf_token)

    # Raw 결과 먼저 확인
    st.subheader("Raw Model Response:")
    st.json(result)

    # 결과 분석 및 처리
    if isinstance(result, dict) and "error" in result:
        st.write("API returned an error:", result["error"])
    elif isinstance(result, dict):
        generated_text = result.get("generated_text", "No generated text")
        st.subheader("Generated Text:")
        st.write(generated_text)
    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        generated_text = result[0].get("generated_text", "No generated text")
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.write("No recognized structure in result.")
