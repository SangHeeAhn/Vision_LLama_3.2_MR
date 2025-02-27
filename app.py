import streamlit as st
import requests
from PIL import Image
import io

# Streamlit UI 설정
st.title("Medical Image Tumor Detection")

# Hugging Face API Key 입력
hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")

# 모델 엔드포인트
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"

# API 요청 함수
def query_huggingface(image, prompt):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": {
            "image": image,
            "text": prompt
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# 이미지 업로드
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

prompt = st.text_area("Enter your prompt:", "You are a medical imaging expert trained to detect brain tumors in MRI scans...")

# 실행 버튼
if uploaded_file and st.button("Run Model"):
    image = Image.open(uploaded_file).convert("RGB")

    # 이미지를 바이트 형태로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # API 요청
    result = query_huggingface(img_byte_arr, prompt)

    # 결과 출력
    st.subheader("Model Response:")
    st.write(result)
