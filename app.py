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
    # "inputs"는 문자열만, 이미지는 별도 필드 "image"에 담음
    payload = {
        "inputs": prompt,
        "image": encoded_image
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        st.write(f"Status Code: {response.status_code}")
        st.write("Response Text:", response.text)
        return {"error": "Non-200 response received"}
    
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
    # 파일 업로드
    image = Image.open(uploaded_file).convert("RGB")
    # 업로드된 이미지 출력
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # 바이트 배열로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 모델 호출
    result = query_huggingface(img_byte_arr, prompt)

    # 결과 표시
    st.subheader("Model Response:")
    st.json(result)  # JSON 형태로 보기 좋게 출력
