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
        "inputs": prompt,     # 문자열
        "image": encoded_image  # Base64 인코딩된 이미지
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
    "You are a medical imaging expert trained to detect brain tumors in MRI scans. Your task is to find and precisely localize the tumor. Given an MRI image, identify the tumor region and return the bounding box coordinates in the format (x1, y1), (x2, y2). Additionally, explain why this region is considered a tumor based on the image features. The tumor is usually in a high-intensity region (bright area in grayscale). Be accurate and return only one bounding box"
)

if uploaded_file and st.button("Run Model"):
    # 파일 업로드
    image = Image.open(uploaded_file).convert("RGB")
    # 업로드된 이미지 표시
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # 바이트 배열로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 모델 호출
    result = query_huggingface(img_byte_arr, prompt)

    # Raw 결과 먼저 확인
    st.subheader("Raw Model Response:")
    st.json(result)

    # 다양한 응답 형태에 대응
    generated_text = ""

    # 1) 에러 구조인지 확인
    if isinstance(result, dict) and "error" in result:
        st.write("API returned an error:", result["error"])
    
    # 2) 딕셔너리 형태
    elif isinstance(result, dict):
        generated_text = result.get("generated_text", "")
        st.subheader("Generated Text:")
        st.write(generated_text)

    # 3) 리스트 형태(예: [{"generated_text": "..."}])
    elif isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if isinstance(first_item, dict):
            generated_text = first_item.get("generated_text", "")
            st.subheader("Generated Text:")
            st.write(generated_text)
        else:
            st.write("Unexpected structure in the first item of the list.")
    else:
        st.write("No recognized structure in result.")
