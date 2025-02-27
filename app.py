import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import base64
import re

st.title("Medical Image Tumor Detection")

# 정규식을 이용해 (x1, y1)부터 (x2, y2)까지 좌표를 추출하는 함수
def parse_coordinates(generated_text):
    """
    예: "Tumor detected at (100, 150) to (200, 250) ..."
    """
    pattern = r"\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\)"
    match = re.search(pattern, generated_text)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1, y1, x2, y2)
    return None

# 좌표로 추출된 영역을 빨간색 사각형(투명도 포함)으로 강조하는 함수
def highlight_tumor_on_image(image, coords):
    """
    coords: (x1, y1, x2, y2)
    PIL.ImageDraw.Draw(image, 'RGBA')를 사용해 투명 오버레이를 그림.
    """
    if not coords:
        return image  # 좌표가 없으면 원본 그대로 반환

    # 'RGBA' 모드로 그리기 위해 복사본을 만들거나, 이미지가 'RGBA'가 아니라면 변환
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    draw = ImageDraw.Draw(image, 'RGBA')
    x1, y1, x2, y2 = coords

    # 사각형 내부를 반투명 빨간색(fill)으로 채우고, 테두리(outline)는 빨간색, 두께 3
    fill_color = (255, 0, 0, 100)   # 빨간색 + 투명도 (0~255)
    outline_color = (255, 0, 0, 255)
    draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=3)

    return image

# Hugging Face API 정보
hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"

# 모델 요청
def query_huggingface(image_bytes, prompt):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,   # 문자열만
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

# 파일 업로더
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])
prompt = st.text_area(
    "Enter your prompt:",
    "You are a medical imaging expert trained to detect brain tumors in MRI scans..."
)

# 실행
if uploaded_file and st.button("Run Model"):
    # 업로드된 이미지 열기
    image = Image.open(uploaded_file).convert("RGB")

    # 업로드된 이미지 바로 표시
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # 바이트 배열로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 모델 호출
    result = query_huggingface(img_byte_arr, prompt)

    st.subheader("Model Response (Raw JSON):")
    st.json(result)  # 모델 응답을 JSON 형태로 출력

    # 아래는 모델 응답에서 'generated_text'라는 필드가 있다고 가정한 예시
    generated_text = result.get("generated_text", "")
    if generated_text:
        # 1) 텍스트에서 좌표 파싱
        coords = parse_coordinates(generated_text)

        # 2) 원본 이미지에 표시
        highlighted_image = highlight_tumor_on_image(image, coords)

        st.subheader("Highlighted MRI Image")
        st.image(highlighted_image, use_column_width=True)
    else:
        st.write("No 'generated_text' field found or it's empty. Cannot highlight tumor region.")
