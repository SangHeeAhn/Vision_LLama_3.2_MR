import streamlit as st
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
import numpy as np
import re

# Streamlit UI 설정
st.title("Medical Image Tumor Detection")

# Hugging Face 로그인 토큰 입력
hf_token = st.text_input("Enter your Hugging Face token:", type="password")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)

# 모델 불러오기
@st.cache_resource()
def load_model():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

model, processor = load_model()

# 사용자 입력 받기
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

prompt = st.text_area("Enter your prompt:", "You are a medical imaging expert trained to detect brain tumors in MRI scans...")

# Bounding Box 및 Masking 관련 함수
def create_binary_mask(image_size, bounding_boxes):
    mask = np.zeros(image_size, dtype=np.uint8)
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        x1, x2 = max(0, min(x1, image_size[1])), max(0, min(x2, image_size[1]))
        y1, y2 = max(0, min(y1, image_size[0])), max(0, min(y2, image_size[0]))
        mask[y1:y2, x1:x2] = 255
    return mask

def extract_coordinates_from_response(response):
    pattern = r'\((\d+),\s*(\d+)\)'
    matches = re.findall(pattern, response)
    coordinates = [(int(x), int(y)) for x, y in matches]
    return coordinates

def convert_to_bounding_boxes(coordinates):
    if len(coordinates) < 2:
        raise ValueError("At least two coordinates are required to define a bounding box.")
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    return [(x1, y1, x2, y2)]

# 모델 실행 버튼
if uploaded_file and st.button("Run Model"):
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    images = [image]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=images, text=text, return_tensors="pt").to(model.device)
    
    generate_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.5)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:-1]
    generated_text = processor.decode(generate_ids[0], clean_up_tokenization_spaces=False)
    
    coordinates = extract_coordinates_from_response(generated_text)
    bounding_boxes = convert_to_bounding_boxes(coordinates)
    
    binary_mask = create_binary_mask((image.height, image.width), bounding_boxes)
    binary_image = Image.fromarray(binary_mask, mode="L")
    
    st.subheader("Model Response:")
    st.write(generated_text)
    
    st.subheader("Generated Binary Mask:")
    st.image(binary_image, caption="Binary Mask", use_column_width=True)
