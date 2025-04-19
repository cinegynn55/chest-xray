import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# 👉 스타일 설정
st.set_page_config(page_title="AI 폐렴 예측기", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4B9CD3;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #aaa;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# 👉 타이틀 영역
st.markdown('<div class="title">🩻 AI 폐렴 예측기</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">엑스레이 이미지를 업로드하면, 폐렴 여부를 예측합니다.</div>', unsafe_allow_html=True)

# 👉 모델 및 전처리
class_names = ['NORMAL', 'PNEUMONIA']

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("my_resnet18_half.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        return class_names[predicted.item()], probs[0][predicted.item()].item()

# 👉 파일 업로드
uploaded_file = st.file_uploader("### 🔽 여기에 엑스레이 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 업로드한 이미지", use_column_width=True)

    image_tensor = transform_image(image)
    model = load_model()

    with st.spinner("🧠 AI가 분석 중입니다..."):
        label, confidence = predict(model, image_tensor)

    st.success(f"✅ 예측 결과: **{label}** ({confidence * 100:.2f}% 확신)")

# 👉 푸터
st.markdown('<div class="footer">© 2025 Cinegynn AI Project</div>', unsafe_allow_html=True)