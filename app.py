import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# 클래스 이름
class_names = ['NORMAL', 'PNEUMONIA']

# 모델 불러오기 함수
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("my_resnet18_epoch3.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# 이미지 전처리
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# 예측 함수
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        return class_names[predicted.item()], probs[0][predicted.item()].item()

# Streamlit 앱 구성
st.title("🩻 AI 폐렴 예측기")
st.write("엑스레이 이미지를 업로드하면, 폐렴 여부를 예측합니다.")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    image_tensor = transform_image(image)
    model = load_model()

    with st.spinner("예측 중..."):
        label, confidence = predict(model, image_tensor)

    st.success(f"✅ 예측 결과: **{label}** ({confidence * 100:.2f}% 확신)")
