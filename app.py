import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import pandas as pd
import numpy as np
import plotly.express as px

# ===== Device setup =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load disaster labels.csv once =====
LABELS_DF = pd.read_csv(r"C:\Users\DELL\Desktop\satellite_img_classification\data\disaster\labels.csv")

@st.cache_resource
def load_model(task):
    if task == "land_cover":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(
            torch.load("models/resnet18_land_cover.pth", map_location=DEVICE, weights_only=True)
        )
        class_names = sorted(os.listdir("data/land_cover"))
    else:
        checkpoint = torch.load("models/vit_disaster.pth", map_location=DEVICE, weights_only=True)
        class_names = checkpoint["class_names"]
        num_classes = len(class_names)
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, class_names

def preprocess_image(img, task):
    if task == "land_cover":
        t = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    return t(img).unsqueeze(0).to(DEVICE)

# ===== Prediction with confidence =====
def predict(model, img_tensor, class_names):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        confidence = float(np.max(probs))
        pred_idx = np.argmax(probs)
        predicted_class = class_names[pred_idx]
    return predicted_class, confidence, probs

def main():
    st.set_page_config(page_title="GeoVision", page_icon="🌍", layout="wide")

    # ===== Styling =====
    st.markdown("""
        <style>
        .title-font {font-size: 60px !important;font-weight: bold;text-align: center;color: #2C3E50;}
        .tagline-font {font-size: 25px !important;text-align: center;color: #666666;margin-bottom: 20px;}
        section[data-testid="stSidebar"] {width: 40%;background-color: #7EC8E3;color: white;}
        .sidebar-sub {font-size: 18px;margin-top: -5px;margin-bottom: 15px;color: white;}
        .pred-box {background-color: #2ecc71;color: white;padding: 10px 20px;border-radius: 8px;
                   font-size: 20px;font-weight: 600;display: inline-block;margin-top: 10px;text-align: center;}
        /* Transparent radio buttons */
        div.row-widget.stRadio > div {background-color: transparent; color: white;}
        </style>
    """, unsafe_allow_html=True)

    # ----- MAIN TITLE -----
    st.markdown("<h1 class='title-font'>🌍 GeoVision</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline-font'>Deep Learning-Based Satellite Image Classifier</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ----- SIDEBAR -----
    st.sidebar.markdown("<div style='font-size:24px; font-weight:700; color:yellow;'>Task Selection</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-sub'>Choose a classification model to use</div>", unsafe_allow_html=True)
    task = st.sidebar.radio("Choose Task", ["land_cover", "disaster"])

    # --- Model Info (Both Models) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:24px; font-weight:700; color:yellow;'>⚙️ Model Info</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size:18px; font-weight:700; color:white;'>🏞️ Land Cover (ResNet18)</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size:18px; font-weight:700; color:white;'>🌪️ Disaster (ViT-B16)</div>", unsafe_allow_html=True)

    # --- Theme Selector ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Page Theme")
    theme = st.sidebar.selectbox("Choose Theme", ["Default", "Ocean", "Sunset", "Forest"])

    if theme == "Sunset":
        st.markdown("<style>.stApp {background-color: #FFE5B4;}</style>", unsafe_allow_html=True)
    elif theme == "Forest":
        st.markdown("<style>.stApp {background-color: #C1E1C1;}</style>", unsafe_allow_html=True)
    elif theme == "Ocean":
        st.markdown("<style>.stApp {background-color: #e6f7ff;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>.stApp {background-color: #F8FBFF;}</style>", unsafe_allow_html=True)

    # ----- CENTERED FILE UPLOADER -----
    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("🔍 Predict"):
                model, class_names = load_model(task)
                img_tensor = preprocess_image(image, task)
                predicted, confidence, probs = predict(model, img_tensor, class_names)

                # Prediction box
                st.markdown(f"<div class='pred-box'>Prediction: {predicted}</div>", unsafe_allow_html=True)

                # Confidence Text in black
                st.markdown(f"<p style='font-size:20px; color:black; font-weight:600;'>Confidence: {confidence*100:.2f}%</p>", unsafe_allow_html=True)

                # Custom Confidence Progress Bar in deep purple
                st.markdown(f"""
                <div style="background-color:#e0e0e0; border-radius:10px; padding:3px; margin-top:5px; margin-bottom:5px;">
                  <div style="width:{confidence*100}%; background-color:#6A0DAD; height:25px; border-radius:10px; text-align:center; color:white; font-weight:bold;">
                    {confidence*100:.2f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Vertical bar chart of probabilities
                prob_df = pd.DataFrame({"Class": class_names, "Probability": probs})
                fig_bar = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    title="Prediction Probability Distribution",
                )
                fig_bar.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
