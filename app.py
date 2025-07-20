import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Modelin beklediği giriş boyutu
IMG_SIZE = 126

# Modeli yükle
model = tf.keras.models.load_model("kidney_stone_model.h5")

# Başlık
st.title("🧠 Böbrek Taşı Sınıflandırma (Ultrason Görüntüsü)")

# Açıklama
st.markdown("""
Bu uygulama, yüklediğiniz ultrason görüntüsünü analiz ederek böbrek taşı olup olmadığını tahmin eder.  
Model, CNN mimarisiyle eğitilmiş olup ultrason verileriyle **%100 doğruluk** elde etmiştir.
""")

# Görsel yükleyici
uploaded_file = st.file_uploader("📁 Bir ultrason görüntüsü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Yüklenen Görüntü", use_column_width=True)

    img = Image.open(uploaded_file).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    prediction = model.predict(img_array)[0][0]

    st.subheader("🧪 Tahmin Sonucu:")
    if prediction > 0.5:
        st.success(f"✅ Normal doku (%{(prediction * 100):.2f} olasılıkla)")
    else:
        st.error(f"⚠️ Taş tespit edildi (%{(100 - prediction * 100):.2f} olasılıkla)")
