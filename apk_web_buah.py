import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title = "Prediksi Buah",
    page_icon = ":tangerine:"
)
model = joblib.load("ML_Meprediksi_Buah.joblib")
st.title(":tangerine: Selamat Datang Di Prediksi Buah")

size = st.slider("Size (cm)", 0.0, 27.5, 10.0)
berat = st.slider("Weight (g)", 0.0, 3500.0, 1700.0)
harga = st.slider("Price (₹)", 0.0, 165.0, 80.0)
bentuk = st.pills("Shape", ["round", "oval", "long"], default="round")
warna = st.pills("Color", ["green", "red", "brown", "yellow", "pink", "orange", "purple", "blue"], default="green")
rasa = st.pills("taste", ["sweet", "tangy", "sour"], default="sweet")

prediksi = st.button("Prediksi", type="primary")

if (prediksi):
    data_baru = pd.DataFrame([[size, bentuk, berat, harga, warna, rasa]], columns=['size (cm)', 'shape', 'weight (g)', 'avg_price (₹)', 'color', 'taste'])
    hasil_pred = model.predict(data_baru)[0]
    akurasi = max(model.predict_proba(data_baru)[0])
    menyapa = ''
    if (akurasi >= 0.8):
        menyapa = "itu buah"
    else:
        menyapa = "mungkin itu buah"
    st.success(f"**{menyapa} {hasil_pred}** Dengan Keyakinan {akurasi*100:.2f}%")

