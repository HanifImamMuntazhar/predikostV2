import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import plotly.graph_objects as go

# -------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# -------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Kost Sukabumi",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------------------
# 2. LOAD MODEL, FITUR, & KOORDINAT PETA
# -------------------------------------------------------------
@st.cache_resource
def load_ai_model():
    model = joblib.load('model_rf_sukabumi.pkl')
    fitur_cols = joblib.load('kolom_fitur.pkl')
    return model, fitur_cols

model, fitur_cols = load_ai_model()

koordinat_wilayah = {
    "Baros": [-6.9536, 106.9272], "Cibeureum": [-6.9442, 106.9511],
    "Cikole": [-6.9155, 106.9325], "Citamiang": [-6.9361, 106.9317],
    "Gunung Puyuh": [-6.9189, 106.9186], "Lembursitu": [-6.9600, 106.9131],
    "Warudoyong": [-6.9325, 106.9178], "Cisaat": [-6.9111, 106.8922],
    "Sukaraja": [-6.9147, 106.9667], "Selabintana": [-6.8731, 106.9458]
}

# -------------------------------------------------------------
# 3. SIDEBAR NAVIGASI
# -------------------------------------------------------------
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ("Hitung Prediksi Harga", "Insight & Fun Fact")
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tentang Aplikasi:**\n"
    "Sistem ini ditenagai oleh Machine Learning (Random Forest) "
    "untuk memprediksi harga sewa kost di Sukabumi berdasarkan data riil."
)

# -------------------------------------------------------------
# 4. HALAMAN 1: PREDIKSI HARGA KOST
# -------------------------------------------------------------
if menu == "Hitung Prediksi Harga":
    st.title("Prediksi Harga Kost Sukabumi")
    st.markdown("Masukkan kriteria kost yang kamu inginkan, dan biarkan kami menebak harga sewanya!")

    # --- FORM INPUT ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informasi Dasar")
        lokasi_pilihan = st.selectbox("Pilih Kecamatan (Lokasi):", list(koordinat_wilayah.keys()))
        tipe_pilihan = st.selectbox("Pilih Tipe Kost:", ['Putra', 'Putri', 'Campur'])

    with col2:
        st.subheader("Fasilitas Utama")
        st.write("Centang fasilitas yang tersedia:")
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            fas_ac = st.checkbox("AC")
            fas_wifi = st.checkbox("WiFi")
            fas_km_dalam = st.checkbox("Kamar Mandi Dalam")
        with f_col2:
            fas_kasur = st.checkbox("Kasur")
            fas_kloset = st.checkbox("Kloset Duduk")
            fas_24jam = st.checkbox("Akses 24 Jam")

    st.markdown("---")
    
    # --- LOGIKA PREDIKSI ---
    if st.button("Prediksi Harga Sekarang!", type="primary", use_container_width=True):
        with st.spinner("Tunggu Kami sedang menghitung..."):
            
            input_data = {col: 0 for col in fitur_cols}
            
            nama_kolom_lokasi = f"Lokasi_{lokasi_pilihan}"
            nama_kolom_tipe = f"Tipe_Kost_{tipe_pilihan}"
            
            if nama_kolom_lokasi in input_data: input_data[nama_kolom_lokasi] = 1
            if nama_kolom_tipe in input_data: input_data[nama_kolom_tipe] = 1
            
            if fas_ac and "Fas_AC" in input_data: input_data["Fas_AC"] = 1
            if fas_wifi and "Fas_WiFi" in input_data: input_data["Fas_WiFi"] = 1
            if fas_km_dalam and "Fas_K_Mandi_Dalam" in input_data: input_data["Fas_K_Mandi_Dalam"] = 1
            if fas_kasur and "Fas_Kasur" in input_data: input_data["Fas_Kasur"] = 1
            if fas_kloset and "Fas_Kloset_Duduk" in input_data: input_data["Fas_Kloset_Duduk"] = 1
            if fas_24jam and "Fas_Akses_24_Jam" in input_data: input_data["Fas_Akses_24_Jam"] = 1

            df_input = pd.DataFrame([input_data])
            
            prediksi_log = model.predict(df_input)[0]
            prediksi_final = np.expm1(prediksi_log) 
            
            st.success("Tebakan Selesai! Berikut adalah analisis untuk spesifikasi kost Anda:")
            
            # --- VISUALISASI HASIL ---
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                st.markdown(f"### Gambaran Kamar")
                
                img_path = "assets/kamar_kosongan.jpeg"
                if fas_kasur and fas_ac:
                    img_path = "assets/kamar_ac.jpeg"
                elif fas_kasur and not fas_ac:
                    img_path = "assets/kamar_standar.jpeg"
                    
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.image("https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?w=800", use_container_width=True, caption="*Gambar ilustrasi")
                
                st.info(f"**Insight Properti:**\nUnit indekos bertipe **{tipe_pilihan}** ini berlokasi strategis di **{lokasi_pilihan}**. "
                        f"Kombinasi fasilitas yang Anda pilih menjadikan properti ini masuk dalam segmen pasar yang kompetitif di wilayah tersebut.")

            with res_col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediksi_final,
                    number = {'prefix': "Rp ", 'valueformat': ",.0f"},
                    title = {'text': "Estimasi Harga Sewa (Per Bulan)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [200000, 3000000]},
                        'bar': {'color': "#00B4D8"},
                        'steps': [
                            {'range': [200000, 600000], 'color': "#E0F7FA"},
                            {'range': [600000, 1500000], 'color': "#B2EBF2"},
                            {'range': [1500000, 3000000], 'color': "#80DEEA"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prediksi_final}
                    }
                ))
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**🗺️ Area Wilayah ({lokasi_pilihan})**")
                df_map = pd.DataFrame([{"lat": koordinat_wilayah[lokasi_pilihan][0], "lon": koordinat_wilayah[lokasi_pilihan][1]}])
                st.map(df_map, zoom=13)


# -------------------------------------------------------------
# 5. HALAMAN 2: INSIGHT & FUN FACT
# -------------------------------------------------------------
elif menu == "Insight & Fun Fact":
    st.title("Insight Penelitian & Fun Fact")
    st.markdown("Penasaran apa yang sebenarnya membuat harga kost bisa sangat mahal atau murah? Berikut adalah temuan dari algoritma *Machine Learning*!")
    
    try:
        img = Image.open('grafik_feature_importance.png')
        st.image(img, caption="Hasil Analisis Feature Importance Random Forest", use_column_width=True)
    except FileNotFoundError:
        st.warning("Gambar grafik_feature_importance.png tidak ditemukan di folder. Pastikan file gambar sudah dirender ulang setelah perbaikan model.")

    st.markdown("### 3 Fun Fact Menarik Seputar Kost di Sukabumi")
    
    st.info("**1. AC Adalah Raja Harga**\n\n"
            "Ternyata, ketersediaan AC menjadi garis batas mutlak antara kost standar dan kost premium. Algoritma menemukan bahwa pemilik kost menaikkan harga paling drastis hanya dengan menambahkan fasilitas AC, jauh mengalahkan pengaruh fasilitas lain.")
    
    st.success("**2. Pengaruh Lokasi Geografis**\n\n"
               "Dari seluruh wilayah yang diteliti, algoritma memetakan bahwa properti di area tertentu memiliki segmentasi pasar dan pola harga yang paling khas dibandingkan pusat kota. Letak geografis terbukti menjadi patokan penting kedua bagi mesin.")
    
    st.warning("**3. Aturan Gender Tidak Memengaruhi Harga (Pengaruh < 1%)**\n\n"
               "Ada mitos bahwa kost khusus putri lebih mahal karena fasilitas keamanan ekstra. Namun, data membuktikan hal itu salah! Secara empiris, algoritma melihat bahwa tipe kost (Putra/Putri/Campur) hampir tidak memberikan pengaruh sama sekali terhadap fluktuasi harga.")
