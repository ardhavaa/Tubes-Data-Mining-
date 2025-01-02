import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Muat dataset
df = pd.read_csv('data_hasil_clustering_dan_label.csv')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kategori Harga Rumah Bandung",
    page_icon="🏠",
    layout="wide"
)

# Inisialisasi state
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    # Menampilkan Halaman Selamat Datang dengan CSS 
    st.markdown(
        """
        <style>
            .welcome-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                height: 100vh;
                background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
                color: #1b5e20;
                font-family: 'Arial', sans-serif;
            }
            .welcome-logo img {
                border-radius: 50%;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
            .welcome-title {
                font-size: 2.8rem;
                font-weight: bold;
                margin-bottom: 15px;
                text-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
            }
            .welcome-text {
                font-size: 1.1rem;
                line-height: 1.6;
                max-width: 600px;
                margin-bottom: 30px;
                padding: 0 20px;
            }
            .start-button-container {
                margin-top: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .start-button {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 20px 40px;
                font-size: 2rem;
                font-weight: bold;
                border-radius: 30px;
                cursor: pointer;
                transition: background-color 0.3s ease-in-out, transform 0.3s ease-in-out;
            }
            .start-button:hover {
                background-color: #388e3c;
                transform: scale(1.1);
            }
        </style>
        <div class="welcome-container">
            <div class="welcome-logo">
                <img src="" width="150" alt="">
            </div>
            <div class="welcome-title">
                Selamat Datang di Aplikasi Prediksi Kategori Harga Rumah 🏠
            </div>
            <div class="welcome-text">
                Aplikasi ini dirancang untuk membantu Anda memprediksi kategori harga rumah di Bandung.
                Masukkan data seperti jumlah kamar, luas tanah, dan lokasi untuk mengetahui prediksi harga.
                Jelajahi fitur menarik kami untuk analisis data rumah.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Membuat dua kolom dengan rasio yang sama
    col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])
    
    # Kolom tengah untuk menampilkan tombol Mulai Aplikasi
    with col2:
    # Menambahkan CSS untuk memperbesar ukuran tombol
        st.markdown(
        """
        <style>
            .stButton>button {
                width: 300px;
                height: 60px;
                font-size: 20px;
                background-color: #4CAF50;  
                color: white;
                border-radius: 10px;
            }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Membuat tombol "Mulai Aplikasi"
    if st.button("Mulai Aplikasi", help="Klik untuk memulai aplikasi", key="start_button"):
        st.session_state.page = "main"


elif st.session_state.page == "main":
    # Menampilkan logo dan judul
    st.image("logo.jpg",use_container_width=True) 
    st.title("🏠 Prediksi Kategori Harga Rumah di Bandung")

    # Sidebar
    with st.sidebar:
        st.header("🔧 **Pengaturan dan Navigasi**")
        feature_option = st.radio(
            "Pilih fitur untuk eksplorasi data:", 
            ["Prediksi Harga Rumah", "Data Harga Rumah", "Visualisasi Data", "Cari Properti Sesuai Kriteria"], 
            label_visibility="visible"
        )
        st.markdown("---")
        st.write("**Petunjuk**: Gunakan pilihan di atas untuk menjelajahi fitur aplikasi.")
        if st.button("Kembali ke Halaman Awal"):
            st.session_state.page = "welcome"

    # Encode kolom 'City/Regency' ke dalam nilai numerik 1, 2, 3
    df['City/Regency'] = df['City/Regency'].replace({0: 1, 1: 2, 2: 3})

    # Standarisasi data
    scaler = StandardScaler()
    features = ['Price', 'Bedroom', 'Bathroom', 'Land', 'Building', 'City/Regency']
    X_scaled = scaler.fit_transform(df[features])
    y = df['Price_Category']  # Target

    # Pisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Inisialisasi dan latih model Logistic Regression
    logreg = LogisticRegression(max_iter=200, multi_class='ovr', solver='liblinear')
    logreg.fit(X_train, y_train)

    # Halaman fitur
    if feature_option == "Prediksi Harga Rumah":
        st.subheader("🛠️ Masukkan Fitur untuk Prediksi Kategori Harga")
        col1, col2 = st.columns(2)
        with col1:
            price = st.number_input("Harga Rumah (Price)", min_value=0)
            bedroom = st.number_input("Jumlah Kamar Tidur (Bedroom)", min_value=1)
            bathroom = st.number_input("Jumlah Kamar Mandi (Bathroom)", min_value=1)
        with col2:
            land = st.number_input("Luas Tanah (Land, dalam m²)", min_value=0)
            building = st.number_input("Luas Bangunan (Building, dalam m²)", min_value=0)
            city = st.selectbox("Kota/Regensi (City/Regency)", options=[1, 2, 3])

        if st.button("Prediksi"):
            input_data = np.array([[price, bedroom, bathroom, land, building, city]])
            input_data_scaled = scaler.transform(input_data)
            prediction = logreg.predict(input_data_scaled)
            proba = logreg.predict_proba(input_data_scaled)

            category = prediction[0]
            probability = np.max(proba)

            st.write(f"**Kategori Harga Rumah Prediksi**: {category}")
            st.write(f"**Probabilitas**: {probability:.2f}")

            st.subheader("Visualisasi Hasil Prediksi")
            fig, ax = plt.subplots()
            sns.barplot(x=['Prediksi'], y=[probability], ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probabilitas')
            ax.set_title(f'Probabilitas Kategori: {category}')
            st.pyplot(fig)

    elif feature_option == "Data Harga Rumah":
        st.subheader("📊 Data Harga Rumah")
        st.write("Berikut adalah data harga rumah yang telah dikategorikan.")
        st.dataframe(df[['Price', 'Bedroom', 'Bathroom', 'Land', 'Building', 'City/Regency']])

    elif feature_option == "Visualisasi Data":
        st.subheader("📈 Visualisasi Data")
        visual_option = st.radio("Pilih Visualisasi", ["Distribusi Harga", "Jumlah Rumah per Kategori Harga", "Penyebaran Data"])
        if visual_option == "Distribusi Harga":
            st.subheader("Distribusi Harga Rumah")
            plt.figure(figsize=(10, 5))
            sns.histplot(df['Price'], bins=30, kde=True)
            plt.title('Distribusi Harga Rumah')
            plt.xlabel('Harga')
            plt.ylabel('Frekuensi')
            st.pyplot(plt)
        elif visual_option == "Jumlah Rumah per Kategori Harga":
            st.subheader("Jumlah Rumah per Kategori Harga")
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='Price_Category')
            plt.title('Jumlah Rumah per Kategori Harga')
            plt.xlabel('Kategori Harga')
            plt.ylabel('Jumlah Rumah')
            st.pyplot(plt)
        elif visual_option == "Penyebaran Data":
            st.subheader("Penyebaran Data")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df, x='Land', y='Price', hue='City/Regency', style='Price_Category', ax=ax)
            plt.title('Penyebaran Harga Rumah Berdasarkan Luas Tanah dan Kategori Harga')
            plt.xlabel('Luas Tanah (m²)')
            plt.ylabel('Harga Rumah')
            st.pyplot(fig)

    elif feature_option == "Cari Properti Sesuai Kriteria":
        st.subheader("🔎 Cari Properti Berdasarkan Kriteria")
        col1, col2 = st.columns(2)
        with col1:
            max_price = st.number_input("Harga Maksimum (Max Price)", min_value=0)
            min_bedroom = st.number_input("Jumlah Kamar Tidur Minimum (Min Bedrooms)", min_value=1)
        with col2:
            min_bathroom = st.number_input("Jumlah Kamar Mandi Minimum (Min Bathrooms)", min_value=1)
            min_land = st.number_input("Luas Tanah Minimum (Min Land, dalam m²)", min_value=0)
            min_building = st.number_input("Luas Bangunan Minimum (Min Building, dalam m²)", min_value=0)
            city = st.selectbox("Kota/Regensi (City/Regency)", options=[1, 2, 3])

        if st.button("Cari Properti"):
            filtered_data = df[
                (df['Price'] <= max_price) &
                (df['Bedroom'] >= min_bedroom) &
                (df['Bathroom'] >= min_bathroom) &
                (df['Land'] >= min_land) &
                (df['Building'] >= min_building) &
                (df['City/Regency'] == city)
            ]
            if not filtered_data.empty:
                st.write("Properti yang cocok dengan kriteria:")
                st.dataframe(filtered_data)
            else:
                st.write("Tidak ada properti yang cocok dengan kriteria.")
