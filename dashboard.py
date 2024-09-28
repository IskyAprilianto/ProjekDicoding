import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('products_dataset.csv')
    return data

data = load_data()

# Sidebar - User Input
st.sidebar.header("User Input Parameters")
analysis_type = st.sidebar.selectbox("Pilih Analisis", ("EDA", "Clustering"))

# Title
st.title("Analisis Dataset Produk")
st.write("Dataset berisi informasi tentang berbagai produk, termasuk kategori, panjang nama, deskripsi, dan dimensi produk.")

# Display Data
if st.checkbox("Tampilkan Dataset"):
    st.write(data.head())

# Exploratory Data Analysis (EDA)
if analysis_type == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    # Distribusi berat produk berdasarkan kategori
    st.subheader("Distribusi Berat Produk Berdasarkan Kategori")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='product_category_name', y='product_weight_g', data=data, ax=ax)
    plt.xticks(rotation=90)
    plt.title('Distribusi Berat Produk di Setiap Kategori')
    plt.xlabel('Kategori Produk')
    plt.ylabel('Berat Produk (gram)')
    st.pyplot(fig)  # Menampilkan grafik di Streamlit

    # Hubungan antara jumlah foto dan panjang deskripsi produk
    st.subheader("Hubungan Jumlah Foto Produk dan Panjang Deskripsi Produk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='product_photos_qty', y='product_description_lenght', data=data, ax=ax)
    plt.title('Hubungan Jumlah Foto Produk dan Panjang Deskripsi Produk')
    plt.xlabel('Jumlah Foto Produk')
    plt.ylabel('Panjang Deskripsi Produk')
    st.pyplot(fig)  # Menampilkan grafik di Streamlit

# Clustering Analysis
if analysis_type == "Clustering":
    st.header("Clustering Analysis")

    # Preprocessing untuk clustering
    st.subheader("Clustering Berdasarkan Dimensi dan Berat Produk")
    features = data[['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']]

    # Mengisi nilai yang hilang pada fitur dengan median
    features = features.fillna(features.median())

    # Standarisasi fitur
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Clustering menggunakan KMeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(scaled_features)
    data['cluster'] = kmeans.labels_

    # Visualisasi hasil clustering
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='product_length_cm', y='product_weight_g', hue='cluster', data=data, palette='viridis', ax=ax)
    plt.title('Clustering Berdasarkan Dimensi dan Berat Produk')
    plt.xlabel('Panjang Produk (cm)')
    plt.ylabel('Berat Produk (gram)')
    st.pyplot(fig)

    # Unduh dataset yang sudah dianalisis
    st.subheader("Unduh Dataset yang Sudah Dianalisis")
    csv = data.to_csv(index=False)
    st.download_button(
        label="Unduh CSV",
        data=csv,
        file_name='products_dataset_analyzed.csv',
        mime='text/csv'
    )

# Run Streamlit App
# Save this file as 'app.py' and run using 'streamlit run app.py'
