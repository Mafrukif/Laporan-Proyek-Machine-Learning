# Laporan Proyek Machine Learning - Mafrukhif Dzulfahmil Nur
## Domain Proyek

**Kategori**

**Rubrik / Kriteria Penilaian (Wajib)**

---

## Domain Proyek

### Latar Belakang

Perkembangan sektor properti merupakan salah satu indikator utama pertumbuhan ekonomi. Harga rumah yang terus mengalami fluktuasi membuat banyak pihak, baik individu maupun lembaga keuangan, memerlukan sistem prediksi harga rumah yang akurat. Dengan bantuan teknologi machine learning, prediksi harga rumah dapat dilakukan dengan lebih cepat dan tepat, memanfaatkan data historis serta karakteristik rumah itu sendiri.

Misalnya, calon pembeli rumah akan lebih terbantu dengan adanya estimasi harga yang realistis, dan perusahaan real estate dapat mengatur strategi pemasaran berdasarkan proyeksi harga. Oleh karena itu, masalah prediksi harga rumah perlu ditangani secara sistematis.

### Referensi

Dataset yang digunakan bersumber dari Kaggle, yang merupakan platform berbagi dataset yang kredibel:

* [Kaggle - Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

---

## Business Understanding

### Problem Statement

Bagaimana membangun model machine learning yang mampu memprediksi harga rumah secara akurat berdasarkan fitur-fitur seperti luas bangunan, jumlah kamar, adanya fasilitas tambahan, dan lain-lain?

### Goals

Membuat model prediksi harga rumah yang akurat dan dapat digunakan untuk membantu pengambilan keputusan.

### Solution Statement

Untuk mencapai tujuan tersebut, dilakukan pendekatan sebagai berikut:

1. Menerapkan tiga algoritma regresi:

   * Random Forest Regressor
   * XGBoost Regressor
   * Gradient Boosting Regressor

2. Metrik evaluasi yang digunakan:

   * Mean Squared Error (MSE)
   * R-squared (R2)

3. Dilakukan **hyperparameter tuning** pada model Random Forest menggunakan GridSearchCV untuk meningkatkan performa.

---

## Data Understanding

### Deskripsi Dataset

* Jumlah data: 545 baris
* Target: `price`
* Beberapa fitur penting: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`, `mainroad`, `airconditioning`, dll.
* Beberapa fitur kategorikal telah diubah menggunakan one-hot encoding.

### Link Dataset

* [Housing Prices Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

### Visualisasi dan Statistik

* Dilakukan distribusi awal harga rumah dan korelasi antar fitur.
* PCA diterapkan pada fitur numerik untuk mereduksi dimensi menjadi satu fitur representatif (`housing_pca`).

---

## Data Preparation

### Teknik yang Digunakan

1. One-hot encoding pada fitur kategorikal
2. PCA pada fitur numerik untuk ekstraksi fitur utama
3. StandardScaler digunakan untuk melakukan standarisasi fitur numerik (`housing_pca`)

### Alasan

* One-hot encoding diperlukan untuk mengubah variabel kategorikal menjadi bentuk numerik.
* PCA membantu menyederhanakan fitur numerik tanpa mengorbankan informasi penting.
* Standardisasi diperlukan agar model regresi berbasis jarak bekerja lebih optimal.

---

## Modeling

### Model yang Digunakan

1. **Random Forest Regressor**

   * MSE: 1.78 triliun
   * R2: 0.44
2. **XGBoost Regressor**

   * MSE: 1.74 triliun
   * R2: 0.45 (terbaik)
3. **Gradient Boosting Regressor**

   * MSE: 1.79 triliun
   * R2: 0.44

### Hyperparameter Tuning

GridSearchCV dilakukan pada Random Forest untuk mendapatkan parameter terbaik:

* Best Params: `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt'}`
* R2 setelah tuning (CV): 0.50

### Kelebihan & Kekurangan

* Random Forest: cepat dan stabil, namun bisa overfitting
* XGBoost: akurasi tinggi, waktu training lebih lama
* Gradient Boosting: performa bagus, tapi sensitif terhadap outlier

Model terbaik dipilih berdasarkan nilai MSE dan R2: **XGBoost Regressor**.

---

## Evaluation

### Metrik Evaluasi

1. **Mean Squared Error (MSE)**: mengukur rata-rata kuadrat error antara prediksi dan nilai asli.

   * Formula: $MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$
2. **R-squared (R2)**: proporsi variansi yang dapat dijelaskan oleh model.

   * Formula: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

### Hasil Evaluasi

* XGBoost memiliki MSE terendah (1.74T) dan R2 tertinggi (0.45)
* Visualisasi MSE menunjukkan bahwa XGBoost lebih unggul dibanding dua model lain.

---

## Struktur Laporan

Laporan ini disusun secara sistematis sesuai dengan template proyek:

* Terdiri dari bagian: domain, pemahaman masalah, data, modeling, evaluasi
* Penjelasan disertai reasoning, hasil numerik, dan visualisasi (MSE bar chart)
* Snippet kode diberikan di notebook terpisah dan dijelaskan sesuai relevansinya

---

