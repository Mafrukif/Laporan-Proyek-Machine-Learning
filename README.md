# Laporan Proyek Machine Learning - Mafrukhif Dzulfahmil Nur

---

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

* Jumlah data: 545 baris dan 13 kolom
* Target: `price`
* Sumber: [Housing Prices Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

### Fitur dalam Dataset:

1. `price`: Harga rumah (target variabel)
2. `area`: Luas bangunan
3. `bedrooms`: Jumlah kamar tidur
4. `bathrooms`: Jumlah kamar mandi
5. `stories`: Jumlah lantai
6. `mainroad`: Apakah rumah berada di jalan utama (yes/no)
7. `guestroom`: Ada ruang tamu terpisah (yes/no)
8. `basement`: Ada basement (yes/no)
9. `hotwaterheating`: Ada pemanas air (yes/no)
10. `airconditioning`: Ada AC (yes/no)
11. `parking`: Jumlah tempat parkir
12. `prefarea`: Apakah berada di area preferensi (yes/no)
13. `furnishingstatus`: Status perabotan (furnished, semi-furnished, unfurnished)

### Kondisi Data Awal:

* Tidak terdapat missing values pada semua kolom.
* Terdapat outlier pada beberapa kolom numerik seperti `price`, `area`, dan `parking`, yang ditangani dengan metode IQR capping.
* Tidak ditemukan duplikat pada data.

### Visualisasi dan Statistik

* Distribusi awal harga rumah menunjukkan skew ke kanan.
* Korelasi fitur divisualisasikan menggunakan heatmap.
* PCA diterapkan pada fitur numerik untuk mereduksi dimensi menjadi fitur `housing_pca`.

---

## Data Preparation

### Teknik yang Digunakan

1. **Penanganan Outlier**:

   * Capping dengan IQR method pada kolom `price`, `area`, dan `parking`

2. **One-hot Encoding**:

   * Pada fitur kategorikal seperti `mainroad`, `guestroom`, `basement`, `furnishingstatus`, dan lain-lain

3. **Penghapusan Kolom**:

   * Kolom `furnishingstatus` dihapus karena dianggap redundan setelah encoding

4. **PCA**:

   * Diterapkan pada fitur numerik untuk menghasilkan fitur baru `housing_pca`

5. **Standarisasi**:

   * Menggunakan `StandardScaler` untuk menskalakan `housing_pca`

6. **Pembagian Data**:

   * Data dibagi menjadi data latih dan data uji menggunakan `train_test_split` dengan rasio 80:20

### Alasan

* Outlier dapat mempengaruhi performa model, sehingga dilakukan capping.
* Encoding dibutuhkan untuk mengubah fitur kategorikal ke bentuk numerik.
* PCA membantu menyederhanakan fitur numerik.
* Standarisasi penting untuk model berbasis regresi.
* Pembagian data diperlukan untuk evaluasi model secara obyektif.

---

## Modeling

### Model yang Digunakan dan Parameternya:

1. **Random Forest Regressor**

   * `random_state=123`
   * Kelebihan: cepat, stabil, menangani data non-linear dengan baik
   * Kekurangan: bisa overfitting
   * MSE: 1.78 triliun
   * R2: 0.44

2. **XGBoost Regressor**

   * `objective='reg:squarederror'`, `random_state=123`
   * Kelebihan: akurasi tinggi, menangani missing value
   * Kekurangan: waktu training relatif lebih lama
   * MSE: 1.74 triliun (terbaik)
   * R2: 0.45 (terbaik)

3. **Gradient Boosting Regressor**

   * `random_state=123`
   * Kelebihan: performa bagus pada data kompleks
   * Kekurangan: sensitif terhadap outlier
   * MSE: 1.79 triliun
   * R2: 0.44

### Hyperparameter Tuning

Dilakukan pada Random Forest Regressor menggunakan GridSearchCV:

* Best Params: `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt'}`
* R2 setelah tuning (CV): 0.50

### Cara Kerja Algoritma:

* **Random Forest**: Menggabungkan banyak pohon keputusan untuk membuat prediksi rata-rata.
* **XGBoost**: Membuat model secara bertahap dengan membetulkan error dari model sebelumnya menggunakan teknik boosting.
* **Gradient Boosting**: Serupa dengan XGBoost, tetapi lebih sederhana dan cenderung lebih lambat.

Model terbaik dipilih berdasarkan nilai MSE dan R2: **XGBoost Regressor**.

---

## Evaluation

### Metrik Evaluasi

1. **Mean Squared Error (MSE)**: mengukur rata-rata kuadrat error antara prediksi dan nilai asli.

   * Formula: $MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$

2. **R-squared (R2)**: proporsi variansi yang dapat dijelaskan oleh model.

   * Formula: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

### Hasil Evaluasi Model

| Model             | MSE          | R2   |
| ----------------- | ------------ | ---- |
| Random Forest     | 1.78 Triliun | 0.44 |
| XGBoost           | 1.74 Triliun | 0.45 |
| Gradient Boosting | 1.79 Triliun | 0.44 |

Visualisasi MSE menunjukkan bahwa XGBoost lebih unggul dibanding dua model lain.

---

## Struktur Laporan

Laporan ini disusun secara sistematis sesuai dengan template proyek:

* Terdiri dari bagian: domain, pemahaman masalah, data, modeling, evaluasi
* Penjelasan disertai reasoning, hasil numerik, dan visualisasi (MSE bar chart)
* Snippet kode diberikan di notebook terpisah dan dijelaskan sesuai relevansinya

---

