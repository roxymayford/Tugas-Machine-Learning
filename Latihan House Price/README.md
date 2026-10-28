# Proyek Machine Learning: Prediksi Harga Rumah (House Prices)
> **Pipeline End-to-End: Data Cleaning, Preprocessing, Exploratory Data Analysis (EDA), dan Pemodelan Regresi**

---

## Ringkasan Proyek
Proyek ini merupakan dokumentasi komprehensif dari proses belajar kami dalam membangun pipeline *machine learning* regresi dari tahap data mentah (*raw data*) hingga siap produksi (*production-ready*). Studi kasus yang diangkat adalah memprediksi harga jual properti/rumah (`SalePrice`) berdasarkan karakteristik fisik, fasilitas, dan lokasi bangunan menggunakan **Ames Housing Dataset**.

Dataset mentah memiliki berbagai tantangan nyata seperti *missing value* tinggi, *outlier* ekstrem, perbedaan skala fitur numerik, serta fitur kategorikal yang kompleks. Kami mendemonstrasikan bagaimana teknik preprocessing yang tepat secara langsung meningkatkan performa generalisasi model machine learning.

---

## Struktur File dan Folder

```text
├── pertemuan6_data_cleaning_transformation.ipynb   # Notebook utama (Data Cleaning, EDA, Modeling, Export)
├── app.py                                          # Frontend aplikasi web interaktif (Streamlit)
├── train_and_export.py                             # Skrip otomatisasi pelatihan & ekspor model
├── model.joblib                                    # Model Gradient Boosting Regressor (Joblib)
├── model.pkl                                       # Model Gradient Boosting Regressor (Pickle)
├── house_price_pipeline.joblib                     # Pipeline lengkap (Model + Scaler + Encoder + Meta)
├── house_price_pipeline.pkl                        # Pipeline lengkap format Pickle
├── requirements.txt                                # Daftar dependensi Python untuk deployment
├── train.csv                                       # Dataset latihan mentah (1.460 baris, 81 kolom)
├── test.csv                                        # Dataset pengujian (1.459 baris, 80 kolom)
├── sample_submission.csv                           # Format penyerahan prediksi kompetisi
└── README.md                                       # Dokumentasi lengkap proyek
```

---

## Tahapan Alur Kerja (Pipeline Workflow)

Pipeline pengolahan data dan pemodelan pada proyek ini dibagi menjadi 15 tahapan terstruktur:

### 1. Eksplorasi Awal & Audit Data
- **Dataset:** 1.460 baris dan 81 kolom (38 fitur numerik, 43 fitur kategorikal).
- **Audit Tipe Data:** Memastikan kesesuaian tipe data dan mengidentifikasi kolom integer yang otomatis terbaca sebagai `float64` akibat adanya nilai `NaN`.

### 2. Penanganan Missing Value
- **Missing > 75%:** Dihapus (*drop*) sebanyak 4 fitur (`Alley`, `PoolQC`, `Fence`, `MiscFeature`) karena tingkat kekosongannya melebihi 1.000 baris dan berisiko menimbulkan bias imputasi.
- **Missing < 75% (Numerik):** Diimputasi menggunakan **Median** per kolom karena lebih tahan terhadap nilai ekstrem dibandingkan Mean.
- **Missing < 75% (Kategorikal):** Diimputasi menggunakan **Modus** (nilai/kategori frekuensi tertinggi).

### 3. Deteksi & Pembersihan Outlier (Metode IQR)
- Menggunakan visualisasi **Boxplot** untuk mendeteksi sebaran titik di luar rentang kuartil.
- Menghitung **Interquartile Range**:
  $$\text{IQR} = Q3 - Q1$$
  - Batas Bawah: $Q1 - 1.5 \times \text{IQR}$
  - Batas Atas: $Q3 + 1.5 \times \text{IQR}$
- Baris data yang berada di luar rentang batas pada fitur numerik dieliminasi, menghasilkan dataset yang lebih homogen dan bersih (557 baris).

### 4. Standardisasi Skala Fitur (Z-Score Scaling)
- Rentang nilai antar fitur numerik sangat bervariasi (mulai dari biner 0–1 hingga puluhan ribu pada `LotArea`).
- Menggunakan `StandardScaler` dari Scikit-Learn:
  $$z = \frac{x - \mu}{\sigma}$$
  sehingga seluruh fitur numerik terpusat pada mean $\approx 0$ dan standar deviasi $\approx 1$.

### 5. Pemeriksaan Data Duplikat
- Menggunakan `df.duplicated().sum()`. Terverifikasi bahwa tidak ada baris duplikat yang identik (0 duplikat).

### 6. Rekayasa Fitur Kategorikal (Feature Encoding)
- Menguji dua pendekatan:
  1. **One-Hot Encoding (`pd.get_dummies`):** Menghasilkan lonjakan dimensi menjadi ratusan kolom (*curse of dimensionality*).
  2. **Label Encoding (`LabelEncoder`):** Mengonversi kategori menjadi integer urut tanpa menambah jumlah kolom (tetap 77 kolom).
- Dataframe hasil Label Encoding (`df_lencoder`) dipilih untuk tahapan analisis dan pemodelan lanjutan.

### 7. Exploratory Data Analysis (EDA) & Analisis Korelasi
- Memverifikasi ulang bahwa data 100% bebas dari missing value.
- Visualisasi sebaran histogram seluruh kolom.
- Menghitung **Korelasi Pearson** terhadap variabel target `SalePrice`. Fitur penentu utama dengan korelasi linier terkuat:
  1. `OverallQual` ($r = 0.79$) — Kualitas material dan pengerjaan rumah.
  2. `GrLivArea` ($r = 0.71$) — Luas area tempat tinggal di atas tanah.
  3. `GarageCars` ($r = 0.64$) & `GarageArea` ($r = 0.62$) — Fasilitas garasi kendaraan.
  4. `TotalBsmtSF` ($r = 0.61$) — Luas ruang bawah tanah (basement).
  5. `YearBuilt` ($r = 0.52$) — Tahun pembuatan konstruksi bangunan.

### 8. Pembagian Data (Train-Test Split)
- Data dibagi dengan rasio **80% Data Latih (445 sampel)** dan **20% Data Uji (112 sampel)** menggunakan `random_state=1`.
- Verifikasi kurva Kernel Density Estimation (KDE) mengonfirmasi bahwa distribusi target pada data latih dan data uji sangat identik (bebas dari sampling bias).

### 9. Pemodelan Machine Learning
Menguji 7 algoritma dengan paradigma berbeda:
1. **DummyRegressor (Mean Baseline):** Patokan tebakan naif rata-rata harga.
2. **Linear Regression (OLS):** Regresi parametrik linier klasik.
3. **Ridge Regression (L2):** Regularisasi kuadrat bobot untuk menekan multikolinearitas.
4. **Lasso Regression (L1):** Regularisasi nilai mutlak yang menyeleksi fitur otomatis.
5. **Random Forest Regressor:** Ensemble pohon keputusan berbasis Bagging.
6. **Gradient Boosting Regressor:** Ensemble sekuensial Boosting.
7. **Support Vector Regression (SVR - RBF Kernel):** Model regresi berbasis hyperplane dan margin.

### 9.1 Landasan Teori & Rasional Pemilihan Linear Regression

Meskipun model ensemble non-linier (*Gradient Boosting* dan *Random Forest*) memberikan performa prediktif tertinggi, **Linear Regression (Ordinary Least Squares - OLS)** dipilih sebagai salah satu algoritma inti dalam pipeline penelitian ini dengan pertimbangan teoretis dan metodologis berikut:

1. **Target Prediksi Numerik Kontinu (`SalePrice`):**  
   Variabel target yang diestimasi adalah nilai harga jual rumah (`SalePrice`) yang merupakan variabel kontinu berskala rasio. **Fatah & Muhtajin (2026, hlm. 344)** menyatakan bahwa algoritma regresi linier merupakan metode statistik fundamental dan efektif untuk *"memprediksi nilai numerik kontinu"* berdasarkan relasi multivariat antarfitur penjelas.
2. **Interpretabilitas Tinggi & Kemudahan Komputasi:**  
   **Rico-Juan & Taltavull de la Paz (2021, p. 6)** menguraikan bahwa model regresi linear klasik *"easy to calculate and interpret but assume in their models only linear relationships"*. Sifat transparansi (*white-box*) ini memungkinkan koefisien bobot parameter ($\beta$) dari tiap fitur diinterpretasikan secara langsung, sehingga analis dapat memahami kontribusi marjinal karakteristik fisik rumah (seperti `OverallQual` dan `GrLivArea`) terhadap harga pasar properti secara transparan dan logis.
3. **Performa sebagai Baseline Empiris yang Solid & Stabil:**  
   Dalam literatur penetapan harga properti hedonik (*hedonic housing prices*), model OLS konvensional terbukti mampu menjelaskan **80,8%** variasi variabel target (**Rico-Juan & Taltavull de la Paz, 2021, p. 7**) dengan capaian rata-rata $R^2$ sebesar **0,823** pada pengujian *10-fold cross-validation* (**Rico-Juan & Taltavull de la Paz, 2021, p. 13**). Sejalan dengan literatur internasional tersebut, studi empiris **Fatah & Muhtajin (2026, hlm. 343)** juga mencatatkan $R^2 \approx 0,91$ (0,91 pada abstrak / 0,916 pada teks isi pembahasan), MAE 81.250, dan RMSE 100.500 dalam memprediksi harga rumah. Capaian ini membuktikan adanya korelasi linier yang kuat antara atribut properti dan harga pasar, serta menegaskan posisi regresi linear sebagai **baseline empiris yang stabil dan interpretabel**, bukan sebagai model paling akurat dibanding ensemble.
4. **Sensitivitas terhadap Outlier & Justifikasi Preprocessing:**  
   **Fatah & Muhtajin (2026, hlm. 347–348)** mencatat bahwa regresi linier sangat sensitif terhadap kehadiran data pencilan (*outliers*). Fakta teoretis ini memberikan justifikasi ilmiah yang kuat atas keharusan dilakukannya tahap pembersihan *outlier* berbasis **Metode Interquartile Range (IQR)** serta **standardisasi fitur (Z-Score)** pada tahap preprocessing sebelumnya di proyek ini guna mencegah bias estimasi.

#### Tabel Ringkasan Lokasi Kutipan (In-Text Citation) & Status Verifikasi Dokumen Sumber

| No | Sitasi Sumber | Lokasi Kutipan | Kutipan / Pernyataan Kunci | Catatan Status Verifikasi Dokumen Sumber |
|:---:|:---|:---|:---|:---|
| 1 | **Fatah & Muhtajin (2026)** | Hlm. 344 | Regresi linier efektif untuk "memprediksi nilai numerik kontinu" | Terverifikasi langsung dari naskah publikasi *JAMASTIKA*, Vol. 5, No. 1, hlm. 344. |
| 2 | **Fatah & Muhtajin (2026)** | Hlm. 343 | Performa model: $R^2 \approx 0,91$ (0,91 di abstrak / 0,916 di isi), MAE 81.250, RMSE 100.500 | Terverifikasi. Terdapat variasi pembulatan minor antara bagian abstrak (0,91) dan teks pembahasan (0,916). |
| 3 | **Fatah & Muhtajin (2026)** | Hlm. 347–348 | Analisis kerentanan regresi linier terhadap keberadaan *outlier* | Terverifikasi. Menjadi rujukan justifikasi pembersihan outlier berbasis IQR dan standardisasi fitur. |
| 4 | **Rico-Juan & Taltavull de la Paz (2021)** | p. 6 | Regresi linier klasik *"easy to calculate and interpret but assume in their models only linear relationships"* | Terverifikasi langsung dari artikel jurnal *Expert Systems with Applications* (Elsevier, Scopus Q1), hlm. 6. |
| 5 | **Rico-Juan & Taltavull de la Paz (2021)** | p. 7 | Model OLS hedonik menjelaskan 80,8% ($R^2 = 0,808$) variasi variabel target | Terverifikasi pada bagian hasil OLS baseline, hlm. 7. |
| 6 | **Rico-Juan & Taltavull de la Paz (2021)** | p. 13 | Rata-rata $R^2$ mencapai 0,823 pada evaluasi *10-fold cross-validation* di pasar properti | Terverifikasi pada tabel komparasi performa cross-validation, hlm. 13. |

#### Daftar Pustaka (Format APA 7):
- Fatah, Z., & Muhtajin, M. (2026). Prediksi harga rumah menggunakan algoritma regresi linier. *JAMASTIKA, 5*(1), 343–350.
- Rico-Juan, J. R., & Taltavull de la Paz, P. (2021). Machine learning with explainability or spatial hedonics tools? An analysis of the asking prices in the housing market in Alicante, Spain. *Expert Systems with Applications, 171*, Article 114590. https://doi.org/10.1016/j.eswa.2021.114590

---

## Hasil Evaluasi Model

### 1. Komparasi Kinerja pada Data Uji (Skala Terstandardisasi & Satuan Asli $ USD)

| Model | CV $R^2$ (Train) | Test $R^2$ | Test RMSE (Z-Score) | Test RMSE ($ USD) | Test MAE ($ USD) | Test MAPE (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Dummy)** | -0.0125 | -0.0258 | 0.9075 | $50.108,41 | $39.906,02 | 26.68% |
| **Linear Regression** | 0.8124 | 0.8791 | 0.3117 | $17.208,45 | $12.356,80 | 8.01% |
| **Ridge Regression** | 0.8159 | 0.8846 | 0.3045 | $16.812,79 | $12.189,45 | 7.91% |
| **Lasso Regression** | 0.8142 | 0.8816 | 0.3084 | $17.027,33 | $12.062,11 | **7.78%** |
| **Random Forest** | 0.8351 | 0.8890 | 0.2987 | $16.491,12 | $12.221,05 | 7.95% |
| **Gradient Boosting** | **0.8412** | **0.8898** | **0.2974** | **$16.421,55** | **$12.055,14** | **7.79%** |
| **SVR (RBF Kernel)** | 0.8298 | 0.8654 | 0.3288 | $18.155,20 | $12.870,14 | 8.35% |

### 2. Hasil Hyperparameter Tuning (GridSearchCV pada Gradient Boosting)
Melalui pencarian grid pada hyperparameter:
- `learning_rate`: **0.05**
- `n_estimators`: **150**
- `max_depth`: **3**
- `subsample`: **0.8**

**Hasil Peningkatan Model Terbaik:**
- **Test $R^2$ Score:** Naik menjadi **0.8915** (mampu menjelaskan 89.15% varians harga rumah).
- **Test RMSE ($ USD):** Turun menjadi **$16.303,46**.
- **Test MAE ($ USD):** Turun menjadi **$12.042,87**.
- **Test MAPE:** Turun menjadi **7.76%** (kategori *high precision valuation*).

---

## Rujukan Ilmiah: Landasan Teori Pemodelan & Evaluasi Metrik

Dalam mengevaluasi performa model regresi serta justifikasi pemilihan algoritma pada proyek ini, analisis didasarkan pada landasan literatur ilmiah berikut:

> **Prinsip Utama Evaluasi Metrik:**  
> Semakin mendekati **1 nilai $R^2$**, serta semakin mendekati **0 nilai RMSE dan MAE**, menunjukkan kinerja model yang semakin baik (*Chicco et al., 2021; Yasin et al., 2025; Zarbakhsh et al., 2024*).

### 1. Publikasi Internasional (Scopus Q1) - Landasan Teori Evaluasi Metrik
**Sitasi:**  
Chicco, D., Warrens, M. J., & Jurman, G. (2021). The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation. *PeerJ Computer Science, 7*, e623. [https://doi.org/10.7717/peerj-cs.623](https://doi.org/10.7717/peerj-cs.623)
- **$R^2$ (Koefisien Determinasi):** Rentang nilai dari $-\infty$ (terburuk) hingga $+1$ (kecocokan sempurna). Nilai 1 berarti model mampu memprediksi target dengan presisi tanpa kesalahan.
- **RMSE & MAE:** Rentang nilai dari $0$ (sempurna tanpa error) hingga $+\infty$ (terburuk). Nilai yang makin besar menunjukkan model yang semakin meleset.
- **Catatan Penting:** Besaran mutlak MAE atau RMSE harus selalu dievaluasi terhadap skala data aslinya. Oleh karena itu, kami menyajikan metrik dalam skala terstandardisasi maupun dalam nilai mata uang riil ($ USD) dan persentase relatif (MAPE).

### 2. Publikasi Internasional (Scopus Q1) - Komparasi Kinerja Model Ensemble
**Sitasi:**  
Zarbakhsh, S., Shahsavar, A. R., & Soltani, M. (2024). Optimizing PGRs for in vitro shoot proliferation of pomegranate with bayesian-tuned ensemble stacking regression and NSGA-II: A comparative evaluation of machine learning models. *Plant Methods, 20*, Article 82. [https://doi.org/10.1186/s13007-024-01211-5](https://doi.org/10.1186/s13007-024-01211-5)
- **$R^2$ (Koefisien Determinasi):** Semakin dekat nilai $R^2$ ke 1, semakin baik kapasitas model dalam menerangkan variabilitas data.
- **RMSE & MAE:** Nilai metrik yang semakin rendah merefleksikan performa prediksi yang lebih akurat.
- **Kriteria Model Terbaik:** Model terbaik ditentukan oleh kombinasi nilai $R^2$ yang lebih tinggi serta nilai metrik error (RMSE, MAE, dan MAPE) yang lebih kecil pada data latih maupun data uji.

### 3. Publikasi Nasional (SINTA 2) - Evaluasi Pemodelan Prediksi
**Sitasi:**  
Yasin, F., Firmansyah, M. R., Aldo, D., & Amrustian, M. A. (2025). Multivariate forecasting of paddy production: A comparative study of machine learning models. *Jurnal Teknik Informatika (JUTIF), 6*(3), 1431–1442. [https://doi.org/10.52436/1.jutif.2025.6.3.4681](https://doi.org/10.52436/1.jutif.2025.6.3.4681)
- **MAE:** Semakin rendah nilai MAE, semakin baik akurasi estimasi model.
- **RMSE:** Memberikan penalti kuadrat yang lebih besar pada error ekstrem; nilai yang semakin rendah menunjukkan performa model yang semakin solid.
- **$R^2$:** Mengukur proporsi variabilitas target yang berhasil diterangkan oleh kombinasi fitur prediktor.

### 4. Publikasi Internasional (Scopus Q1) - Pemodelan Hedonik & Baseline Linear Regression
**Sitasi:**  
Rico-Juan, J. R., & Taltavull de la Paz, P. (2021). Machine learning with explainability or spatial hedonics tools? An analysis of the asking prices in the housing market in Alicante, Spain. *Expert Systems with Applications, 171*, Article 114590. [https://doi.org/10.1016/j.eswa.2021.114590](https://doi.org/10.1016/j.eswa.2021.114590)
- **Karakteristik OLS:** Model regresi linier mudah dihitung dan koefisiennya dapat diinterpretasikan secara langsung (*"easy to calculate and interpret but assume in their models only linear relationships"*).
- **Performa Baseline Empiris:** Terbukti mampu menjelaskan 80,8% ($R^2 = 0,808$) variasi variabel target harga rumah dan memperoleh rata-rata $R^2 = 0,823$ pada pengujian *10-fold cross-validation*.

### 5. Publikasi Nasional (JAMASTIKA) - Prediksi Harga Rumah Menggunakan Regresi Linier
**Sitasi:**  
Fatah, Z., & Muhtajin, M. (2026). Prediksi harga rumah menggunakan algoritma regresi linier. *JAMASTIKA, 5*(1), 343–350.
- **Prediksi Variabel Kontinu:** Regresi linier efektif untuk memprediksi nilai numerik kontinu seperti harga rumah.
- **Korelasi Linier Kuat:** Mencapai performa $R^2 \approx 0,91$, MAE 81.250, dan RMSE 100.500 dalam estimasi harga properti.
- **Sensitivitas Outlier:** Menekankan sensitivitas metode terhadap outlier, yang menjustifikasi keharusan filtering outlier (metode IQR) dan standardisasi fitur pada tahap preprocessing.

---


## Temuan Utama & Kesimpulan Proyek

1. **Efektivitas Data Preprocessing:**  
   Pembersihan missing value terarah, eliminasi outlier dengan IQR, dan standardisasi fitur berkontribusi langsung menaikkan $R^2$ model linier dan ensemble ke rentang $0.88 - 0.89$.
2. **Model Terbaik untuk Estimasi Nilai Rumah:**  
   Model **Gradient Boosting Regressor (Tuned)** memberikan performa tertinggi dengan rata-rata kesalahan prediksi hanya **\$12.042 (7.76%)**, memangkas error baseline naif sebesar **~70%**.
3. **Pilar Penentu Harga Properti (Feature Importance):**  
   Faktor fisik yang paling menentukan nilai jual rumah di Ames secara berurutan adalah:
   - Kualitas material bangunan & finishing (`OverallQual`)
   - Luas total area ruang huni di atas tanah (`GrLivArea`)
   - Fasilitas dan kapasitas kendaraan pada garasi (`GarageCars`, `GarageArea`)
   - Luas ruangan basement (`TotalBsmtSF`)
   - Tahun awal pembuatan rumah (`YearBuilt`)
4. **Rencana Pengembangan Selanjutnya:**
   - Eksperimen transformasi logaritmik $\log(1 + y)$ pada target `SalePrice`.
   - Feature engineering rasio ruangan gabungan (`TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`).
   - Menerapkan arsitektur *Stacking Regressor* (menggabungkan prediksi Ridge, Random Forest, dan Gradient Boosting).

---

## Deployment & Frontend Interaktif (Streamlit Web App)

Proyek ini telah dilengkapi dengan model yang diekspor ke format serialisasi standar industri (**Joblib** dan **Pickle**) serta aplikasi web frontend interaktif berbasis **Streamlit** untuk memfasilitasi pengujian inferensi harga rumah secara langsung.

### 1. Struktur Artifact Model Serialisasi
Model dan pipeline pendukung diekspor dalam 2 varian format:
- `model.joblib` & `model.pkl`: Model estimator `GradientBoostingRegressor` murni hasil tuning.
- `house_price_pipeline.joblib` & `house_price_pipeline.pkl`: Bundle komprehensif berisi model, objek `StandardScaler`, daftar `LabelEncoder`, konstanta pemulihan skala (`sp_mean`, `sp_scale`), nilai default median/mode, dan metrik performa.

### 2. Cara Menjalankan Aplikasi Web Frontend
Pastikan dependensi telah terpasang, lalu jalankan antarmuka Streamlit:

```bash
# 1. Instalasi dependensi
pip install -r requirements.txt

# 2. Jalankan aplikasi web
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di peramban (browser) pada alamat `http://localhost:8501`.

### 3. Fitur Utama Frontend Web:
- **Preset Skenario Instan:** Pilihan cepat profil rumah (Mewah, Keluarga Nyaman, Minimalis, atau Kustom).
- **Formulir Interaktif:** Kontrol slider dan input untuk 15+ parameter paling berpengaruh (`OverallQual`, `GrLivArea`, `GarageCars`, `YearBuilt`, `Neighborhood`, dll.).
- **Kalkulasi Nilai Ganda:** Estimasi harga jual ditampilkan dalam satuan **USD ($)** dan konversi otomatis ke **Rupiah (IDR)** berdasarkan kurs yang dapat disesuaikan.
- **Rentang Keyakinan (Error Margin):** Menampilkan batas estimasi atas dan bawah berdasarkan toleransi metrik MAE ($\pm \$11.870$).
- **Visualisasi Feature Importance:** Grafik kontribusi fitur terhadap pembentukan harga pasar.

### 4. Pelatihan Ulang & Ekspor Ulang Otomatis
Untuk melatih ulang model dan memperbarui seluruh file artifact serialisasi secara otomatis:

```bash
python train_and_export.py
```

### 5. Contoh Pemanggilan Model via Python:
```python
import joblib

# Memuat pipeline bundle
pipeline = joblib.load("house_price_pipeline.joblib")
model = pipeline["model"]
sp_scale = pipeline["sp_scale"]
sp_mean = pipeline["sp_mean"]

# Prediksi data uji
# y_pred_usd = model.predict(X_scaled) * sp_scale + sp_mean
```
