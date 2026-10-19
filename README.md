# Proyek Machine Learning: Prediksi & Klasifikasi Pokémon GO

## Data Diri Mahasiswa
- **Nama**: Raihan Shandi Adrida Meilano
- **NIM**: 241011401315
- **Kelas**: 05TPLP023

---

## Sumber Dataset & Referensi
Workflow dan dataset dalam proyek ini bersumber dari Kaggle:
- **Dataset**: [Pokémon GO Dataset (pokemon.csv)](https://www.kaggle.com/datasets/shreyasur965/pokemon-go/data?select=pokemon.csv)
- **Author / Kontributor**: Shreyas Ur
- **File Digunakan**: `pokemon.csv`

---

## Alur Kerja (Workflow) Proyek

Proyek ini mencakup siklus pemrosesan machine learning end-to-end yang diimplementasikan pada notebook [`pokemon.ipynb`](file:///c:/Users/SANRIO/Documents/Machine%20Learning/KAMPUS/datasetgua/pokemon.ipynb):

1. **Data Ingestion & Understanding**
   - Membaca file `pokemon.csv` menggunakan pandas.
   - Memeriksa struktur dataset, tipe data, jumlah entri, serta sampel data awal.

2. **Data Preprocessing & Data Cleaning**
   - **Pemilihan Fitur & Penghapusan Kolom**:
     - Memisahkan target `max_cp` ke dalam variabel `y`.
     - Menghapus fitur identitas (`pokemon_id`, `pokemon_name`) serta kolom-kolom non-prediktif (seperti probabilitas capture/flee, moveset, dan metode penemuan Pokémon).
     - **Catatan Penghapusan `pokemon_name`**: Fitur `pokemon_name` sengaja dihapus dari pemodelan machine learning karena merupakan label identitas nominal berkardinalitas tinggi yang tidak memiliki hubungan fungsional langsung terhadap Combat Power. Menyertakan nama (misalnya melalui Label Encoding) berisiko menyisipkan bias urutan (*pseudo-ordinal bias*) dan noise yang merusak kemampuan generalisasi model.
   - **Encoding Variabel Kategorikal**:
     - *MultiLabelBinarizer* untuk fitur `type` (karena Pokémon dapat memiliki tipe ganda / dual-typing).
     - *One-Hot Encoding* untuk tingkat kelangkaan (`rarity`) dengan menghapus kategori acuan (*dummy variable trap*).
   - **Pembersihan Data Missing & Duplikat**:
     - Pengisian missing values pada kolom `candy_required` dengan nilai `0` (untuk Pokémon yang tidak memerlukan permen evolusi / bentuk final).
     - Verifikasi tidak adanya baris data duplikat.
   - **Penyaringan Anomali Rusak**:
     - Memfilter dan menghapus data dengan statistik dasar (*base attack*, *base defense*, *base stamina*) yang bernilai `0` (indikasi data rusak/placeholder).

3. **Alasan Mempertahankan Outlier (Handling Outliers)**
   - Dalam analisis data dan machine learning, pencilan (*outliers*) sering kali dijumpai pada statistik dasar dan nilai `max_cp` (misalnya pada Pokémon Legendary, Mythic, atau Pokémon dengan spesialisasi statistik ekstrem). Proyek ini **secara sengaja mempertahankan nilai-nilai outlier tersebut** dengan landasan metodologis berikut:
     1. **Karakteristik Alami Domain (Ground Truth / Legitimate Variation)**:
        - Outlier pada dataset ini bukan disebabkan oleh kesalahan input manusia (*human error*), kegagalan sensor, atau anomali pencatatan data.
        - Nilai ekstrem tersebut merupakan karakteristik resmi dan sah (*canon*) dari Pokémon di dalam game Pokémon GO. Contohnya, Pokémon legendaris seperti Mewtwo, Rayquaza, Slaking, Kyogre, dan Groudon memang dirancang memiliki statistik serangan dan `max_cp` yang sangat tinggi (> 4000 CP). Sebaliknya, Shuckle memiliki statistik pertahanan yang sangat tinggi (~396) dengan attack sangat rendah, serta Blissey memiliki stamina/HP masif (~496).
     2. **Integritas Formula Deterministik Game**:
        - Nilai Combat Power (`max_cp`) dihitung oleh algoritma game berdasarkan formula deterministik dari kombinasi statistik:
          $$\text{CP} \approx \frac{\text{Attack} \times \sqrt{\text{Defense}} \times \sqrt{\text{Stamina}}}{10}$$
        - Hubungan matematis ini berlaku secara universal untuk seluruh Pokémon, baik kelas rendah maupun kelas tertinggi. Menghapus outlier akan memotong distribusi data pada rentang batas atas (*upper bound*), yang menyebabkan model regresi mengalami bias meremehkan (*underestimation bias*) saat memprediksi Pokémon kuat.
     3. **Relevansi Penggunaan Praktis Model**:
        - Pokémon berkategori outlier (terutama tier Legendary, Mythic, dan Ultra Beast) justru merupakan Pokémon yang paling sering dianalisis, dicari, dan digunakan oleh pemain dalam pertarungan kompetitif (*Raid Battle* dan *Master League PvP*). Menghapus mereka dari data latih akan membuat model kehilangan relevansi praktis terhadap skenario nyata.
     4. **Pembedaan Tegas antara Outlier Valid vs Anomali Rusak**:
        - Data cleaning tetap dilakukan secara selektif dengan hanya menghapus anomali rusak yang tidak masuk akal secara logika game (baris dengan attack/defense/stamina bernilai $\le 0$). Data ekstrem yang sah tetap dijaga keutuhannya.
     5. **Solusi Rekayasa Fitur melalui Standardisasi (`StandardScaler`)**:
        - Dampak perbedaan skala angka ekstrem ditangani secara tepat menggunakan standardisasi fitur (`StandardScaler`), bukan dengan memangkas data. Hal ini menjaga distribusi fitur numerik berada pada skala seimbang tanpa kehilangan satu pun entri data berharga.

4. **Exploratory Data Analysis (EDA)**
   - Menghitung ringkasan statistik deskriptif pada seluruh fitur numerik dan kategorikal.
   - Visualisasi korelasi antar fitur dan target menggunakan Heatmap Korelasi Pearson.

5. **Data Splitting & Feature Scaling**
   - Membagi data menjadi data latih (*training set* 80%) dan data uji (*testing set* 20%) menggunakan `train_test_split` dengan `random_state=42`.
   - Melakukan standardisasi fitur numerik menggunakan `StandardScaler` yang dipelajari pada data latih (*fit_transform*) dan diterapkan pada data uji (*transform*).

6. **Pemodelan & Evaluasi Machine Learning**
   - **Linear Regression (Kasus Regresi)**:
     - Melatih model untuk memprediksi nilai kontinu Combat Power (`max_cp`).
     - Hasil evaluasi performa:
       - $R^2$ Latih: **0.9764**
       - $R^2$ Uji: **0.9825**
       - Mean Absolute Error (MAE): **99.78 CP**
       - Root Mean Squared Error (RMSE): **135.90 CP**
     - Hasil menunjukkan model memiliki kemampuan generalisasi yang sangat kuat tanpa indikasi overfitting.
   - **Logistic Regression (Kasus Klasifikasi)**:
     - Melatih model untuk mengklasifikasikan kategori kekuatan Pokémon menjadi *CP Rendah* (di bawah atau sama dengan median) vs *CP Tinggi* (di atas median).
     - Hasil evaluasi performa:
       - Akurasi Data Uji: **99.00%**
       - Macro F1-Score: **0.99**
     - **Analisis Confusion Matrix**:
       - True Negative (TN): 97 sampel CP Rendah diprediksi benar.
       - True Positive (TP): 102 sampel CP Tinggi diprediksi benar.
       - False Positive (FP): 1 sampel.
       - False Negative (FN): 1 sampel.
       - Total prediksi benar mencapai 199 dari 201 data uji (hanya 2 kesalahan prediksi).

7. **Model Serialization (Ekspor Joblib)**
   - Menyimpan seluruh artefak model dan transformer ke format `.joblib` untuk kebutuhan inferensi dan deployment:
     - [`linear_regression_model.joblib`](file:///c:/Users/SANRIO/Documents/Machine%20Learning/KAMPUS/datasetgua/linear_regression_model.joblib)
     - [`logistic_regression_model.joblib`](file:///c:/Users/SANRIO/Documents/Machine%20Learning/KAMPUS/datasetgua/logistic_regression_model.joblib)
     - [`scaler.joblib`](file:///c:/Users/SANRIO/Documents/Machine%20Learning/KAMPUS/datasetgua/scaler.joblib)
   - Uji verifikasi inferensi dengan memuat kembali model menggunakan `joblib.load` pada sampel data uji telah berhasil dilakukan.

---

## Deskripsi Kolom Dataset (`pokemon.csv`)

| Nama Kolom | Deskripsi |
| :--- | :--- |
| `pokemon_id` | Identifikator unik untuk setiap Pokémon |
| `pokemon_name` | Nama Pokémon |
| `base_attack` | Nilai dasar statistik serangan (*base attack*) Pokémon |
| `base_defense` | Nilai dasar statistik pertahanan (*base defense*) Pokémon |
| `base_stamina` | Nilai dasar stamina / HP (*base stamina*) Pokémon |
| `type` | Tipe elemental dari Pokémon (bisa multi-tipe) |
| `rarity` | Klasifikasi kelangkaan Pokémon (*Standard, Legendary, Mythic*, dll.) |
| `charged_moves` | Daftar serangan bermuatan (*charged moves*) yang tersedia |
| `fast_moves` | Daftar serangan cepat (*fast moves*) yang tersedia |
| `candy_required` | Jumlah permen (*candy*) yang dibutuhkan untuk evolusi |
| `distance` | Jarak jalan (km) yang dibutuhkan untuk mendapatkan buddy candy |
| `max_cp` | Maksimum Combat Power (CP) yang dapat dicapai Pokémon (**Target Regresi**) |
| `attack_probability` | Probabilitas Pokémon melakukan serangan dalam pertarungan |
| `base_capture_rate` | Probabilitas dasar menangkap Pokémon |
| `base_flee_rate` | Probabilitas Pokémon melarikan diri saat encounter |
| `dodge_probability` | Probabilitas Pokémon menghindar dalam pertarungan |
| `max_pokemon_action_frequency` | Frekuensi aksi maksimum Pokémon dalam pertarungan |
| `min_pokemon_action_frequency` | Frekuensi aksi minimum Pokémon dalam pertarungan |
| `found_egg` | Indikator boolean apakah Pokémon dapat diperoleh dari telur |
| `found_evolution` | Indikator boolean apakah Pokémon dapat diperoleh melalui evolusi |
| `found_wild` | Indikator boolean apakah Pokémon dapat ditemukan di alam liar |
| `found_research` | Indikator boolean apakah Pokémon dapat diperoleh dari tugas riset |
| `found_raid` | Indikator boolean apakah Pokémon dapat ditemui di pertempuran raid |
| `found_photobomb` | Indikator boolean apakah Pokémon dapat muncul melalui photobomb |

---

## Struktur Berkas Repositori

```text
datasetgua/
├── pokemon.csv                         # Dataset mentah dari Kaggle
├── pokemon.ipynb                       # Notebook workflow data analysis & modeling
├── linear_regression_model.joblib      # Model Linear Regression tersimpan
├── logistic_regression_model.joblib    # Model Logistic Regression tersimpan
├── scaler.joblib                       # Objek StandardScaler tersimpan
├── pokemon column descriptors.txt      # Catatan deskripsi fitur asli
└── README.md                           # Dokumentasi proyek
```