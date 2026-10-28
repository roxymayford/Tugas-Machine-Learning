# Algoritma Klasifikasi dalam Machine Learning

Catatan ringkas untuk mahasiswa yang sedang mempelajari bab klasifikasi: pengertian, lima algoritma yang umum dipakai dalam penelitian, serta panduan memilih algoritma sesuai karakter data.

## Pengertian klasifikasi

Klasifikasi adalah tugas dalam *supervised learning*, yaitu pembelajaran dari data yang sudah berlabel. Model dilatih memakai contoh yang jawabannya diketahui, lalu dipakai untuk menentukan kategori dari data yang belum pernah dilihat.

Contoh penerapan: menentukan apakah sebuah email termasuk spam, atau apakah suatu cuitan bernada positif atau negatif.

---

## Ringkasan perbandingan

| Algoritma | Pendekatan | Keunggulan utama | Kelemahan utama | Paling sesuai untuk |
|---|---|---|---|---|
| Decision Tree | Aturan bercabang | Mudah dijelaskan | Rawan overfitting | Keputusan yang harus bisa dipertanggungjawabkan |
| Random Forest | Gabungan banyak pohon | Stabil dan tahan outlier | Sulit diinterpretasi | Data tabular besar dan kompleks |
| KNN | Kemiripan dengan tetangga | Konsep sederhana | Sensitif noise, berat di data besar | Data kecil dengan fitur sedikit |
| SVM | Pemisah dengan margin maksimal | Akurat, kuat di dimensi tinggi | Tuning rumit, kurang skalabel | Teks dan gambar |
| Naive Bayes | Probabilitas Bayes | Sangat cepat | Asumsi independensi tidak realistis | Teks dan dokumen |

---

## Rincian algoritma

### Decision Tree

**Prinsip kerja.** Data dipecah berulang kali berdasarkan aturan kondisi sederhana hingga mencapai daun yang berisi keputusan akhir. Hasilnya menyerupai diagram alir.

**Kekuatan.**
- Hasil keputusan mudah dijelaskan kepada pihak non-teknis.
- Pohon keputusan dikenal interpretabel dan tangguh. Penelitian terbaru banyak membahas cara meningkatkan kecocokan terhadap data latih, kemampuan generalisasi, dan interpretabilitasnya (Costa & Pedreira, 2023).

**Keterbatasan.**
- Mudah overfitting: sangat baik pada data latih tetapi turun pada data baru. Pemangkasan (*pruning*) umum dipakai untuk mengatasinya.

**Cocok untuk.** Bidang medis dan keuangan, misalnya penilaian kelayakan kredit, ketika alasan di balik keputusan sama pentingnya dengan akurasi.

### Random Forest

**Prinsip kerja.** Membangun banyak decision tree sekaligus (*ensemble*), lalu hasil akhirnya ditentukan dari suara terbanyak seluruh pohon.

**Kekuatan.**
- Relatif tahan terhadap outlier dan lebih jarang overfitting dibanding satu pohon.
- Versi yang diperbaiki dengan cara memilih pohon berdasarkan akurasi dan tingkat korelasinya terbukti lebih akurat daripada lima varian RF pembanding (Sun dkk., 2024).

**Keterbatasan.**
- Model berperilaku sebagai kotak hitam sehingga sulit menjelaskan alasan tiap keputusan.
- RF konvensional dapat menurun kinerjanya pada dataset kecil atau yang memiliki banyak kategori kelas (Sun dkk., 2024).

**Cocok untuk.** Data tabular besar dengan campuran fitur numerik dan kategorikal.

### K-Nearest Neighbor (KNN)

**Prinsip kerja.** Untuk data baru, algoritma mencari `k` data latih yang paling mirip, lalu memilih kelas yang paling banyak muncul di antara tetangga tersebut. Tidak ada proses pelatihan yang berarti, sehingga disebut *lazy learner*.

**Kekuatan.**
- Mudah dipahami dan diimplementasikan.
- Tidak mengasumsikan bentuk distribusi data tertentu.

**Keterbatasan.**
- Pada data berdimensi tinggi, ukuran jarak menjadi kurang bermakna sehingga kinerja menurun. KNN juga mudah terpengaruh noise dan outlier (Halder dkk., 2024).
- Pemilihan `k` menentukan hasil: nilai kecil membuat model peka terhadap noise, sedangkan nilai besar memperberat komputasi dan berisiko underfitting (Halder dkk., 2024).

**Cocok untuk.** Dataset kecil sampai menengah dengan jumlah fitur yang terbatas.

### Support Vector Machine (SVM)

**Prinsip kerja.** Mencari *hyperplane* yang memisahkan kelas dengan jarak (margin) terbesar.

**Kekuatan.**
- Akurasi tinggi dan tetap kuat pada dimensi tinggi.
- Data yang tidak bisa dipisahkan secara linear dapat ditangani lewat fungsi kernel, yang memetakan data ke ruang fitur berdimensi lebih tinggi sehingga peluang pemisahan linear meningkat (Du dkk., 2024).

**Keterbatasan.**
- Pengaturan hyperparameter memerlukan ketelitian.
- Skalabilitas pada dataset besar masih menjadi tantangan terbuka (Du dkk., 2024).

**Cocok untuk.** Klasifikasi teks, analisis sentimen, dan pengenalan gambar dengan batas antar kelas yang cukup jelas.

### Naive Bayes

**Prinsip kerja.** Menghitung peluang setiap kelas menggunakan Teorema Bayes. Disebut "naive" karena setiap fitur dianggap tidak saling bergantung.

**Kekuatan.**
- Proses pelatihan sangat cepat dan efisien, termasuk pada data yang banyak.

**Keterbatasan.**
- Anggapan bahwa fitur saling bebas jarang terpenuhi pada data nyata.

**Cocok untuk.** Klasifikasi teks dan dokumen, misalnya penyaring spam atau pengelompokan topik berita.

---

## Panduan memilih algoritma

- Perlu penjelasan keputusan yang mudah dipahami: Decision Tree.
- Data tabular besar dan berantakan, akurasi stabil diutamakan: Random Forest.
- Data kecil dan ingin baseline sederhana: KNN.
- Data teks atau gambar dengan akurasi sebagai prioritas: SVM.
- Data teks dan butuh proses yang cepat: Naive Bayes.

---

## Daftar pustaka

Costa, V. G., & Pedreira, C. E. (2023). Recent advances in decision trees: An updated survey. *Artificial Intelligence Review, 56*(5), 4765-4800. https://doi.org/10.1007/s10462-022-10275-5

Du, K.-L., Jiang, B., Lu, J., Hua, J., & Swamy, M. N. S. (2024). Exploring kernel machines and support vector machines: Principles, techniques, and future directions. *Mathematics, 12*(24), 3935. https://doi.org/10.3390/math12243935

Halder, R. K., Uddin, M. N., Uddin, M. A., Aryal, S., & Khraisat, A. (2024). Enhancing K-nearest neighbor algorithm: A comprehensive review and performance analysis of modifications. *Journal of Big Data, 11*, 113. https://doi.org/10.1186/s40537-024-00973-y

Sun, Z., Wang, G., Li, P., Wang, H., Zhang, M., & Liang, X. (2024). An improved random forest based on the classification accuracy and correlation measurement of decision trees. *Expert Systems with Applications, 237*, 121549. https://doi.org/10.1016/j.eswa.2023.121549
