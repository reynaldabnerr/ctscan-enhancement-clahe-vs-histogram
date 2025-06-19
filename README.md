# CT Scan Enhancement: CLAHE vs Histogram Equalization

Proyek ini bertujuan untuk membandingkan dua metode peningkatan kualitas citra, yaitu Histogram Equalization (HE) dan Contrast Limited Adaptive Histogram Equalization (CLAHE), dalam meningkatkan kualitas visual citra CT scan dari pasien SARS-CoV-2.

## 📂 Dataset
Dataset yang digunakan adalah **SARS-CoV-2 CT Scan Dataset** yang tersedia secara publik:

- Sumber: [Kaggle - SARS-CoV-2 CT Scan Dataset](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset)
- Total gambar: 2482 CT scan (1252 positif, 1230 negatif)

## ⚙️ Metode Pemrosesan
1. **Gaussian Filtering** - untuk mengurangi noise awal
2. **Sharpening** - untuk menegaskan tepi dan struktur penting
3. **Histogram Equalization (HE)** - peningkatan kontras global
4. **CLAHE** - peningkatan kontras lokal dengan pembatasan

## 🖼️ Contoh Hasil Visualisasi

Gambar berikut menunjukkan perbandingan antara citra original, hasil sharpening, HE, dan CLAHE.

![Hasil Visualisasi](results/comparison_Covid%20(1000).png)

## 📈 Evaluasi
Metrik yang digunakan:
- **Entropy** - mengukur kompleksitas informasi
- **RMS Contrast** - mengukur kontras citra
- **PSNR (Peak Signal-to-Noise Ratio)** - mengukur kesamaan dengan citra asli
- **SSIM (Structural Similarity Index)** - menilai kesamaan struktur citra

Contoh hasil evaluasi kuantitatif pada citra `Covid (1000).png`:

| Metode                 | Entropy | RMS    | PSNR (dB) | SSIM    |
|------------------------|---------|--------|-----------|---------|
| Original               | 7.13    | 54.44  | –         | –       |
| Gaussian + Sharpened   | 7.19    | 57.81  | 27.69     | 0.9041  |
| Histogram Equalization | 6.94    | 74.55  | 11.37     | 0.5350  |
| CLAHE                  | 7.68    | 62.80  | 15.18     | 0.6374  |

## 🖼️ Output
Hasil visual disimpan otomatis di folder:


Gambar terdiri atas:
- Citra original
- Citra hasil Gaussian + Sharpened
- Hasil HE
- Hasil CLAHE
- Histogram dan metrik di bawah tiap gambar

## 📚 Referensi
- Soares, Eduardo et al. (2020). SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification. medRxiv. [DOI](https://doi.org/10.1101/2020.04.24.20078584)
- Angelov, Plamen & Soares, Eduardo. (2020). Towards explainable deep neural networks (xDNN). Neural Networks, 130, 185-194.

---



