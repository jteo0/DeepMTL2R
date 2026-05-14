# DeepMTL2R Extensions: Improving Multi-Task Learning to Rank through Modern Techniques

**Proyek Tugas Kelompok – Kuliah Temu-Balik Informasi (CSCE604135)**  
**Semester Genap 2025/2026**

---

## Overview

Proyek ini melakukan **reproduksi** dan **ekstensi** terhadap **DeepMTL2R** (Amazon Science, 2026), sebuah library modern untuk Deep Multi-Task Learning to Rank. Kami fokus pada integrasi berbagai teknik dari mata kuliah Temu-Balik Informasi, khususnya di bidang Neural Learning to Rank, feature representation, efficient fine-tuning, dan optimization.

Kami mengusulkan **dua eksperimen utama** yang saling melengkapi untuk menghasilkan kontribusi yang jelas dan bernilai akademis.

---

## Research Questions

Bergantung pada dua eksperimen utama yang dipilih, pertanyaan riset akan difokuskan pada:
1. **RQ Utama**: Bagaimana efektivitas arsitektur yang diusulkan (misal: Matryoshka / LoRA / Feature Gating) dalam meningkatkan performa atau efisiensi Multi-Task Learning to Rank dibandingkan *baseline*?
2. **RQ Lanjutan (Optimisasi Akhir)**: Bagaimana pilihan optimizer, loss weighting strategy, dan gradient dynamics memengaruhi *trade-off* antar tugas serta kemampuan menembus batasan performa (*late-stage convergence*) pada arsitektur yang telah dimodifikasi?

---

## Daftar Eksperimen Utama

### Matryoshka Feature Projection
**Judul**: Matryoshka Representation Learning for Flexible Feature Projection in Deep Multi-Task Learning to Rank
**Deskripsi**: Modifikasi layer `FCModel` menggunakan teknik Matryoshka Representation Learning agar model menghasilkan *dense embedding* dengan dimensi fleksibel (misal: 64, 128, 256) tanpa perlu *retraining* ulang.
**Metrik Utama (Wajib)**: NDCG@10, NDCG@20 per task, Memory usage, Convergence speed, Hypervolume.

### Dynamic Feature Gating (Penyaringan Fitur)
**Judul**: Sparse Dynamic Feature Gating for Efficient Multi-Task Learning to Rank
**Deskripsi**: Menambahkan lapisan *Learnable Masking* sebelum `FCModel` untuk mematikan atau memberi bobot nol secara dinamis pada fitur numerik MSLR30K yang dianggap tidak relevan (*noise*).
**Metrik Utama (Wajib)**: NDCG@10, NDCG@20 per task, Tingkat *sparsity* (persentase fitur yang dimatikan), Training speed.

---

## Eksperimen Akhir (Tahap Uji Komparasi Lanjutan)

**Judul**: Comparative Analysis of Optimizers and Loss Weighting Strategies on Modified Architectures
**Deskripsi**: Analisis mendalam yang diaplikasikan secara **simetris** pada hasil (*checkpoint*) dari DUA eksperimen utama yang telah dipilih di atas.

**Aturan Main:**
Terapkan variabel di bawah ini pada model dari kedia eksperimen secara adil, lalu bandingkan respon/metrik akhirnya (seperti NDCG puncak dan pola konvergensi gradien).

#### Variabel Pengujian (Berdasarkan Prioritas Waktu):

**Prioritas 1 (Sangat Wajib): Perbandingan Optimizer**
*   Uji kedua *checkpoint* dengan **AdamW** (default) vs **SGD + Momentum**. 
*   *Tujuan:* Melihat arsitektur model mana yang lebih tangguh dan lebih mudah didorong keluar dari batasan performa (*plateau*) pada tahap akhir pelatihan.

**Prioritas 2 (Jika Kuota Tersedia): Loss Weighting Strategies**
*   Uji kedua *checkpoint* dengan **Uniform Weighting** vs **Uncertainty Weighting** atau **Dynamic Weight Averaging (DWA)**.

**Prioritas 3 (Opsional): Learning Rate & Gradient Dynamics**
*   Terapkan **Cosine Annealing** / **OneCycleLR**.
*   Lakukan analisis visual arah gradien (*Gradient norm & conflict*) saat model mendekati konvergensi maksimal.

**Catatan Pelaksanaan:**
Eksperimen Akhir ini **tidak memerlukan full training dari titik nol**. Anda cukup memuat file *checkpoint* terbaik dari 2 Eksperimen Utama yang telah selesai dilatih, lalu jalankan *late-stage fine-tuning* selama beberapa *epoch* menggunakan variabel pembanding di atas.
