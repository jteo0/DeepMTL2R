# Rencana Metrik Evaluasi Proyek DeepMTL2R

Dokumen ini merangkum metrik yang akan digunakan untuk mengukur, membandingkan, dan memvalidasi eksperimen pada arsitektur DeepMTL2R. Evaluasi dibagi menjadi tiga fase berurutan untuk menjamin objektivitas komparasi dan kekuatan narasi riset.

---

## Fase 1: Validasi Kualitas Ranking (Baseline Comparison)
*Tujuan: Membuktikan bahwa modifikasi yang diusulkan (Matryoshka & Feature Gating) tidak merusak performa dasar dan dapat dibandingkan langsung dengan hasil paper asli DeepMTL2R.*

* **NDCG@30 (Main Task):** Metrik utama sesuai paper untuk mengukur kualitas urutan dokumen pada label relevansi asli (Task 0).
* **NDCG@30 (Auxiliary Tasks):** Pengukuran individual untuk tiap label tambahan (Click, Dwell Time, Quality Scores) guna melihat spesialisasi model.
* **Δm% (Average Relative Improvement):** Rata-rata persentase kenaikan/penurunan performa model multi-task dibandingkan dengan model single-task untuk semua tugas.
* **Mean Average Precision (MAP):** Metrik tambahan untuk melihat akurasi rata-rata dokumen relevan di seluruh daftar.

---

## Fase 2: Eksplorasi & Komparasi Internal (Matryoshka vs. Feature Gating)
*Tujuan: Membedah filosofi "Dense Flexibility" (Matryoshka) melawan "Sparse Selection" (Gating) pada kondisi ranking praktis.*

* **NDCG@10 & NDCG@20:** Mengukur ketajaman model pada posisi paling atas (halaman pertama pencarian). Metrik ini adalah standar industri dan esensial untuk membedakan arsitektur mana yang secara praktis lebih superior.
* **Effective Dimensionality Efficiency (MRL-only):** Skor NDCG pada berbagai fraksi dimensi (misal: 1/16, 1/8, 1/4 dimensi). Membuktikan seberapa "padat" informasi yang disandikan oleh Matryoshka.
* **Gating Sparsity Ratio (Gating-only):** Persentase fitur input yang diberikan bobot mendekati nol oleh mekanisme Gating. Menunjukkan tingkat efisiensi pembuangan *noise*.
* **Total Trainable Parameters (Δp):** Membandingkan ukuran penambahan model akibat modifikasi secara *hardware-agnostic*.
* **Robustness to Noisy Features:** Persentase penurunan skor NDCG ketika sebagian fitur sengaja diberi gangguan (*noise*). Metrik brilian untuk menguji apakah Gating secara eksplisit mampu "menutup pintu" bagi fitur buruk dibandingkan fleksibilitas alami Matryoshka.
* **Feature Importance Rank Correlation:** Korelasi *ranking* antara fitur yang diberi bobot tinggi oleh Gating dengan fitur yang aktif dominan pada dimensi-dimensi awal Matryoshka.

---

## Fase 3: Eksperimen Akhir (Late-Stage Fine-Tuning & Dynamics)
*Tujuan: Menganalisis ketahanan dan daya adaptasi kedua model (Matryoshka vs Gating) terhadap intervensi hyperparameter (Optimizer & Loss Weighting) di fase akhir pelatihan.*

### Prioritas 1: Perbandingan Optimizer (AdamW vs Lion vs SGD+Momentum)
* **Epochs to New Plateau (Convergence Speed)**
  * *Penjelasan:* Jumlah epoch tambahan yang diperlukan oleh model untuk kembali mentok (*plateau*) setelah optimizer diganti pada tahap akhir.
  * *Interpretasi Analisis:* Optimizer dengan angka epoch lebih kecil menunjukkan konvergensi yang lebih efisien. Jika Lion lebih cepat konvergen pada model Feature Gating daripada AdamW, ada kemungkinan arsitektur *sparse* lebih bersahabat dengan sifat regularisasi ketat milik Lion. Sebaliknya, Matryoshka (yang loss-nya bersarang) mungkin butuh momentum konstan (SGD+Momentum) untuk menavigasi lanskap loss-nya yang rumit.
* **Peak NDCG Delta (ΔNDCG)**
  * *Penjelasan:* Selisih antara NDCG@10 terbaik sebelum intervensi (checkpoint awal Eksperimen Akhir) dan NDCG@10 puncak setelah intervensi optimizer.
  * *Interpretasi Analisis:* Semakin besar nilai delta positif, semakin sukses arsitektur tersebut didorong keluar dari *local minima*. Jika Opsi C memiliki ΔNDCG jauh lebih tinggi dari Opsi A setelah di-tuning, berarti potensi maksimal Opsi C sempat tertahan oleh batasan AdamW.

### Prioritas 2: Strategi Pembobotan Loss (Uniform vs DWA / Uncertainty Weighting)
* **Task Performance Variance**
  * *Penjelasan:* Varian (sebaran) dari selisih performa NDCG antara *auxiliary tasks* terhadap *main task*.
  * *Interpretasi Analisis:* Jika varian menurun setelah DWA diterapkan, berarti pembobotan berhasil menyeimbangkan *trade-off*. Jika Feature Gating (C) sangat bergantung pada DWA untuk mencapai varian stabil dibanding Matryoshka (A), ini membuktikan bahwa arsitektur yang memangkas fitur sejak awal sangat rawan didominasi oleh salah satu tugas sehingga butuh bimbingan bobot dinamis.
  * *Contoh Numerik:*
    * Baseline (Uniform): Varian = 0.0150
    * Target (with DWA): Varian < 0.0050 (Penurunan > 60%)
    * Kesimpulan: DWA sukses jika varian drop di bawah 0.0050, menunjukkan stabilnya transfer pengetahuan lintas tugas.
* **Minority Task NDCG Retention Ratio**
  * *Penjelasan:* Rasio bertahannya performa tugas sekunder (tugas dengan label terjarang/terkecil) ketika strategi pembobotan diubah.
  * *Interpretasi Analisis:* Metrik ini menguji kekokohan representasi. Jika representasi padat Matryoshka mampu mempertahankan nilai *Retention Ratio* mendekati 1.0 pada pembobotan Uniform, itu membuktikan kekuatannya dalam berbagi informasi lintas tugas tanpa intervensi loss tingkat lanjut.

### Prioritas 3: Dinamika Gradien & Learning Rate
* **Gradient Conflict Norm (Cos Similarity)**
  * *Penjelasan:* Pengukuran sudut (*cosine similarity*) antara vektor gradien dari *main task* dan *auxiliary tasks* pada epoch-epoch konvergensi akhir.
  * *Interpretasi Analisis:* Nilai cosinus negatif berarti arah gradien tugas-tugas saling bertolak belakang (*destructive interference / negative transfer*). Anda bisa menganalisis arsitektur mana (A atau C) yang secara natural menjaga nilai konflik ini tetap positif di akhir masa pelatihan. Jika metode Gating mengurangi konflik lebih drastis, berarti pembuangan *noise* secara logis mencegah *clashing* antar tugas.
  * *Contoh Numerik:*
    * Baseline: Cosine Similarity = -0.15 (konflik tinggi)
    * Target: Cosine Similarity > 0.0 (konflik berhasil diredam)
    * Semakin mendekati 1.0, arah gradien antar tugas semakin sejalan.
* **Gradient Sparsity Tracking**
  * *Penjelasan:* Melacak persentase elemen vektor gradien yang bernilai sangat kecil atau nol (khususnya diamati pada model Gating).
  * *Interpretasi Analisis:* Membantu memberikan narasi untuk laporan. Jika banyak gradien mati, ini menjelaskan secara empiris mengapa model sulit menembus batas performa akhir dan mengapa metode optimisasi tertentu (seperti Cosine Annealing) dibutuhkan untuk "menendang" gradien agar kembali aktif.