# Struktur Direktori LaTeX - DeepMTL2R Paper

Direktori ini telah direstrukturisasi untuk memudahkan maintenance dan kolaborasi. Berikut penjelasan struktur:

## 📁 Struktur Direktori

```
latex/
├── main.tex                    # File utama - compile ini untuk generate PDF
├── preamble.tex               # Konfigurasi package dan setup LaTeX
├── sections/                  # Direktori untuk section-section paper
│   ├── 01_introduction.tex   # Bagian Introduction
│   ├── 02_method.tex         # Bagian Method (dengan subsections)
│   ├── 03_experiments.tex    # Bagian Experiments
│   ├── 04_results.tex        # Bagian Results and Evaluation
│   ├── 05_conclusion.tex     # Bagian Conclusion
│   └── 99_appendix.tex       # Bagian Appendix
├── bib/                       # Bibliography files
│   └── custom.bib            # Reference bibliography
├── images/                    # Gambar dan diagram
│   ├── DeepMLT2R.png
│   ├── FeatureGating.png
│   └── Matryoshka.png
├── templates/                 # Template untuk komponen reusable (optional)
├── acl.sty                    # ACL stylesheet (format paper)
├── acl_latex.tex              # Template ACL (archived)
├── acl_lualatex.tex           # Template ACL LuaLaTeX variant (archived)
├── acl_natbib.tex             # Template ACL dengan natbib (archived)
└── README.md                  # File ini
```

## 🚀 Cara Menggunakan

### 1. Kompilasi Dokumen
```bash
cd latex/
pdflatex main.tex
bibtex main.tex
pdflatex main.tex
pdflatex main.tex
```

Atau gunakan LaTeX editor favorit Anda (e.g., TeXstudio, Overleaf) dan buka `main.tex`.

### 2. Menambah/Edit Section Baru
1. Buat file baru di direktori `sections/` dengan nama `NN_section_name.tex`
2. Tulis konten LaTeX Anda
3. Tambahkan `\input{sections/NN_section_name.tex}` di `main.tex` pada urutan yang tepat

### 3. Menambah/Edit Gambar
- Simpan gambar di direktori `images/`
- Reference di tex file dengan: `\includegraphics[width=\columnwidth]{images/filename.png}`

### 4. Menambah Reference Baru
- Edit `bib/custom.bib` dan tambahkan entry BibTeX
- Cite dengan `\cite{key_name}` di tex file

## 📝 Keuntungan Struktur Modular

✅ **Lebih Rapi**: Setiap section terpisah, mudah dinavigasi  
✅ **Kolaboratif**: Multipel orang bisa kerja di section berbeda tanpa conflict  
✅ **Maintainable**: Perubahan di section tertentu tidak affect keseluruhan  
✅ **Scalable**: Mudah menambah/remove section tanpa repot  
✅ **Reusable**: Template di `templates/` dapat digunakan di multiple documents  

## 💡 Tips

- Selalu compile dari direktori `latex/` agar path relative bekerja dengan baik
- Gunakan descriptive filenames dan nomor urutan (01, 02, 03) untuk section
- Simpan custom templates yang reusable di `templates/` folder
- Archive file-file template lama di folder root (seperti `acl_latex.tex`, dsb)

## 📚 Referensi

- ACL Style Guide: https://acl-org.github.io/ACLPUB/formatting.html
- LaTeX Beginner: https://www.latex-project.org/help/

---

*Last Updated: May 28, 2026*
