# Recipe 1: Image Classification dengan Pretrained CNN (PyTorch)
**Source: Python Image Processing Cookbook, Chapter 7**

---

## Struktur Folder

```
project/
├── classify.py              # script utama
├── requirements.txt         # dependensi
├── README.md
│
├── models/
│   └── imagenet_classes.txt # auto-download saat pertama jalan
│
├── images/                  # TARUH GAMBAR INPUT DI SINI
│   ├── cheetah.png
│   └── swan.png
│
└── outputs/                 # hasil plot otomatis tersimpan di sini
    ├── output_cheetah.png
    └── output_swan.png
```

---

## Perlu Dataset Tidak?

**TIDAK PERLU dataset apapun.**

- Model sudah pretrained di ImageNet (didownload otomatis dari internet saat pertama run)
- Label 1000 kelas ImageNet juga auto-download otomatis
- Yang perlu kamu sediakan hanya: **gambar input** yang ingin diklasifikasi

Taruh gambar apapun di folder `images/`, lalu edit baris `image_files` di `classify.py`:
```python
image_files = ["images/namafile.png", "images/gambar_lain.jpg"]
```

---

## Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Buat folder images dan taruh gambarmu
mkdir images
# copy gambar cheetah.png dan swan.png (atau gambar apapun) ke folder images/

# 3. Jalankan
python classify.py
```

**Catatan first run:** akan download bobot model dari internet (~500MB+ total untuk 4 model).
Download hanya sekali, setelah itu ter-cache di `~/.cache/torch/hub/checkpoints/`.

---

## Output

Setiap gambar menghasilkan:
1. **Gambar teranotasi** — teks prediksi dari 4 model ditulis di atas gambar
2. **Bar chart** — Top-5 kelas prediksi ResNet101 beserta persentase probabilitasnya
3. File PNG tersimpan di folder `outputs/`