# Bab 4 - Deep Learning (Perry Xiao, AI Programming with Python)

Folder:
- `Example/`  : Example 4.1 - 4.24 (kode sesuai buku)
- `Exercise/` : Exercise 4.1 - 4.20 (jawaban/modifikasi dari example terkait)

## Catatan tentang kode Example
Isi kode tidak diubah. Hanya baris yang terpotong karena lebar halaman buku
disambung kembali, dan tanda minus "-" hasil copy PDF dibersihkan.
Hal-hal berikut ada di buku dan sengaja dibiarkan:
- 4.3  : komentar tertulis "Example 4.4" (salah ketik di buku).
- 4.9  : tidak ada `import matplotlib.pyplot as plt` -> NameError. (Sudah ditambahkan di Exercise 4.7/4.8)
- 4.15 : memakai `!pip install` (sintaks Colab/Jupyter) dan `camera`, `image_size` tidak didefinisikan.
         (Diperbaiki di Exercise 4.13/4.14)
- 4.1  : teks buku bilang AND, tetapi data `[0,1,1,1]` sebenarnya sudah OR.
- 4.5, 4.7, 4.9 memakai `keras.layers.normalization`, `keras.layers.convolutional`, `keras.utils.np_utils`
  yang sudah dihapus di Keras/TensorFlow versi baru -> pakai TensorFlow 2.x lama (mis. 2.4)
  atau Google Colab versi lama.
- 4.11, 4.12, 4.17 butuh file `Elephant.jpg` di folder yang sama.
- 4.13 - 4.15 butuh webcam; 4.20 butuh internet (dataset CSV); 4.23/4.24 butuh `pip install transformers`.

## Catatan Exercise
- 4.1: OR (AND tersedia sebagai baris komentar). 4.2: 3 input (OR 3 input).
- 4.3: dense 10 -> 5 neuron + pencatat waktu training.
- 4.4: filter 6/16 -> 8/32. 4.5: dense 4096/4096/1000 -> 2048/2048/500.
- 4.6 / 4.8: tambah Conv2D(64,(3,3)) + MaxPooling + Dropout sebelum Flatten.
- 4.7a: 32 -> 64 filter. 4.7b: kernel (5,5) -> (3,3).
- 4.9 - 4.11: VGG16 -> VGG19. 4.12: tambah DenseNet121.
- 4.13: uncomment model lain satu per satu. 4.14: if-else pemilihan model.
- 4.15: n = 2. 4.16: semua filter. 4.17: n = 2, 4, 5 (layer 3 adalah pooling, tidak punya filter).
- 4.18 - 4.20: teks/kalimat/pertanyaan baru.
- Komentar hasil (4.3, 4.15-4.20) perlu ditulis setelah program dijalankan di komputer Anda.
