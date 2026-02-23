# Isyaratku - Deteksi BISINDO A-Z (Streamlit)

Aplikasi ini menampilkan deteksi huruf BISINDO A-Z secara real-time menggunakan kamera.

## Link Visualisasi Aplikasi

- Deploy (Streamlit Cloud): `https://your-app-name.streamlit.app`
- Lokal: `http://localhost:8501`

Ganti `https://your-app-name.streamlit.app` dengan URL deploy aplikasi kamu.

## Menjalankan Secara Lokal

```powershell
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Fitur Utama

- Deteksi gesture BISINDO A-Z via kamera browser
- Tampilan landmark tangan real-time
- Prediksi huruf pada frame video

## Dependensi

- streamlit
- streamlit-webrtc
- streamlit-autorefresh
- opencv-python-headless
- mediapipe
- numpy
- joblib
- av
- scikit-learn
