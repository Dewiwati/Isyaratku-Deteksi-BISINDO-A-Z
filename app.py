import os
from collections import deque
from threading import Lock

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
import av
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model_bisindo.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler_bisindo.pkl")

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise FileNotFoundError("model_bisindo.pkl atau scaler_bisindo.pkl tidak ditemukan di folder streamlit/")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

if "result_text" not in st.session_state:
    st.session_state.result_text = ""

def extract_features_from_bgr(frame_bgr: np.ndarray, hands: mp_hands.Hands) -> tuple[np.ndarray | None, np.ndarray]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, frame_bgr

    coords: list[float] = []
    sorted_hands = sorted(results.multi_hand_landmarks[:2], key=lambda hand: hand.landmark[0].x)

    for hand_landmarks in sorted_hands:
        mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x - base_x, lm.y - base_y])

        # Visual marker untuk pergelangan tangan.
        px = int(base_x * w)
        py = int(base_y * h)
        cv2.circle(frame_bgr, (px, py), 5, (0, 255, 0), -1)

    if len(coords) < 84:
        coords.extend([0.0] * (84 - len(coords)))

    feature_vector = np.array(coords[:84], dtype=np.float32).reshape(1, -1)
    return feature_vector, frame_bgr


st.set_page_config(page_title="BISINDO Streamlit", page_icon="BIS", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;800&display=swap');

    :root {
        --bg-1: #f5efe4;
        --bg-2: #f7f9ff;
        --ink: #1f2a37;
        --muted: #4b5563;
        --accent: #0f766e;
        --accent-2: #14b8a6;
        --panel: rgba(255, 255, 255, 0.78);
        --stroke: rgba(15, 23, 42, 0.08);
    }

    .stApp {
        font-family: "Manrope", sans-serif;
        color: var(--ink);
        background:
            radial-gradient(900px 400px at 95% -10%, #fde68a 0%, transparent 60%),
            radial-gradient(900px 380px at -10% 110%, #bae6fd 0%, transparent 60%),
            linear-gradient(135deg, var(--bg-1), var(--bg-2));
    }

    footer {visibility: hidden;}
    footer:after {content: '';}

    .main > div {
        background: var(--panel);
        backdrop-filter: blur(6px);
        border: 1px solid var(--stroke);
        border-radius: 22px;
        padding: 1.1rem 1.25rem 1.4rem 1.25rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }

    .bis-hero {
        margin-bottom: 1rem;
        border-radius: 18px;
        padding: 1.1rem 1.1rem 0.2rem 1.1rem;
        background: linear-gradient(120deg, rgba(20, 184, 166, 0.12), rgba(56, 189, 248, 0.09));
        border: 1px solid rgba(20, 184, 166, 0.2);
    }

    .bis-kicker {
        font-family: "Space Grotesk", sans-serif;
        color: var(--accent);
        letter-spacing: .08em;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }

    .bis-title {
        font-family: "Space Grotesk", sans-serif;
        color: #111827;
        font-weight: 700;
        line-height: 1.05;
        font-size: clamp(1.9rem, 2.9vw, 3rem);
        margin: 0 0 .5rem 0;
    }

    .bis-sub {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: .75rem;
    }

    .stAlert {
        border-radius: 14px;
        border: 1px solid var(--stroke);
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(15, 118, 110, 0.35);
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: white;
        font-weight: 700;
    }

    .stExpander {
        border-radius: 14px;
        border: 1px solid var(--stroke);
        background: rgba(255, 255, 255, 0.72);
    }

    .bis-guide {
        margin-top: 0.7rem;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        border: 1px solid rgba(15, 118, 110, 0.22);
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(14, 165, 233, 0.08));
    }

    .bis-guide-title {
        font-family: "Space Grotesk", sans-serif;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: .45rem;
    }

    .bis-guide-item {
        color: #334155;
        margin: 0.15rem 0;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bis-hero">
      <div class="bis-kicker">BISINDO Recognition</div>
      <h1 class="bis-title">Deteksi BISINDO A-Z</h1>
      <p class="bis-sub">Penerjemah gerakan tangan BISINDO untuk mengenali huruf A-Z secara real-time melalui kamera.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


class BISINDOVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.prediction_buffer: deque[str] = deque(maxlen=10)
        self.latest_text = "Menunggu tangan..."
        self.raw_prediction = "-"
        self.stable_prediction = "-"
        self.stability = 0.0
        self.lock = Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Kompensasi mirror bawaan browser agar tampilan tetap orientasi asli.
        img = cv2.flip(img, 1)

        features, annotated = extract_features_from_bgr(img, self.hands)

        if features is not None:
            scaled = scaler.transform(features)
            raw_prediction = str(model.predict(scaled)[0])

            self.prediction_buffer.append(raw_prediction)
            stable_prediction = max(set(self.prediction_buffer), key=self.prediction_buffer.count)
            stable_count = self.prediction_buffer.count(stable_prediction)
            stability = stable_count / len(self.prediction_buffer)

            with self.lock:
                self.raw_prediction = raw_prediction
                self.stable_prediction = stable_prediction
                self.stability = stability
                self.latest_text = f"Prediksi: {raw_prediction} | Stabil: {stable_prediction} ({stability:.0%})"

            cv2.rectangle(annotated, (0, 0), (620, 70), (0, 180, 0), 2)
            cv2.putText(
                annotated,
                f"BISINDO: {stable_prediction}",
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
        else:
            with self.lock:
                self.raw_prediction = "-"
                self.stable_prediction = "-"
                self.stability = 0.0
                self.latest_text = "Tangan belum terdeteksi."

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


ctx = webrtc_streamer(
    key="bisindo-realtime",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_processor_factory=BISINDOVideoProcessor,
)

if not ctx.state.playing:
    st.markdown(
        """
        <div class="bis-guide">
            <div class="bis-guide-title">Panduan Penggunaan</div>
            <div class="bis-guide-item">1. Klik <b>Start</b> lalu izinkan akses kamera di browser.</div>
            <div class="bis-guide-item">2. Pastikan tangan terlihat jelas dengan pencahayaan cukup.</div>
            <div class="bis-guide-item">3. Tahan gesture 1-2 detik agar hasil stabil lebih akurat.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Rerun berkala agar panel status/snapshot prediksi sinkron dengan frame kamera.
    st_autorefresh(interval=500, key="bisindo_live_refresh")
    st.markdown(
        """
        <div class="bis-guide">
            <div class="bis-guide-title">Kamera Aktif</div>
            <div class="bis-guide-item">Mulai lakukan gesture BISINDO A-Z di depan kamera.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if ctx.video_processor:
    with ctx.video_processor.lock:
        st.markdown(f"**Status Prediksi**  \n{ctx.video_processor.latest_text}")
