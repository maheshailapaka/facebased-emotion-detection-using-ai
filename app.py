# streamlit_app.py
import streamlit as st
import cv2
import time
import os
import csv
from fer import FER
from PIL import Image
from datetime import datetime

st.set_page_config(page_title="Realtime Emotion Detection", layout="centered")
st.title("Realtime Emotion Detection (Streamlit + OpenCV + FER)")

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "emotion_log.csv")

def ensure_log():
    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "emotion", "confidence"])

def log_detection(emotion, confidence):
    with open(LOG_PATH, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), emotion, f"{confidence:.4f}"])

camera_index = st.sidebar.number_input("Camera index", min_value=0, max_value=4, value=0, step=1)
scale = st.sidebar.slider("Frame scale (for speed)", 0.25, 1.0, 0.6)
start = st.sidebar.button("Start Camera")
stop = st.sidebar.button("Stop Camera")
show_csv = st.sidebar.checkbox("Show log table", value=False)

placeholder = st.empty()
status = st.empty()

ensure_log()
detector = FER(mtcnn=True)

if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'running' not in st.session_state:
    st.session_state.running = False

if start:
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(camera_index)
    st.session_state.running = True

if stop:
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

try:
    while st.session_state.running:
        cap = st.session_state.cap
        if cap is None or not cap.isOpened():
            status.error("Camera not available. Check camera index and permissions.")
            break

        ret, frame = cap.read()
        if not ret:
            status.error("Failed to read frame")
            break

        if scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_emotions(rgb)

        for face in results:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            score = emotions[top_emotion]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{top_emotion}: {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            log_detection(top_emotion, score)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        placeholder.image(img, use_container_width=True)
        status.text("Running... press Stop to end")

        time.sleep(0.03)  # control refresh speed

except Exception as e:
    st.error(f"Error: {e}")

finally:
    if st.session_state.get('cap'):
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.running = False

if show_csv:
    import pandas as pd
    try:
        df = pd.read_csv(LOG_PATH)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to read log CSV: {e}")
