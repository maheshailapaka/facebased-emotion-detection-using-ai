# emotion-detection-ai

simple python project: realtime face-based emotion detection using opencv + fer
includes:
- emotion_realtime.py  (opencv window)
- streamlit_app.py     (streamlit browser UI)
- requirements.txt
- logs/emotion_log.csv (created when first detection logged)

## quick start

1. create virtualenv and activate
   - mac/linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - windows:
     ```powershell
     python -m venv venv
     venv\Scripts\activate
     ```

2. install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3a. run opencv realtime window:
   ```bash
   python emotion_realtime.py
   ```
   press `q` to quit. detections are logged to `logs/emotion_log.csv`.

3b. run streamlit UI:
   ```bash
   streamlit run streamlit_app.py
   ```
   use sidebar to start/stop camera. detections are logged to `logs/emotion_log.csv`.

## notes
- first run may download model weights used by `fer` and `mtcnn`.
- if you have gpu-enabled tensorflow, performance will be better; otherwise CPU is fine but may be slower.
- if Streamlit UI seems slow, lower the frame scale in the sidebar.

