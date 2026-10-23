"""
AI-Based Smart Traffic Density Analysis with Emergency Vehicle Prioritization
Streamlit demo app — supports uploaded video, browser webcam, and RTSP / IP camera streams.

Setup:
    1. Place trained `emergency_classifier.h5` (downloaded from the Colab
       notebook) in this same folder.
    2. pip install -r requirements.txt
    3. streamlit run app.py
"""

import os

# Must be set BEFORE importing cv2: use TCP for RTSP (more reliable than UDP,
# avoids smeared / corrupted frames) and time out instead of hanging forever.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)

import time
import tempfile
import threading

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="AI Smart Traffic Density & Emergency Priority",
    page_icon="🚦",
    layout="wide",
)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emergency_classifier.h5")
STREAM_TIMEOUT_S = 10  # give up if no new frame arrives for this long


@st.cache_resource(show_spinner="Loading YOLOv8 vehicle detector...")
def load_yolo():
    return YOLO("yolov8n.pt")


@st.cache_resource(show_spinner="Loading emergency vehicle classifier...")
def load_classifier():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


yolo_model = load_yolo()
ev_model = load_classifier()

if ev_model is None:
    st.warning(
        f"⚠️ Emergency vehicle classifier not found at `{MODEL_PATH}`. "
        "Download `emergency_classifier.h5` from your Colab notebook (Step 7b) "
        "and place it in this app's folder. Vehicle detection & density will "
        "still work, but emergency detection will be disabled."
    )


def detect_vehicles(frame, conf=0.25):
    results = yolo_model.predict(frame, conf=conf, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls_id], float(box.conf[0])))
    return boxes


def density_level(count, low_thresh, high_thresh):
    if count <= low_thresh:
        return "LOW"
    elif count <= high_thresh:
        return "MEDIUM"
    else:
        return "HIGH"


def green_time_for(count, min_green, max_green, min_count=0, max_count=20):
    count = max(min_count, min(count, max_count))
    ratio = (count - min_count) / (max_count - min_count + 1e-6)
    return int(min_green + ratio * (max_green - min_green))


def classify_boxes_batch(frame, boxes, threshold=0.5, min_crop_size=50):
    """Batch-classify all vehicle crops in one frame. Crops smaller than
    min_crop_size are skipped (too small/far to classify reliably)."""
    if ev_model is None or not boxes:
        return [(False, 0.0)] * len(boxes)

    results = [None] * len(boxes)
    crops, crop_indices = [], []

    for i, (x1, y1, x2, y2, _, _) in enumerate(boxes):
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        h, w = crop.shape[:2] if crop.size else (0, 0)
        if crop.size == 0 or w < min_crop_size or h < min_crop_size:
            results[i] = (False, 0.0)
            continue
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, IMG_SIZE)
        crops.append(preprocess_input(crop.astype("float32")))
        crop_indices.append(i)

    if crops:
        batch = np.stack(crops, axis=0)
        probs = ev_model(batch, training=False).numpy().flatten()
        for idx, p in zip(crop_indices, probs):
            results[idx] = (p >= threshold, float(p))

    return results


def annotate_frame(frame, boxes, classifications, count, dlevel, gtime):
    emergency_present = False
    for (x1, y1, x2, y2, label, conf), (emergency, prob) in zip(boxes, classifications):
        color = (0, 0, 255) if emergency else (0, 255, 0)
        if emergency:
            emergency_present = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    signal_state = "GREEN (EMERGENCY OVERRIDE)" if emergency_present else f"GREEN {gtime}s ({dlevel})"
    signal_color = (0, 0, 255) if emergency_present else (0, 200, 0)

    cv2.putText(frame, f"Vehicles: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Signal: {signal_state}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, signal_color, 2)
    return frame, emergency_present


def process_one_frame(frame, conf, low_thresh, high_thresh, min_green, max_green,
                       cls_threshold, min_crop_size):
    boxes = detect_vehicles(frame, conf=conf)
    count = len(boxes)
    dlevel = density_level(count, low_thresh, high_thresh)
    gtime = green_time_for(count, min_green, max_green)
    classifications = classify_boxes_batch(frame, boxes, threshold=cls_threshold, min_crop_size=min_crop_size)
    frame, emergency_present = annotate_frame(frame, boxes, classifications, count, dlevel, gtime)
    return frame, {"count": count, "density": dlevel, "green_time": gtime, "emergency": emergency_present}


class LatestFrameReader:
    """Reads a stream in a background thread and keeps ONLY the newest frame.

    Without this, OpenCV buffers old frames while the model is busy, so the
    output drifts further and further behind real time. Here, slow processing
    simply means skipped frames, never lag.
    """

    def __init__(self, source):
        # "0", "1" -> local camera index; anything else -> URL or file path
        src = int(source) if str(source).strip().isdigit() else str(source).strip()
        self.cap = cv2.VideoCapture(src)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # A local file would otherwise be read as fast as possible. Pace it at
        # its own FPS so it behaves like a real live camera (and loop it).
        self.is_file = isinstance(src, str) and os.path.isfile(src)
        self.frame_delay = 0.0
        if self.is_file:
            self.frame_delay = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 25)
        self._lock = threading.Lock()
        self._frame = None
        self._stopped = False
        self.last_ok = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        if self.cap.isOpened():
            self._thread.start()

    def is_opened(self):
        return self.cap.isOpened()

    def _loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the file
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self.last_ok = time.time()
            if self.frame_delay:
                time.sleep(self.frame_delay)

    def read(self):
        """Return the newest unseen frame, or None if there isn't one yet."""
        with self._lock:
            frame, self._frame = self._frame, None
        return frame

    def release(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        self.cap.release()


st.sidebar.title("🚦 Settings")
mode = st.sidebar.radio("Input mode", ["Upload Video", "Live Webcam", "RTSP / IP Camera"])

st.sidebar.markdown("---")
st.sidebar.subheader("Detection")
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)
min_crop_size = st.sidebar.slider("Min crop size for emergency classification (px)", 20, 150, 50, 5)
cls_threshold = st.sidebar.slider("Emergency classification threshold", 0.1, 0.9, 0.5, 0.05)

st.sidebar.subheader("Density thresholds")
low_thresh = st.sidebar.slider("Low/Medium boundary", 1, 15, 5)
high_thresh = st.sidebar.slider("Medium/High boundary", low_thresh + 1, 25, 12)

st.sidebar.subheader("Signal timing")
min_green = st.sidebar.slider("Min green time (s)", 5, 30, 15)
max_green = st.sidebar.slider("Max green time (s)", 30, 90, 60)

if mode == "Upload Video":
    frame_skip = st.sidebar.slider("Process every Nth frame", 1, 10, 3)

if mode == "RTSP / IP Camera":
    proc_width = st.sidebar.slider("Processing width (px) - lower = faster", 320, 1280, 640, 64)


st.title("AI-Based Smart Traffic Density Analysis")
st.caption("with Prioritization for Emergency Services — live demo")


if mode == "Upload Video":
    uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        input_path = tfile.name

        if st.button("▶️ Process Video", type="primary"):
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            progress = st.progress(0, text="Processing video...")
            log = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                frame, stats = process_one_frame(
                    frame, conf, low_thresh, high_thresh, min_green, max_green,
                    cls_threshold, min_crop_size
                )
                stats["frame"] = frame_idx
                log.append(stats)
                out.write(frame)

                if total_frames:
                    progress.progress(min(frame_idx / total_frames, 1.0),
                                       text=f"Processing frame {frame_idx}/{total_frames}...")

            cap.release()
            out.release()
            progress.empty()

            log_df = pd.DataFrame(log)
            st.session_state["log_df"] = log_df
            st.session_state["out_path"] = out_path

        if "log_df" in st.session_state:
            log_df = st.session_state["log_df"]
            out_path = st.session_state["out_path"]

            st.success("Processing complete!")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Annotated Output")
                st.video(out_path)
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download annotated video", f, file_name="annotated_output.mp4")

            with col2:
                st.subheader("Summary")
                n_emergency = int(log_df["emergency"].sum())
                st.metric("Frames processed", len(log_df))
                st.metric("Emergency override frames", f"{n_emergency} / {len(log_df)}")
                st.metric("Avg. vehicles/frame", f"{log_df['count'].mean():.1f}")
                st.metric("Peak vehicles in a frame", int(log_df["count"].max()))

                csv = log_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download log (CSV)", csv, file_name="traffic_log.csv")

            st.subheader("Vehicle count & green time over the video")
            st.line_chart(log_df.set_index("frame")[["count", "green_time"]])

            st.subheader("Density level distribution")
            st.bar_chart(log_df["density"].value_counts())


elif mode == "Live Webcam":
    st.info(
        "Live webcam mode uses your browser's camera. Click **Start** below and "
        "allow camera access. Detected vehicles and emergency status are overlaid "
        "on the live feed."
    )

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av

        class TrafficProcessor(VideoProcessorBase):
            def __init__(self):
                self.conf = conf
                self.low_thresh = low_thresh
                self.high_thresh = high_thresh
                self.min_green = min_green
                self.max_green = max_green
                self.cls_threshold = cls_threshold
                self.min_crop_size = min_crop_size

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img, _ = process_one_frame(
                    img, self.conf, self.low_thresh, self.high_thresh,
                    self.min_green, self.max_green, self.cls_threshold, self.min_crop_size
                )
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="traffic-demo",
            video_processor_factory=TrafficProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    except ImportError:
        st.error(
            "Live webcam mode requires `streamlit-webrtc`. Install it with:\n\n"
            "```\npip install streamlit-webrtc av\n```\n\n"
            "Then restart the app. In the meantime, use **Upload Video** mode instead."
        )


else:  # RTSP / IP Camera
    st.info(
        "Enter the stream URL of an IP camera, then tick **Run stream**. "
        "Untick it to stop. Examples:\n\n"
        "- RTSP camera: `rtsp://user:pass@192.168.1.50:554/stream1`\n"
        "- Phone (IP Webcam app): `http://192.168.1.20:8080/video`\n"
        "- Local video file (for testing): `C:/Users/you/Downloads/traffic.mp4`\n"
        "- Local camera index: `0`"
    )

    stream_url = st.text_input(
        "Stream URL",
        value=st.session_state.get("stream_url", ""),
        placeholder="rtsp://username:password@192.168.1.50:554/stream1",
    )
    st.session_state["stream_url"] = stream_url

    run = st.checkbox("▶️ Run stream", key="rtsp_run")

    alert_ph = st.empty()
    frame_ph = st.empty()
    m1, m2, m3, m4 = st.columns(4)
    count_ph, dens_ph, green_ph, fps_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()
    status_ph = st.empty()

    if run:
        if not stream_url.strip():
            st.warning("Please enter a stream URL first.")
        else:
            status_ph.info("Connecting to stream...")
            reader = LatestFrameReader(stream_url)

            try:
                if not reader.is_opened():
                    status_ph.error(
                        "Could not open the stream. Check the URL, username/password, "
                        "and that the camera is on the same network as this computer."
                    )
                else:
                    smoothed_fps = 0.0
                    n_frames = 0
                    while True:
                        frame = reader.read()

                        if frame is None:
                            if time.time() - reader.last_ok > STREAM_TIMEOUT_S:
                                status_ph.error(
                                    f"No frames received for {STREAM_TIMEOUT_S}s. "
                                    "The stream may have dropped or ended. "
                                    "Untick and re-tick **Run stream** to reconnect."
                                )
                                break
                            time.sleep(0.01)
                            continue

                        status_ph.empty()

                        h0, w0 = frame.shape[:2]
                        if w0 > proc_width:
                            frame = cv2.resize(frame, (proc_width, int(h0 * proc_width / w0)))

                        t0 = time.time()

                        frame, stats = process_one_frame(
                            frame, conf, low_thresh, high_thresh, min_green, max_green,
                            cls_threshold, min_crop_size
                        )

                        dt = max(time.time() - t0, 1e-6)
                        n_frames += 1
                        if n_frames > 1:  # first frame includes model warm-up, ignore it
                            smoothed_fps = (1 / dt) if smoothed_fps == 0 else 0.8 * smoothed_fps + 0.2 * (1 / dt)

                        frame_ph.image(frame, channels="BGR", output_format="JPEG", use_container_width=True)
                        count_ph.metric("Vehicles", stats["count"])
                        dens_ph.metric("Density", stats["density"])
                        green_ph.metric("Green time", "OVERRIDE" if stats["emergency"] else f"{stats['green_time']} s")
                        fps_ph.metric("Processing FPS", f"{smoothed_fps:.1f}")

                        if stats["emergency"]:
                            alert_ph.error("🚨 Emergency vehicle detected — signal override active")
                        else:
                            alert_ph.success("Normal operation — adaptive green timing")
            finally:
                # Runs when the loop breaks OR when Streamlit stops the script
                # (e.g. you untick the checkbox), so the camera is always released.
                reader.release()


st.markdown("---")
st.caption(
    "AI Blend for Engineers Project — AI-Based Smart Traffic Density Analysis "
    "along with Prioritization for Emergency Services"
)
