# AI-Based Smart Traffic Density Analysis with Prioritization for Emergency Services

A software prototype that estimates traffic density from camera video, suggests an adaptive green-signal time, and switches to an emergency override when an emergency vehicle (ambulance, fire truck, police vehicle) is detected.

## Team

- Sreedevi K V (MGP24ECE056)
- Aiby K Binu (MGP24CE008)
- Alam Mubarak R (MGP24CE009)
- Fardeen Muhammed Sadique (MGP24RA046)

## Features

- Vehicle detection and counting (car, motorcycle, bus, truck) with YOLOv8n
- Density level (LOW / MEDIUM / HIGH) and adaptive green time (15 s to 60 s by default)
- Emergency-vehicle classification with a MobileNetV2 transfer-learning model
- Emergency override when a vehicle crosses the classification threshold
- Streamlit web app with three input modes:
  - **Upload Video:** annotated output video, charts and CSV log
  - **Live Webcam:** browser camera through streamlit-webrtc
  - **RTSP / IP Camera:** RTSP URL, phone-camera URL, local video file or camera index, using a latest-frame reader so the display stays real-time
- Adjustable detection confidence, minimum crop size, emergency threshold, density boundaries and green-time limits

## How it works

1. Each frame is read from the chosen source (and resized in live mode for speed).
2. YOLOv8n detects vehicles; the count gives a density level and a green time.
3. Each vehicle crop of at least 50 x 50 px is converted to RGB, resized to 224 x 224 and classified in one batch by MobileNetV2.
4. If any emergency probability is at or above the threshold (default 0.5), the signal state becomes EMERGENCY OVERRIDE.
5. The annotated frame, metrics and a CSV log are produced.

## Results

Measured in the Colab notebook (`AI_DataScience.ipynb`):

| Item | Value |
|---|---|
| Dataset | 1,646 images (965 non-emergency, 681 emergency) |
| Split | 80 : 20 (1,317 train, 329 validation) |
| Training | MobileNetV2 base frozen, Dense(64) + Dropout(0.3) + sigmoid, Adam, 8 epochs |
| Validation accuracy | 92.7% for the saved model (epoch 8); best 95.4% (epoch 5) |
| Training accuracy | 98.4% (epoch 8), so mild overfitting |
| Majority-class baseline | 58.6% |
| Night emergency clip | Override fired on 87 of 95 processed frames (91.6%) |

The last row is the share of frames with an override, not accuracy, because the frames were not labelled one by one. Precision, recall and a confusion matrix are not reported yet.

## Dataset

[Emergency Vehicles Identification](https://www.kaggle.com/datasets/abhisheksinghblr/emergency-vehicles-identification), Kaggle (1,646 labelled images).

## Setup and run

Python 3.10 or 3.11 is recommended (TensorFlow may not support the newest versions).

```
pip install -r requirements.txt
streamlit run app.py
```

The app needs the trained classifier `emergency_classifier.h5` in the same folder as `app.py`. Create it by running the training cells of the notebook in Google Colab and downloading the file. Without it, detection and density still work but emergency detection is disabled. `yolov8n.pt` is downloaded automatically on first run.

### RTSP / IP camera mode

Choose **RTSP / IP Camera**, enter a source and tick **Run stream**:

- RTSP camera: `rtsp://user:pass@192.168.1.50:554/stream1`
- Phone (IP Webcam app): `http://192.168.1.20:8080/video`
- Local video file (for testing): `C:/path/to/traffic.mp4`
- Local camera index: `0`

Lower **Processing width** in the sidebar if the stream is slow.

## Known limitations

- Vehicles are counted per frame; there is no tracking, so the same vehicle is counted again in every frame.
- The emergency decision uses a single frame, so the override can flicker, and a distant normal vehicle can occasionally be flagged.
- Vehicles smaller than 50 px are not classified, so distant emergency vehicles are missed.
- The classifier was trained mostly on clear, close-up images, not CCTV footage, and the COCO-trained detector may struggle with Indian mixed traffic.
- Density thresholds (5 and 12) are set by hand; there is no per-lane logic or yellow-clearance handling.
- Speed on a CPU-only laptop is limited.

## Future work

Temporal smoothing, retraining with augmentation and early stopping, vehicle tracking (ByteTrack), per-lane regions of interest, realistic signal preemption, fine-tuning YOLO on emergency vehicles, siren audio detection, and a fixed-time versus adaptive comparison in SUMO.

## Course

AI Blend for Engineers Project, B.Tech. Third Year, Saintgits College of Engineering (Autonomous)
