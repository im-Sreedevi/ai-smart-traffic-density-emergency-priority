
# AI-Based Smart Traffic Density Analysis with Prioritization for Emergency Services

An AI-based system that analyzes real-time traffic density at road intersections and dynamically prioritizes emergency vehicles (ambulances, fire trucks) by overriding traffic signals for immediate passage.

## Team
- Sreedevi K V (MGP24ECE056)
- Aiby K Binu (MGP24CE008)
- Alam Mubarak R (MGP24CE009)
- Fardeen Muhammed Sadique (MGP24RA046)

## Features
- Real-time vehicle detection and counting using YOLOv8
- Adaptive traffic signal timing based on live vehicle density
- Emergency vehicle classification using a MobileNetV2-based CNN
- Automatic signal override when an emergency vehicle is detected
- Interactive Streamlit web app (video upload + live webcam modes)

## Tech Stack
- Python, OpenCV
- Ultralytics YOLOv8 (vehicle detection)
- TensorFlow/Keras, MobileNetV2 (emergency vehicle classification)
- Streamlit (web interface)
- Google Colab (training environment)

## Results
- Emergency vehicle classifier: 94.2% validation accuracy
- Correctly detects 0% false positives on traffic-only footage
- ~94.7% correct emergency-override trigger rate on ambulance test footage

## Dataset
[Emergency Vehicles Identification](https://www.kaggle.com/datasets/abhisheksinghblr/emergency-vehicles-identification) — Kaggle, 1,646 labelled images.

## Running the Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. YOLOv8 detects vehicles in each frame of a traffic camera feed
2. Vehicle count is converted into a density level (Low/Medium/High)
3. Traffic signal green-time is allocated proportionally to density
4. Each detected vehicle is checked by a classifier for emergency-vehicle status
5. If an emergency vehicle is found, the signal is immediately overridden to GREEN

## Course
AI Blend for Engineers Project — B.Tech. Third Year, Saintgits College of Engineering (Autonomous)
