# Open people detection API

Detect people in photos, videos, webcams, and screens. It works through YOLO and outputs the result via the Flask API.
## Structure

```
people/
├── api_main.py              # the main api file
├── yolo26m.pt               # model (example)
├── classes/
│   ├── base_detect.py       # Basic class for detection 
│   ├── image_detect.py      # Image detection
│   ├── video_detect.py      # Video detection
│   ├── webcam_detect.py     # Webcam detection
│   └── screen_detect.py     # Screen detection (throught mss)
```

## Start

```bash
pip install flask opencv-python ultralytics numpy mss pywin32 

#OR 

pip install -r requirements.txt

#Start app

python api_main.py
```

The server starts at `http://localhost:5000`.

## API

| Method | Path | Description |
|-------|------|----------|
| POST | `/detect/image` | photo (file)|
| POST | `/detect/image/base64` | photo (base64)|
|POST | `/detect/video` | video (file) -> mp4 with frames |
| GET | `/detect/webcam/0` | stream from camera (MJPEG) |
| GET | `/screen/windows` | list of open windows |
| POST | `/detect/screen` | window capture by 'hwnd'|

### Examples

```bash
# Photo
curl -X POST http://localhost:5000/detect/image -F "file=@photo.jpg"

# base64
curl -X POST http://localhost:5000/detect/image/base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>"}'

# Webcam
# http://localhost:5000/detect/webcam/0

# List of windows
curl http://localhost:5000/screen/windows

# Capture the screen
curl -X POST http://localhost:5000/detect/screen \
  -H "Content-Type: application/json" \
  -d '{"hwnd": 12345}'
```

### Answer JSON

```json
{
  "count": 2,
  "detections": [
    {"bbox": [120, 50, 300, 400], "confidence": 0.87},
    {"bbox": [500, 80, 650, 380], "confidence": 0.72}
  ]
}
```

## Variables to configure

All parameters in `classes/base_detect.py`:

```python
MODEL_PATH = "yolo26m.pt" #model
CONFIDENCE = 0.35 #detection threshold
IOU = 0.5 #NMS threshold
IMGSZ = 1280 #input image size

MIN_BOX_W = 12 #minimum width
MIN_BOX_H = 28 #minimum height
MIN_BOX_AREA_RATIO = 0.00008 #minimum frame area
MIN_PERSON_ASPECT = 0.65 #minimum height/width ratio
MAX_PERSON_ASPECT = 7.0 #maximum height/width ratio

NMS_IOU = 0.5 #my NMS after YOLO
SMOOTH_ALPHA = 0.6 #smoothing frames between frames
```

## Contacts

If you have any questions, please contact TG: @ALRU8