import os
import cv2
import base64
import tempfile
import ctypes
import ctypes.wintypes

import win32gui
import numpy as np
from flask import Flask, request, jsonify, Response, send_file

from classes import ImageDetector

app = Flask(__name__)

image_det = ImageDetector().load_model()

video_det = ImageDetector(model_path="yolo26m.pt", device=None) #the model for detection, you can use your model
video_det.load_model()
video_det.smooth = True

webcam_det = ImageDetector(model_path="yolo26m.pt", device=None)
webcam_det.load_model()
webcam_det.smooth = True

DWMWA_EXTENDED_FRAME_BOUNDS = 9 #constant for Windows OS


def get_window_rect(hwnd):
    try:
        rect = win32gui.GetWindowRect(hwnd)
    except Exception:
        return None
    try:
        dwr = ctypes.wintypes.RECT()
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(dwr), ctypes.sizeof(dwr),
        )
        if dwr.right > dwr.left and dwr.bottom > dwr.top:
            return dwr.left, dwr.top, dwr.right - dwr.left, dwr.bottom - dwr.top
    except Exception:
        pass
    return rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]


def capture_window_gdi(hwnd):
    bounds = get_window_rect(hwnd)
    if bounds is None:
        return None

    x, y, w, h = bounds
    if w <= 0 or h <= 0:
        return None

    import win32ui
    import win32con

    hwnd_dc = mfc_dc = save_dc = bitmap = None
    try:
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bitmap)
        save_dc.BitBlt((0, 0), (w, h), mfc_dc, (0, 0), win32con.SRCCOPY)

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
            (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
        )
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception:
        return None
    finally:
        try:
            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())
            if save_dc is not None:
                save_dc.DeleteDC()
            if mfc_dc is not None:
                mfc_dc.DeleteDC()
            if hwnd_dc is not None:
                win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass


def list_windows():
    windows = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append({"hwnd": hwnd, "title": title})

    win32gui.EnumWindows(cb, None)
    return windows


@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "endpoints": {
            "POST /detect/image": "Photo file",
            "POST /detect/image/base64": "Photo base64",
            "POST /detect/video": "Video",
            "GET /detect/webcam/<camera_id>": "Webcam",
            "GET /screen/windows": "List of opened windows",
            "POST /detect/screen": "Capture the window",
        },
    })


@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "file not found"}), 400

    file = request.files["file"]
    arr = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "couldn't decode the image"}), 400

    return jsonify(image_det.detect(frame))


@app.route("/detect/image/base64", methods=["POST"])
def detect_image_base64():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Need the field 'image'"}), 400

    try:
        image_bytes = base64.b64decode(data["image"])
    except Exception:
        return jsonify({"error": "invalid base64"}), 400

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "couldn't decode the image"}), 400

    return jsonify(image_det.detect(frame))


@app.route("/detect/video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "file not found"}), 400

    file = request.files["file"]
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, file.filename)
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        os.remove(input_path)
        return jsonify({"error": "couldn't open the video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(tmp_dir, "output.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        det = video_det.detect(frame)
        writer.write(video_det.draw(frame, det))
        frame_idx += 1

    cap.release()
    writer.release()

    if frame_idx == 0:
        os.remove(input_path)
        return jsonify({"error": "Video is empty"}), 400

    response = send_file(output_path, mimetype="video/mp4",
                         as_attachment=True, download_name="detected.mp4")

    @response.call_on_close
    def cleanup():
        try:
            os.remove(input_path)
            os.remove(output_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

    return response


@app.route("/detect/webcam/<int:camera_id>")
def detect_webcam(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return jsonify({"error": f"Camera {camera_id} not found"}), 400

    def generate():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                det = webcam_det.detect(frame)
                frame = webcam_det.draw(frame, det)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        finally:
            cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/screen/windows")
def get_windows():
    return jsonify({"windows": list_windows()})


@app.route("/detect/screen", methods=["POST"])
def detect_screen():
    data = request.get_json()
    if not data or "hwnd" not in data:
        return jsonify({"error": "Need the field 'hwnd'"}), 400

    frame = capture_window_gdi(data["hwnd"])
    if frame is None:
        return jsonify({"error": "couldn't open the window"}), 400

    return jsonify(image_det.detect(frame))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
