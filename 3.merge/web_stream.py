import threading
import time

import cv2
import numpy as np
from flask import Flask, Response
from werkzeug.serving import make_server


class WebFrameServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = int(port)
        self._app = Flask(__name__)
        self._server = None
        self._thread = None
        self._running = False
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg = None
        self._setup_routes()

    def _setup_routes(self):
        @self._app.route("/")
        def index():
            return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LiDAR + ZED Merge Stream</title>
  <style>
    body { margin: 0; background: #111; color: #ddd; font-family: Arial, sans-serif; }
    .wrap { padding: 10px; }
    img { width: 100%; height: auto; border: 1px solid #333; }
  </style>
</head>
<body>
  <div class="wrap">
    <h3>LiDAR + ZED Merge Stream</h3>
    <img src="/stream.mjpg" alt="stream" />
  </div>
</body>
</html>
"""

        @self._app.route("/stream.mjpg")
        def stream():
            return Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def _mjpeg_generator(self):
        while self._running:
            jpeg = None
            with self._jpeg_lock:
                jpeg = self._latest_jpeg
            if jpeg is None:
                time.sleep(0.03)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(0.03)

    def update_frame(self, frame):
        if frame is None:
            return
        try:
            if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                return

            channels = frame.shape[2]
            if channels == 3:
                # Accept RGB by default. If source is BGR, output color may look swapped.
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif channels == 4:
                # ZED Mat often comes as BGRA/RGBA depending on source path.
                # Try BGRA first for practical compatibility.
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                return

            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return
            with self._jpeg_lock:
                self._latest_jpeg = encoded.tobytes()
        except Exception:
            # Keep rendering loop stable even if encoding fails temporarily.
            pass

    def start(self):
        if self._running:
            return
        self._server = make_server(self.host, self.port, self._app, threaded=True)
        self._running = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
