import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
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
        self._control_callback = None
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
    img { width: 100%; height: auto; border: 1px solid #333; user-select: none; -webkit-user-drag: none; touch-action: none; }
    .hint { margin: 8px 0 0; color: #aaa; font-size: 14px; }
    .row { display: flex; gap: 8px; align-items: center; }
    button { background: #2a2a2a; color: #ddd; border: 1px solid #444; padding: 6px 10px; cursor: pointer; }
  </style>
</head>
<body>
  <div class="wrap">
    <h3>LiDAR + ZED Merge Stream</h3>
    <div class="row">
      <button id="resetBtn" type="button">Reset View (R)</button>
    </div>
    <img id="stream" src="/stream.mjpg" alt="stream" />
    <p class="hint">Right 3D panel only: drag to pan, wheel to zoom.</p>
  </div>
<script>
(() => {
  const stream = document.getElementById("stream");
  const resetBtn = document.getElementById("resetBtn");
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let pendingDx = 0;
  let pendingDy = 0;
  let panScheduled = false;

  function sendControl(payload) {
    fetch("/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true
    }).catch(() => {});
  }

  function in3DPanel(offsetX, width) {
    return offsetX >= (width / 3);
  }

  function flushPan() {
    panScheduled = false;
    if (pendingDx === 0 && pendingDy === 0) return;
    sendControl({ action: "pan_pixels", dx: pendingDx, dy: pendingDy });
    pendingDx = 0;
    pendingDy = 0;
  }

  stream.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    if (!in3DPanel(e.offsetX, stream.clientWidth)) return;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    e.preventDefault();
  });

  window.addEventListener("mouseup", () => {
    dragging = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    pendingDx += dx;
    pendingDy += dy;
    if (!panScheduled) {
      panScheduled = true;
      setTimeout(flushPan, 30);
    }
  });

  stream.addEventListener("wheel", (e) => {
    if (!in3DPanel(e.offsetX, stream.clientWidth)) return;
    const steps = e.deltaY < 0 ? 1 : -1;
    sendControl({ action: "zoom_steps", steps });
    e.preventDefault();
  }, { passive: false });

  resetBtn.addEventListener("click", () => {
    sendControl({ action: "reset_view" });
  });
})();
</script>
</body>
</html>
"""

        @self._app.route("/stream.mjpg")
        def stream():
            return Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self._app.route("/control", methods=["POST"])
        def control():
            if self._control_callback is None:
                return jsonify({"ok": False, "error": "control callback not set"}), 503
            payload = request.get_json(silent=True) or {}
            action = str(payload.get("action", ""))
            try:
                accepted = bool(self._control_callback(action, payload))
                return jsonify({"ok": accepted})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

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

    def set_control_callback(self, callback):
        self._control_callback = callback

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
