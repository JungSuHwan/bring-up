import threading
import logging
import os

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
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
        self._frame_cond = threading.Condition(self._jpeg_lock)
        self._latest_jpeg = None
        self._frame_id = 0
        self._control_callback = None
        self._state_callback = None
        self._jpeg_quality = 70
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
  <title>LiDAR + ZED Web View</title>
  <style>
    :root {
      --bg: #0f1218;
      --panel: #171b24;
      --text: #e7ecf6;
      --muted: #9ea8bb;
      --line: #2a3242;
      --accent: #56b6ff;
    }
    html, body {
      margin: 0;
      height: 100%;
      background: radial-gradient(1200px 600px at 20% -10%, #1e2636 0%, var(--bg) 45%);
      color: var(--text);
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
    }
    .layout {
      display: grid;
      grid-template-rows: auto 1fr auto;
      height: 100%;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(23, 27, 36, 0.8);
      backdrop-filter: blur(4px);
    }
    .title {
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
    }
    .stage {
      padding: 10px;
      min-height: 0;
    }
    .stream-box {
      height: 100%;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #0b0f15;
      overflow: hidden;
      position: relative;
    }
    .stream {
      width: 100%;
      height: 100%;
      object-fit: contain;
      user-select: none;
      -webkit-user-drag: none;
      touch-action: none;
      display: block;
    }
    .stream.mode-map {
      width: 150%;
      transform: translateX(-33.3333%);
      transform-origin: top left;
    }
    .hintbar {
      border-top: 1px solid var(--line);
      background: rgba(23, 27, 36, 0.9);
      color: var(--muted);
      font-size: 13px;
      padding: 9px 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .ok { color: #90e39a; }
    .warn { color: #ffd580; }
    .mono { font-family: Consolas, "Courier New", monospace; }
  </style>
</head>
<body>
  <div class="layout">
    <div class="topbar">
      <div class="title">LiDAR + ZED Web Display</div>
      <div id="meta" class="meta mono">state: loading...</div>
    </div>
    <div class="stage">
      <div class="stream-box">
        <img id="stream" class="stream" src="/stream.mjpg" alt="stream" />
      </div>
    </div>
    <div class="hintbar">
      <span class="ok">Drag</span>: pan map,
      <span class="ok">Wheel</span>: zoom,
      <span class="warn">View</span>: <span class="mono">?view=full</span> or <span class="mono">?view=map</span>,
      <span class="warn">Console</span>: extrinsic tuning keys
    </div>
  </div>
<script>
(() => {
  const stream = document.getElementById("stream");
  const meta = document.getElementById("meta");
  const params = new URLSearchParams(window.location.search);
  const viewMode = (params.get("view") || "full").toLowerCase();
  if (viewMode === "map") {
    stream.classList.add("mode-map");
  }
  let mouseDown = false;
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
    if (viewMode === "map") {
      return true;
    }
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
    mouseDown = true;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    e.preventDefault();
  });

  window.addEventListener("mouseup", () => {
    mouseDown = false;
    dragging = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (!mouseDown) return;
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

  async function refreshState() {
    try {
      const res = await fetch("/lidar_state", { cache: "no-store" });
      if (!res.ok) {
        meta.textContent = "state: unavailable";
        return;
      }
      const data = await res.json();
      const lidars = Array.isArray(data.lidars) ? data.lidars : [];
      if (lidars.length === 0) {
        meta.textContent = "lidar: none";
        return;
      }
      const selectedName = data.selected_name || lidars[0].name;
      const target = lidars.find((x) => x.name === selectedName) || lidars[0];
      const off = target.offset || { x: 0, y: 0, z: 0 };
      const yaw = Number(target.yaw_deg || 0);
      meta.textContent = `lidar=${target.name} status=${target.connected ? "UP" : "DOWN"} fps=${Number(target.fps || 0).toFixed(1)} off=(${Number(off.x).toFixed(3)},${Number(off.y).toFixed(3)},${Number(off.z).toFixed(3)}) yaw=${yaw.toFixed(2)}`;
    } catch (_) {
      meta.textContent = "state: error";
    }
  }

  setInterval(refreshState, 1000);
  refreshState();
})();
</script>
</body>
</html>
"""

        @self._app.route("/controller")
        def controller():
            base_dir = os.path.dirname(os.path.abspath(__file__))
            controller_dir = os.path.join(base_dir, "web_lidar_controller")
            return send_from_directory(controller_dir, "index.html")

        @self._app.route("/stream.mjpg")
        def stream():
            return Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "X-Accel-Buffering": "no",
                },
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

        @self._app.route("/lidar_state", methods=["GET"])
        def lidar_state():
            if self._state_callback is None:
                return jsonify({"ok": False, "error": "state callback not set", "lidars": []}), 503
            try:
                payload = self._state_callback()
                if not isinstance(payload, dict):
                    payload = {}
                payload["ok"] = True
                return jsonify(payload)
            except Exception as e:
                return jsonify({"ok": False, "error": str(e), "lidars": []}), 500

    def _mjpeg_generator(self):
        last_frame_id = -1
        while self._running:
            with self._frame_cond:
                while self._running and self._frame_id == last_frame_id:
                    self._frame_cond.wait(timeout=1.0)
                if not self._running:
                    break
                jpeg = self._latest_jpeg
                last_frame_id = self._frame_id
            if jpeg is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

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

            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)])
            if not ok:
                return
            with self._frame_cond:
                self._latest_jpeg = encoded.tobytes()
                self._frame_id += 1
                self._frame_cond.notify_all()
        except Exception:
            # Keep rendering loop stable even if encoding fails temporarily.
            pass

    def set_control_callback(self, callback):
        self._control_callback = callback

    def set_state_callback(self, callback):
        self._state_callback = callback

    def set_jpeg_quality(self, quality):
        q = int(quality)
        self._jpeg_quality = max(30, min(95, q))

    def start(self):
        if self._running:
            return
        # Suppress Werkzeug access logs (e.g. repeated /lidar_state polling).
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self._server = make_server(self.host, self.port, self._app, threaded=True)
        self._running = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
