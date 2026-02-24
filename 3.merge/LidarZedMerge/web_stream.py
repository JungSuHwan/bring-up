from flask import Flask
from flask import jsonify, request, send_from_directory
import logging
import os
import threading
from werkzeug.serving import make_server


class WebFrameServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = int(port)
        self._app = Flask(__name__)
        self._server = None
        self._thread = None
        self._running = False
        self._control_callback = None
        self._state_callback = None
        self._setup_routes()

    def _setup_routes(self):
        @self._app.route("/")
        def index():
            base_dir = os.path.dirname(os.path.abspath(__file__))
            controller_dir = os.path.join(base_dir, "web_lidar_controller")
            return send_from_directory(controller_dir, "index.html")

        @self._app.route("/controller")
        def controller():
            base_dir = os.path.dirname(os.path.abspath(__file__))
            controller_dir = os.path.join(base_dir, "web_lidar_controller")
            return send_from_directory(controller_dir, "index.html")

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

    def set_control_callback(self, callback):
        self._control_callback = callback

    def set_state_callback(self, callback):
        self._state_callback = callback

    def start(self):
        if self._running:
            return
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
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
