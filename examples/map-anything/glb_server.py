"""Simple HTTP server that serves GLB files with a model-viewer web page."""

import os
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pyarrow as pa
from dora import Node

SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MapAnything 3D Viewer</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0/model-viewer.min.js"></script>
  <style>
    body {{ margin: 0; background: #1a1a2e; }}
    model-viewer {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <model-viewer src="/{glb_file}" auto-rotate camera-controls
    shadow-intensity="1" environment-image="neutral" exposure="0.8">
  </model-viewer>
</body>
</html>
"""

latest_glb = None


class ViewerHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves the model-viewer page on / and GLB files from OUTPUT_DIR."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            if latest_glb:
                html = HTML_TEMPLATE.format(glb_file=Path(latest_glb).name)
            else:
                html = "<html><body><h2>Waiting for reconstruction...</h2><script>setTimeout(()=>location.reload(),3000)</script></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress request logs


def _start_server():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    handler = partial(ViewerHandler, directory=OUTPUT_DIR)
    server = HTTPServer(("0.0.0.0", SERVER_PORT), handler)
    print(f"[glb-server] Serving on http://0.0.0.0:{SERVER_PORT}")
    server.serve_forever()


def main():
    """Run the GLB viewer server as a dora node."""
    global latest_glb

    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()

    node = Node()
    pa.array([])  # initialize pyarrow

    for event in node:
        if event["type"] == "INPUT":
            if "glb_path" in event["id"]:
                path = event["value"].to_pylist()[0]
                latest_glb = path
                print(f"[glb-server] New GLB: {path}")
                print(
                    f"[glb-server] View at http://0.0.0.0:{SERVER_PORT}"
                )
        elif event["type"] == "ERROR":
            raise RuntimeError(event["error"])


if __name__ == "__main__":
    main()
