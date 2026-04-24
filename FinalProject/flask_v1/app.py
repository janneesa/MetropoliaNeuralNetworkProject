from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory, stream_with_context

from .backend import ChatBackend


def create_app() -> Flask:
    project_dir = Path(__file__).resolve().parents[1]
    client_dist_dir = project_dir / "client" / "dist"

    app = Flask(__name__)
    backend = ChatBackend()

    def ensure_client_build() -> None:
        if client_dist_dir.exists():
            return
        abort(
            503,
            description="Client build not found. Run `npm run build` in FinalProject/client first.",
        )

    @app.get("/api/settings")
    def get_settings():
        return jsonify(backend.get_settings_payload())

    @app.put("/api/settings")
    @app.post("/api/settings")
    def update_settings():
        payload = request.get_json(silent=True) or {}
        return jsonify(backend.update_settings(payload))

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        if backend.should_stream(payload):
            def generate():
                for event in backend.stream_chat(prompt=prompt, payload=payload):
                    yield json.dumps(event) + "\n"

            return Response(
                stream_with_context(generate()),
                mimetype="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            return jsonify(backend.chat(prompt=prompt, payload=payload))
        except Exception as error:  # noqa: BLE001
            return jsonify({"error": str(error)}), 500

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def serve_client(path: str):
        ensure_client_build()

        if path.startswith("api/"):
            abort(404)

        requested_path = client_dist_dir / path
        if path and requested_path.is_file():
            return send_from_directory(client_dist_dir, path)

        return send_from_directory(client_dist_dir, "index.html")

    return app


app = create_app()
