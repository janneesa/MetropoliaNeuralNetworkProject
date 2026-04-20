# React + Vite

## Chatbox endpoint

This project now includes a simple chat UI in `src/App.jsx`.

- Configure the endpoint with `VITE_CHAT_ENDPOINT` (and optionally `VITE_CHAT_PROVIDER`) — see `.env`.

The app sends a POST request with JSON that includes:

- `prompt` (string)
- `temperature` (number)
- `max_tokens` (number)

### Generic mode (default)

- `VITE_CHAT_PROVIDER=generic` (or omit `VITE_CHAT_PROVIDER` entirely)
- Sends: `POST <endpoint>` with JSON like:
  - `{ "prompt": "...", "temperature": 1, "max_tokens": 1024 }`
- Accepts:
  - JSON with any of: `response`, `answer`, `message`, `text`
  - OpenAI-style: `{ choices: [{ message: { content: "..." } }] }` or `{ choices: [{ text: "..." }] }`
  - or a plain-text response body

#### Using a Flask API endpoint (recommended)

Point the frontend at your Flask route and keep `generic` mode. Your Flask backend can run your own model (or call any internal service) and return a normalized response to the UI.

Example `.env`:

```bash
VITE_CHAT_PROVIDER=generic
VITE_CHAT_ENDPOINT=http://127.0.0.1:5000/api/chat
```

Your Flask endpoint should return something like:

```json
{ "response": "..." }
```

Minimal Flask adapter example (your custom model, returns a normalized response):

```py
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # dev only; restrict origins in production


@app.post("/api/chat")
def chat():
	data = request.get_json(force=True) or {}
	prompt = (data.get("prompt") or "").strip()
	temperature = float(data.get("temperature", 1))
	max_tokens = int(data.get("max_tokens", 1024))

	if not prompt:
		return jsonify({"error": "Missing prompt"}), 400

	# TODO: Replace this with your own model inference.
	# Examples:
	# - Call a Python function that runs a local model
	# - Call an internal HTTP service (then you'd use `requests`)
	# - Call your own GPU worker queue, etc.
	def generate_reply(prompt: str, temperature: float, max_tokens: int) -> str:
		return f"You said: {prompt}"  # placeholder

	reply = generate_reply(prompt, temperature, max_tokens)
	return jsonify({"response": reply})


if __name__ == "__main__":
	app.run(host="127.0.0.1", port=5000, debug=True)
```

Install deps:

```bash
pip install flask flask-cors
```

### Optional: Ollama mode

- `VITE_CHAT_PROVIDER=ollama`
- Requires `VITE_OLLAMA_MODEL`.
- If `VITE_CHAT_ENDPOINT` is the base URL (e.g. `http://127.0.0.1:11434`), the app will POST to `/api/generate`.
- You can also set `VITE_CHAT_ENDPOINT` directly to `http://127.0.0.1:11434/api/chat`.

Example `.env`:

```bash
VITE_CHAT_PROVIDER=ollama
VITE_CHAT_ENDPOINT=http://127.0.0.1:11434
VITE_OLLAMA_MODEL=phi3
```

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
