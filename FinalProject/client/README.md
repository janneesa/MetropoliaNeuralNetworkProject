# React + Vite

## Chatbox endpoint

This project now includes a simple chat UI in `src/App.jsx`.

- Configure the endpoint with `VITE_CHAT_ENDPOINT` (and optionally `VITE_CHAT_PROVIDER`) — see `.env.example`.

### Generic mode (default)

- `VITE_CHAT_PROVIDER=generic`
- Sends: `POST <endpoint>` with JSON: `{ "prompt": "..." }`.
- Accepts: JSON with `answer`/`message`/`text` or a plain-text body.

### Ollama mode

- `VITE_CHAT_PROVIDER=ollama`
- Requires `VITE_OLLAMA_MODEL`.
- If `VITE_CHAT_ENDPOINT` is the base URL (e.g. `http://127.0.0.1:11434`), the app will POST to `/api/generate`.
- You can also set `VITE_CHAT_ENDPOINT` directly to `http://127.0.0.1:11434/api/chat`.

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
