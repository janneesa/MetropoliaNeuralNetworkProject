# Flask V1 Backend

Small prototype backend for the `FinalProject` chat UI.

## Features

- Serves the built React app from `FinalProject/client/dist`
- `POST /api/chat` for chatting with the active model
- `GET /api/settings` and `PUT /api/settings` for backend-owned model settings
- `/reset` command support to clear the active model session

## Run

Install Flask if needed:

```bash
pip install -r FinalProject/flask_v1/requirements.txt
```

Start the backend from the repository root:

```bash
python FinalProject/flask_v1/run.py
```

Default address:

- `http://127.0.0.1:8000`

Environment variables:

- `FLASK_HOST`
- `FLASK_PORT`
- `FLASK_DEBUG`
- `EMMA_MODEL_PATH` to point Emma at a specific `.keras` file
