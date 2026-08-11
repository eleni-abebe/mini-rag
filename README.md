# Agentic RAG Mini-Project

![CI/CD](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml/badge.svg)

A small agentic RAG (Retrieval-Augmented Generation) API with a full CI/CD
pipeline: lint + test + build on every push,auto-deploy to Railway  on
`main`.


## How it  works

```
query --> [auth check] --> [agent loop] --> answer
                              |
                    retrieve --> good enough? --yes--> return
                        ^              |
                        |              no
                        +--- reformulate query
```

- **chunking.py** — splits documents into overlapping word chunks.
- **embeddings.py** — deterministic hashing-trick vectorizer (no API key
  needed, so CI never touches the network).
- **retrieval.py** — cosine similarity, ranks chunks best-first.
- **agent.py** — the "agentic" loop: retrieve, check confidence, and if it's
  too low, reformulate the query and try again (bounded by `max_iterations`).
- **auth.py** — signup/login/JWT verify, guards the `/query` endpoint.
- **main.py** — FastAPI app wiring it all together.

## Run locally

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST localhost:8000/signup -d '{"username":"me","password":"pw"}' -H 'Content-Type: application/json'
curl -X POST localhost:8000/login  -d '{"username":"me","password":"pw"}' -H 'Content-Type: application/json'
# copy the token from the response, then:
curl -X POST localhost:8000/query -d '{"question":"Where is the Eiffel Tower?"}' \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer <token>'
```

## Run tests

```bash
pytest tests/ -v
```

10 tests covering chunking,embeddings, similarity order, and the auth flow.

## CI/CD pipeline

On every push, GitHub Actions runs, in order: **lint** (ruff) → **test**
(pytest) → **build** (confirms the app imports cleanly with prod deps).
On pushes to `main` only, a final **deploy** job ships to Railway.

### One-time setup to make deploy work

1. Create a Railway project and a service inside it.
2. In Railway: Project Settings → get a **Railway API token**.
3. In your GitHub repo: Settings → Secrets and variables → Actions, add:
   - `RAILWAY_TOKEN` — the token from step 2.
   - `RAILWAY_SERVICE_NAME` — the service name in your Railway project.
4. Push to `main` and watch the **Actions** tab.
