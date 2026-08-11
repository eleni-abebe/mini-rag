"""
FastAPI app: this is what Railway actually runs.

Endpoints:
  GET  /health          -- for uptime checks / Railway healthcheck
  POST /signup           {username, password}
  POST /login             {username, password} -> {token}
  POST /ingest            {documents: [str, ...]} -> loads docs into memory
  POST /query   (auth)    {question} -> agentic RAG answer, requires Bearer token
"""
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.agent import agentic_answer
from app.auth import login as auth_login
from app.auth import signup as auth_signup
from app.auth import verify_token
from app.embeddings import embed_text

app = FastAPI(title="Agentic RAG Mini-Project")

# Toy in-memory document store, populated via /ingest.
_documents: list[str] = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for data science.",
    "CI/CD pipelines automate testing and deployment.",
    "Retrieval-augmented generation combines search with language models.",
]


class Credentials(BaseModel):
    username: str
    password: str


class IngestRequest(BaseModel):
    documents: list[str]


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/signup")
def signup(creds: Credentials):
    if not auth_signup(creds.username, creds.password):
        raise HTTPException(status_code=400, detail="username already exists")
    return {"status": "created"}


@app.post("/login")
def login(creds: Credentials):
    token = auth_login(creds.username, creds.password)
    if token is None:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return {"token": token}


@app.post("/ingest")
def ingest(req: IngestRequest):
    _documents.extend(req.documents)
    return {"status": "ok", "total_documents": len(_documents)}


@app.post("/query")
def query(req: QueryRequest, authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    username = verify_token(token)
    if username is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")

    result = agentic_answer(req.question, _documents, embed_text)
    return {"user": username, **result}
