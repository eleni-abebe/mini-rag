"""
Minimal auth flow: signup -> login (issues JWT) -> verify_token (guards
the /query endpoint).

In-memory user store is intentional -- this project is about the CI/CD and
RAG mechanics, not building a production user system. Swap `_users_db` for
a real database later; `signup`/`login`/`verify_token` are the seam.
"""
from __future__ import annotations

import hashlib
import os
import time

import jwt

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 3600

# username -> password hash
_users_db: dict[str, str] = {}


def _hash_password(password: str, salt: str = "static-dev-salt") -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def signup(username: str, password: str) -> bool:
    """Register a new user. Returns False if the username is taken."""
    if username in _users_db:
        return False
    _users_db[username] = _hash_password(password)
    return True


def login(username: str, password: str) -> str | None:
    """Return a signed JWT on success, None on bad credentials."""
    stored_hash = _users_db.get(username)
    if stored_hash is None or stored_hash != _hash_password(password):
        return None
    payload = {"sub": username, "exp": time.time() + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> str | None:
    """Return the username encoded in a valid token, else None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None
