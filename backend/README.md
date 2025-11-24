# Corpus Collection Platform (Backend)

FastAPI backend implementing the corpus collection workflow described in the specification. It includes JWT authentication, topic/persona management, session lifecycle logic, and basic statistics endpoints.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
pip install -r backend/requirements.txt
```

2. Start the API (SQLite by default):

```bash
uvicorn app.main:app --reload --app-dir backend
```

3. Update the following environment variables to target PostgreSQL or customize secrets:

- `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@localhost:5432/corpus`)
- `JWT_SECRET`
- `JWT_ALGORITHM` (default `HS256`)
- `ACCESS_TOKEN_EXPIRE` (minutes)

## Module layout

- `app/main.py`: FastAPI app setup and router registration.
- `app/models/`: SQLAlchemy models matching the corpus specification.
- `app/schemas/`: Pydantic schemas for requests/responses.
- `app/services/`: Business logic for auth, topics, personas, sessions, and stats.
- `app/routers/`: API routes grouped by feature.
- `app/utils/`: Token counting and password hashing helpers.

## Session lifecycle

- `POST /api/sessions/start` selects the least-covered active topic and seeds the first assistant question using the persona prompt.
- `POST /api/sessions/{id}/reply` stores the annotator reply (token-counted via `tiktoken`), updates progress, and either sends a follow-up question or completes the session. When a session meets token/turn thresholds, it updates topic counters and automatically spawns the next eligible session when available.

## Dashboards and stats endpoints

- Annotator stats: `GET /api/stats/me` returns token totals, per-topic rollups, and per-day token counts for the current annotator.
- Admin overview: `GET /api/stats/admin/overview` summarizes global tokens, sessions per topic, per-topic tallies, and annotator rankings.

## Authentication

- `POST /api/auth/login` issues a JWT access token.
- `POST /api/auth/register` (admin only) creates a new user.
- `GET /api/auth/me` returns the current user profile.

Use the `Authorization: Bearer <token>` header for authenticated routes.
