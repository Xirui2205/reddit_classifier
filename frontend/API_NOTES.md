# Backend API Notes

Discovered from `/backend/app`:

## Auth
- **POST** `/api/auth/login` — obtain JWT. Body: `{ "email": string, "password": string }`. Response: `{ "access_token": string, "token_type": "bearer" }`. Auth not required.
- **POST** `/api/auth/register` — admin-only create user. Body matches `UserCreate` (email, full_name?, role?, password). Response: `UserOut`. Requires bearer token with admin role.
- **GET** `/api/auth/me` — current user profile. Response: `UserOut`. Requires bearer token.

## Topics
- **GET** `/api/topics` — list all topics. Requires bearer token.
- **POST** `/api/topics` — create topic (admin). Body: `TopicCreate` with fields code, title?, description?, domain?, target_tokens_per_session, total_token_target, min_turns?, max_turns?, persona_id?, is_active?.
- **PUT** `/api/topics/{topic_id}` — update topic (admin). Body: partial fields from `TopicUpdate`.

## Personas
- **GET** `/api/personas` — list persona templates. Requires bearer token.
- **POST** `/api/personas` — create persona (admin). Body: `PersonaCreate` with code, name?, description?, prompt_text, is_active?.
- **PUT** `/api/personas/{persona_id}` — update persona (admin). Body: partial `PersonaUpdate`.

## Sessions (chat annotation)
- **POST** `/api/sessions/start` — start a new annotation session for current user. Response: `{ session: SessionOut, first_message: MessageOut }`.
- **POST** `/api/sessions/{session_id}/reply` — send annotator reply. Body: `{ text: string }`. Response includes session summary, user_message, optional assistant_message, session_finished flag, and optional `new_session` payload when chaining.
- **GET** `/api/sessions/{session_id}` — fetch session details with messages. Requires bearer token.

## Stats
- **GET** `/api/stats/me` — annotator stats summary. Requires bearer token.
- **GET** `/api/stats/admin/overview` — admin overview metrics. Requires admin bearer token.

## Missing features
- No endpoints found for translation tasks, leaderboards, or review workflows. Frontend surfaces those areas as “backend support missing”.
