# MindSeek Annotator Frontend

A Vite + React + TypeScript frontend for the annotation platform. Backend lives separately under `/backend` and is read-only.

## Prerequisites
- Node.js 18+

## Setup
1. Install dependencies (internet access required):
   ```bash
   npm install
   ```
2. Create a `.env` file in this directory:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

## Development
- Start the dev server:
  ```bash
  npm run dev
  ```
- Build for production:
  ```bash
  npm run build
  ```
- Preview the production build:
  ```bash
  npm run preview
  ```

Ensure the FastAPI backend is running separately (e.g., `uvicorn app.main:app --reload` from the `backend` folder).
