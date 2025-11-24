from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routers import auth, topics, personas, sessions, stats

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Corpus Collection Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(topics.router)
app.include_router(personas.router)
app.include_router(sessions.router)
app.include_router(stats.router)


@app.get("/")
def healthcheck():
    return {"status": "ok"}
