from datetime import datetime
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..models import CorpusMessage, CorpusSession, CorpusTopic, PersonaTemplate, User
from ..schemas.session import SessionReply
from ..utils.tokens import count_tokens
from .topic import pick_next_topic


SYSTEM_PROMPT_TEMPLATE = (
    "You are an English-speaking AI collecting training data.\n"
    "Persona: {persona_prompt}\n"
    "Rules:\n- ALWAYS answer in ENGLISH.\n- User replies in Swahili/Sheng.\n"
    "- Topic: {topic_title}.\n- Description: {topic_description}.\n"
    "- Ask open questions inviting detailed, realistic answers.\n"
    "- Ask ONE question per message.\n- 1–2 short paragraphs max."
)


def build_question(topic: CorpusTopic, last_user_message: str | None = None) -> str:
    persona_prompt = topic.persona.prompt_text if topic.persona else "You are concise and curious."
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        persona_prompt=persona_prompt,
        topic_title=topic.title or "General conversation",
        topic_description=topic.description or "",
    )
    if last_user_message:
        return (
            f"{system_prompt}\n\n"
            f"Thanks for your answer. Could you share more details about {topic.title}? "
            f"What else stands out to you given what you said about '{last_user_message[:120]}'?"
        )
    return (
        f"{system_prompt}\n\n"
        f"To begin, tell me about your experiences or opinions on {topic.title or 'this topic'}."
        " What is a vivid story or detail that captures your perspective?"
    )


def start_session(db: Session, annotator: User) -> dict:
    topic = pick_next_topic(db)
    if not topic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="All topic quotas completed")

    session = CorpusSession(
        annotator_id=annotator.id,
        topic_id=topic.id,
        user_token_target=topic.target_tokens_per_session,
    )
    db.add(session)
    db.flush()

    first_question = build_question(topic)
    assistant_message = CorpusMessage(
        session_id=session.id,
        sender_role="assistant",
        lang="en",
        text=first_question,
        token_count=count_tokens(first_question),
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(session)

    return {"session": session, "first_message": assistant_message}


def add_user_reply(db: Session, session_id: int, annotator: User, payload: SessionReply) -> dict:
    session = db.query(CorpusSession).filter(CorpusSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.annotator_id != annotator.id and annotator.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    if session.status != "active":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session already finished")

    tokens = count_tokens(payload.text)
    user_message = CorpusMessage(
        session_id=session.id,
        sender_role="user",
        lang="sw_sheng",
        text=payload.text,
        token_count=tokens,
    )
    session.user_token_count += tokens
    session.turn_count += 1
    db.add(user_message)

    stopping = False
    topic = session.topic
    min_turns = topic.min_turns or 0
    max_turns = topic.max_turns or 999
    if (session.user_token_count >= session.user_token_target and session.turn_count >= min_turns) or session.turn_count >= max_turns:
        stopping = True

    assistant_message = None
    if stopping:
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        topic.total_sessions += 1
        topic.total_user_tokens_collected += session.user_token_count
    else:
        reply_text = build_question(topic, payload.text)
        assistant_message = CorpusMessage(
            session_id=session.id,
            sender_role="assistant",
            lang="en",
            text=reply_text,
            token_count=count_tokens(reply_text),
        )
        db.add(assistant_message)

    db.commit()
    db.refresh(session)

    new_session_payload = None
    if stopping:
        next_topic = pick_next_topic(db)
        if next_topic:
            new_session_payload = start_session(db, annotator)
            new_session_payload["session_finished"] = True
            new_session_payload["new_session_started"] = True
        else:
            new_session_payload = {"session_finished": True, "new_session_started": False}

    return {
        "session": session,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "session_finished": stopping,
        "new_session": new_session_payload,
    }


def get_session_detail(db: Session, session_id: int, annotator: User) -> CorpusSession:
    session = db.query(CorpusSession).filter(CorpusSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.annotator_id != annotator.id and annotator.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return session
