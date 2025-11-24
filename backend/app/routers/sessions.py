from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas.session import MessageOut, SessionDetail, SessionOut, SessionReply
from ..security import get_current_user, get_db
from ..services import session as session_service

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/start")
def start_session(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    payload = session_service.start_session(db, current_user)
    return {
        "session": SessionOut.from_orm(payload["session"]),
        "first_message": MessageOut.from_orm(payload["first_message"]),
    }


@router.post("/{session_id}/reply")
def reply(session_id: int, reply_in: SessionReply, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    result = session_service.add_user_reply(db, session_id, current_user, reply_in)
    response = {
        "session": SessionOut.from_orm(result["session"]),
        "user_message": MessageOut.from_orm(result["user_message"]),
        "assistant_message": MessageOut.from_orm(result["assistant_message"]) if result["assistant_message"] else None,
        "session_finished": result["session_finished"],
    }
    if result["new_session"]:
        new_session = result["new_session"]
        if new_session.get("new_session_started"):
            response["new_session"] = {
                "session": SessionOut.from_orm(new_session["session"]),
                "first_message": MessageOut.from_orm(new_session["first_message"]),
                "session_finished": True,
                "new_session_started": True,
            }
        else:
            response["new_session"] = new_session
    return response


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: int, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    session = session_service.get_session_detail(db, session_id, current_user)
    return session
