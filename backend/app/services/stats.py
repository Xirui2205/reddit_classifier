from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import CorpusMessage, CorpusSession, CorpusTopic, User
from ..schemas.stats import AdminOverview, AnnotatorStats


def annotator_stats(db: Session, annotator: User) -> AnnotatorStats:
    total_tokens = db.query(func.coalesce(func.sum(CorpusSession.user_token_count), 0)).filter(
        CorpusSession.annotator_id == annotator.id
    ).scalar() or 0

    tokens_per_topic = dict(
        db.query(CorpusTopic.title, func.coalesce(func.sum(CorpusSession.user_token_count), 0))
        .join(CorpusSession, CorpusSession.topic_id == CorpusTopic.id)
        .filter(CorpusSession.annotator_id == annotator.id)
        .group_by(CorpusTopic.title)
        .all()
    )

    tokens_per_day = dict(
        db.query(func.date(CorpusMessage.created_at), func.coalesce(func.sum(CorpusMessage.token_count), 0))
        .join(CorpusSession, CorpusSession.id == CorpusMessage.session_id)
        .filter(CorpusSession.annotator_id == annotator.id, CorpusMessage.sender_role == "user")
        .group_by(func.date(CorpusMessage.created_at))
        .all()
    )

    return AnnotatorStats(
        total_tokens=int(total_tokens),
        tokens_per_topic={k or "Unknown": int(v) for k, v in tokens_per_topic.items()},
        tokens_per_day={str(k): int(v) for k, v in tokens_per_day.items()},
    )


def admin_overview(db: Session) -> AdminOverview:
    global_tokens = db.query(func.coalesce(func.sum(CorpusSession.user_token_count), 0)).scalar() or 0

    tokens_per_topic = dict(
        db.query(CorpusTopic.title, func.coalesce(func.sum(CorpusSession.user_token_count), 0))
        .join(CorpusSession, CorpusSession.topic_id == CorpusTopic.id)
        .group_by(CorpusTopic.title)
        .all()
    )

    sessions_per_topic = dict(
        db.query(CorpusTopic.title, func.count(CorpusSession.id))
        .join(CorpusSession, CorpusSession.topic_id == CorpusTopic.id)
        .group_by(CorpusTopic.title)
        .all()
    )

    annotator_rankings = [
        {"annotator_id": row[0], "total_tokens": int(row[1])}
        for row in db.query(CorpusSession.annotator_id, func.coalesce(func.sum(CorpusSession.user_token_count), 0))
        .group_by(CorpusSession.annotator_id)
        .order_by(func.sum(CorpusSession.user_token_count).desc())
        .all()
    ]

    tokens_per_day = dict(
        db.query(func.date(CorpusMessage.created_at), func.coalesce(func.sum(CorpusMessage.token_count), 0))
        .filter(CorpusMessage.sender_role == "user")
        .group_by(func.date(CorpusMessage.created_at))
        .all()
    )

    return AdminOverview(
        global_tokens=int(global_tokens),
        sessions_per_topic={k or "Unknown": int(v) for k, v in sessions_per_topic.items()},
        tokens_per_topic={k or "Unknown": int(v) for k, v in tokens_per_topic.items()},
        annotator_rankings=annotator_rankings,
        tokens_per_day={str(k): int(v) for k, v in tokens_per_day.items()},
    )
