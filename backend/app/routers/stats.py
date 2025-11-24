from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas.stats import AdminOverview, AnnotatorStats
from ..security import get_current_user, get_db, require_admin
from ..services import stats as stats_service

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/me", response_model=AnnotatorStats)
def my_stats(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    return stats_service.annotator_stats(db, current_user)


@router.get("/admin/overview", response_model=AdminOverview)
def admin_overview(admin=Depends(require_admin), db: Session = Depends(get_db)):
    return stats_service.admin_overview(db)
