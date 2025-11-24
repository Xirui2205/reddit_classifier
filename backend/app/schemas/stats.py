from pydantic import BaseModel


class AnnotatorStats(BaseModel):
    total_tokens: int
    tokens_per_topic: dict[str, int]
    tokens_per_day: dict[str, int]


class AdminOverview(BaseModel):
    global_tokens: int
    sessions_per_topic: dict[str, int]
    tokens_per_topic: dict[str, int]
    annotator_rankings: list[dict[str, int | str]]
    tokens_per_day: dict[str, int]
