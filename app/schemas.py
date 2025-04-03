from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated, Optional


class QuestionRequest(BaseModel):
    text: Annotated[str, StringConstraints(max_length=256)]
    max_sources: Annotated[int, Field(strict=True, gt=0, lt=6)] = 3
    similarity_threshold: Annotated[float, Field(strict=True, ge=0.0)] = 0.7

class QuestionResponse(BaseModel):
    answer: str
    sources: Optional[list[str]] = []