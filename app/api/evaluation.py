from fastapi import APIRouter

from app.schemas.evaluation import EvaluationRunRequest, EvaluationRunResponse
from app.services.evaluation_service import evaluation_service


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationRunResponse)
def run_evaluation(req: EvaluationRunRequest):
    return evaluation_service.run_cases(req.cases)

