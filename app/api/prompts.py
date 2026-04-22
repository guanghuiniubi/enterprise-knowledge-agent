from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.prompts.registry import prompt_registry


router = APIRouter(prefix="/prompts", tags=["prompts"])


class ActivatePromptVersionRequest(BaseModel):
    version: str


@router.get("")
def list_prompts():
    return {
        "prompts": prompt_registry.list_prompts(),
        "active_versions": prompt_registry.active_versions(),
    }


@router.post("/reload")
def reload_prompts():
    prompt_registry.reload()
    return {
        "status": "ok",
        "active_versions": prompt_registry.active_versions(),
    }


@router.get("/{name}")
def get_prompt(name: str):
    try:
        prompt = prompt_registry.get(name=name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "name": prompt.name,
        "active_version": prompt.version,
        "template": prompt.template,
        "description": prompt.description,
        "metadata": prompt.metadata,
    }


@router.post("/{name}/activate")
def activate_prompt_version(name: str, req: ActivatePromptVersionRequest):
    try:
        prompt = prompt_registry.activate(name=name, version=req.version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "name": prompt.name,
        "active_version": prompt.version,
        "description": prompt.description,
        "metadata": prompt.metadata,
    }

