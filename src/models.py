from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    type: str


class ReturnSpec(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ReturnSpec] = None


class PromptInput(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
