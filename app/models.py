from pydantic import BaseModel, Field
from typing import Literal, List, Union, Optional


class Input(BaseModel):
    id: str
    type: Literal["input"]
    content: str


class Output(BaseModel):
    id: str
    type: Literal["output"]
    content: str


class Regular(BaseModel):
    id: str
    type: Literal["regular"]
    content: int
    rules: List[str]


class Synapse(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    weight: float


class SNPSystem(BaseModel):
    neurons: List[Union[Input, Output, Regular]]
    synapses: List[Synapse]
    expected: Optional[List[object]] = None
