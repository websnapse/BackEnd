from pydantic import BaseModel, Field, AliasChoices
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
    from_: str = Field(..., alias=AliasChoices('from', 'from_'))
    to: str
    weight: float


class SNPSystem(BaseModel):
    neurons: List[Union[Input, Output, Regular]]
    synapses: List[Synapse]
    expected: Optional[List[object]] = None
    rule_dict: Optional[List[str]] = []
