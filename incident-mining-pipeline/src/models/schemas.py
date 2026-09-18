"""Định nghĩa Pydantic JSON Schemas cho cây tri thức sự cố."""

from typing import Literal

from pydantic import BaseModel

from src.models.taxonomy import ENTITY_TYPES, RELATION_TYPES


class Entity(BaseModel):
    id: str
    type: Literal[tuple(ENTITY_TYPES)]
    text: str
    confidence: float


class Relation(BaseModel):
    source_id: str
    target_id: str
    relation_type: Literal[tuple(RELATION_TYPES)]
    confidence: float
    evidence: str = ""


class IncidentGraph(BaseModel):
    incident_id: str
    entities: list[Entity]
    relations: list[Relation]
