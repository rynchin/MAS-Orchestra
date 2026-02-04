from pydantic import BaseModel
from enum import Enum


class DomLevel(str, Enum):
    LOW = "low"
    HIGH = "high"


class AgentType(str, Enum):
    COT = "CoTAgent"
    SC = "SCAgent"
    DEBATE = "DebateAgent"
    REFLEXION = "ReflexionAgent"
    WEBSEARCH = "WebSearchAgent"


class Agent(BaseModel):
    id: str
    type: AgentType
    description: str
    input: str
    depends_on: list[str]


class Edge(BaseModel):
    source: str
    target: str


class Graph(BaseModel):
    agents: list[Agent]
    edges: list[Edge]
    answer_agent: str
    direct_solution: str | None = None


class RunRequest(BaseModel):
    problem: str
    dom_level: DomLevel = DomLevel.HIGH
