"""FastAPI backend for MAS-Orchestra demo."""
import asyncio
import json
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from .models import RunRequest, Graph, DomLevel
from .parser import parse, topo_sort
from .metaagent import call_metaagent
from .executor import execute_agent

load_dotenv()

app = FastAPI(title="MAS-Orchestra Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class PlanRequest(BaseModel):
    problem: str
    dom_level: DomLevel


class PlanResponse(BaseModel):
    xml: str
    graph: dict
    thinking: str | None = None


class ExecuteRequest(BaseModel):
    problem: str
    graph: dict


def sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


@app.post("/plan")
async def plan(req: PlanRequest) -> PlanResponse:
    xml = await call_metaagent(req.problem, req.dom_level)
    match = re.search(r"<thinking>(.*?)</thinking>", xml, re.DOTALL | re.IGNORECASE)
    thinking = match.group(1).strip() if match else None
    graph = parse(xml, req.dom_level.value)
    return PlanResponse(xml=xml, graph=graph.model_dump(), thinking=thinking)


async def run_execution(problem: str, graph_dict: dict):
    graph = Graph(**graph_dict)
    yield sse("graph", graph.model_dump())

    order = topo_sort(graph)
    agents = {a.id: a for a in graph.agents}
    ctx: dict[str, str] = {}
    remaining = set(order)

    while remaining:
        ready = [aid for aid in remaining if all(d in ctx for d in agents[aid].depends_on)]
        if not ready:
            yield sse("error", {"message": "Cycle detected"})
            return

        for aid in ready:
            yield sse("agent_start", {"agentId": aid})

        async def run(aid: str):
            try:
                return aid, await execute_agent(agents[aid], problem, ctx), None
            except Exception as e:
                return aid, None, str(e)

        results = await asyncio.gather(*[run(aid) for aid in ready])

        for aid, output, err in results:
            if err:
                yield sse("agent_error", {"agentId": aid, "error": err})
                return
            ctx[aid] = output
            yield sse("agent_complete", {"agentId": aid, "output": output})
            remaining.remove(aid)

    yield sse("final_answer", {"answer": ctx.get(graph.answer_agent, "No answer")})


@app.post("/execute")
async def execute(req: ExecuteRequest):
    return EventSourceResponse(run_execution(req.problem, req.graph))


@app.get("/health")
async def health():
    return {"status": "ok"}
