import os
import re
from openai import AsyncOpenAI
from .models import Agent, AgentType

client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global client
    if client is None:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


SYSTEM_PROMPTS = {
    AgentType.COT: "You are a precise reasoning assistant. Think step by step, then give a clear final answer.",
    AgentType.SC: "You are a careful reasoning assistant. Consider multiple approaches and converge on the most consistent answer.",
    AgentType.DEBATE: "You are a balanced analytical assistant. Present arguments from multiple viewpoints, then synthesize a justified conclusion.",
    AgentType.REFLEXION: "You are a self-improving reasoning assistant. Analyze your answer critically and refine it until you are confident.",
    AgentType.WEBSEARCH: "You are a research assistant. Identify key facts needed, simulate finding them from authoritative sources, and summarize with citations.",
}


def resolve_input(agent_input: str, problem: str, ctx: dict[str, str]) -> str:
    if not agent_input.strip():
        return problem
    return re.sub(r"\$\{(\w+)\}", lambda m: ctx.get(m.group(1), f"[output of {m.group(1)} not available]"), agent_input)


async def execute_agent(agent: Agent, problem: str, ctx: dict[str, str], model: str = "gpt-4.1-mini") -> str:
    task = resolve_input(agent.input, problem, ctx)
    system = SYSTEM_PROMPTS.get(agent.type, SYSTEM_PROMPTS[AgentType.COT])
    is_reasoning = model.startswith("o")

    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": f"{system}\n\nRole: {agent.description}"},
            {"role": "user", "content": f"Original question: {problem}\n\nYour task: {task}"},
        ],
        max_completion_tokens=16000 if is_reasoning else 1024,
    )
    if not is_reasoning:
        kwargs["temperature"] = 0.7

    try:
        res = await get_client().chat.completions.create(**kwargs)
        content = res.choices[0].message.content
        if not content:
            print(f"Agent {agent.id}: empty content from {model}, finish_reason={res.choices[0].finish_reason}")
            return f"[{agent.id} returned empty response]"
        return content
    except Exception as e:
        raise RuntimeError(f"Agent {agent.id} failed: {e}")
