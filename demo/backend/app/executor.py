"""Agent executor - runs individual agents via OpenAI API."""
import os
from openai import AsyncOpenAI
from .models import Agent, AgentType

client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global client
    if client is None:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


PROMPTS = {
    AgentType.COT: "Think through this problem step by step, showing your reasoning:\n\n{problem}\n{context}",
    AgentType.SC: "Consider multiple reasoning paths and find the most consistent answer:\n\n{problem}\n{context}",
    AgentType.DEBATE: "Present arguments for different viewpoints, then synthesize:\n\n{problem}\n{context}",
    AgentType.REFLEXION: "Analyze, critique, and refine into a final answer:\n\n{problem}\n{context}",
    AgentType.WEBSEARCH: "Identify what information would help and simulate relevant search findings:\n\n{problem}\n{context}",
}


async def execute_agent(agent: Agent, problem: str, ctx: dict[str, str]) -> str:
    # Build context from dependencies
    context = ""
    if agent.depends_on:
        parts = [f"[{d}]: {ctx[d]}" for d in agent.depends_on if d in ctx]
        if parts:
            context = "Previous outputs:\n" + "\n".join(parts)

    prompt = PROMPTS.get(agent.type, PROMPTS[AgentType.COT]).format(problem=problem, context=context)

    try:
        res = await get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are {agent.description}"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        return res.choices[0].message.content or "No response"
    except Exception as e:
        raise RuntimeError(f"Agent failed: {e}")
