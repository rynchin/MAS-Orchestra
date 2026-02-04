"""Metaagent interface - calls the orchestrator LLM."""
import asyncio
from gradio_client import Client
from .models import DomLevel

HF_SPACE = "rychin/MAS-Orchestra"

# System prompts (from mas_r1_reasoner)
SYSTEM_LOW = """You are a helpful assistant.

MASness (How much Multi-Agent System-ness): Minimal

Valid Channels: thinking, agent, answer

An agent is a pre-configured AI personalities that can delegate tasks to. Each subagent:
1. Has a specific purpose and expertise area
2. Uses its own context window separate from the main conversation
3. (Optional) Can be configured with specific tools it's allowed to use
4. Includes a custom system prompt that guides its behavior

An agent should be defined in channel <agent>. Each agent must contain <agent_name>, <agent_description>, <required_arguments>, <agent_output_id>

DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!"""

SYSTEM_HIGH = """You are a helpful assistant.

MASness (How much Multi-Agent System-ness): Medium

Valid Channels: thinking, agent, edge

An agent is a pre-configured AI personalities that can delegate tasks to. Each subagent:
1. Has a specific purpose and expertise area
2. Uses its own context window separate from the main conversation
3. (Optional) Can be configured with specific tools it's allowed to use
4. Includes a custom system prompt that guides its behavior

An agent should be defined in channel <agent>. Each agent must contain <agent_id>, <agent_name>, <agent_description>, <required_arguments>. To connect multiple agents to form a multi-agent system, use <edge> channel.

DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!"""

USER_LOW = """Please solve the question step by step and create agents to delegate the task when necessary. First decide whether to solve directly or to delegate to exactly one agent. If you delegate, you must define that agent by outputting <agent> with agent_name (select one of the agents: CoTAgent, SCAgent, DebateAgent, ReflexionAgent), agent_description, required_arguments, and agent_output_id. 
The final output of the delegated agent (identified by its agent_output_id) must represent the complete and final answer to the original question, not an intermediate result.
Always put either the final value or the agent_output_id in the <answer> tag, and use EXACTLY the same field names defined for the agent. If the selected agent uses roles (e.g., DebateAgent), also output debate_roles.

For example,

If you decide to solve the entire task yourself, you will output the following:

Question: What is (20+9)*(30+7)?

 <thinking>
   The problem only requires basic arithmetic.
    No specialized reasoning agent or multi-agent discussion is needed.
    (20+9)*(30+7) = 600 + 140 + 270 + 63 = 1073.
  </thinking>
  <answer>1073</answer>

If you decide to solve the entire task via delegation, you will output the following. In this case, since the agent_input is the same as the original task, you must set the agent_input as empty (""), and the parser will replace it with the original question.

Question: Compute the definite integral of (2x + 5) dx from 0 to 3.

  <thinking>
    This problem involves symbolic integration and applying the Fundamental Theorem of Calculus.
    It requires structured reasoning rather than simple numeric computation.
    I will use a calculus agent that can perform step-by-step Chain-of-Thought reasoning.
    The final answer to the original question will be the output of the CoTAgent.
  </thinking>
  <agent>
    <agent_name>CoTAgent</agent_name>
    <agent_description>Definite integrals with one Chain-of-Thought call.</agent_description>
      <required_arguments>
        <agent_input></agent_input>
      </required_arguments>
      <agent_output_id>
      calc_agent_output
      </agent_output_id>
  </agent>
  <answer>calc_agent_output</answer>

Below is the question to solve:

[QUESTION]"""

USER_HIGH = """Please solve the given question by creating one or more agents and connecting them into a valid computational graph that collaboratively produces the final answer. To create an agent, you must define that agent by outputting <agent> with agent_id (a unique id for the agent), agent_name (exactly one of: CoTAgent, SCAgent, DebateAgent, ReflexionAgent and WebSearchAgent), agent_description, required_arguments (must include at least one <agent_input> tag. DebateAgents must define <debate_roles> with two or more roles. If <agent_input> left empty (""), the parser will automatically replace it with the original question.). 

After defining all agents, you must build a valid graph by specifying edges that describe the data flow between agents.

Thinking Section (Required):

Before defining agents and edges, you must include a <thinking> section.
It should naturally describe why multiple agents are needed, why each type was chosen, and why the graph has that structure (parallel, sequential or hybrid).

Below is the question to solve:

[QUESTION]"""


def build_prompt(problem: str, dom: DomLevel) -> str:
    system = SYSTEM_LOW if dom == DomLevel.LOW else SYSTEM_HIGH
    user = (USER_LOW if dom == DomLevel.LOW else USER_HIGH).replace("[QUESTION]", problem)
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def _call_hf(prompt: str, model: str) -> str:
    return Client(HF_SPACE).predict(prompt=prompt, model_name=model, temperature=0.7, api_name="/generate")


async def call_metaagent(problem: str, dom: DomLevel) -> str:
    prompt = build_prompt(problem, dom)
    model = "Harmony GRPO-7B (step 180)" if dom == DomLevel.LOW else "Harmony Medium GRPO-7B — HotpotQA (step 250)"
    
    try:
        return await asyncio.to_thread(_call_hf, prompt, model)
    except Exception as e:
        print(f"HF Space error: {e}, using mock")
        return MOCK_LOW if dom == DomLevel.LOW else MOCK_HIGH


MOCK_LOW = """<thinking>
This is a straightforward reasoning task. A Chain-of-Thought agent will work well.
</thinking>
<agent>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Reasons through the problem step by step</agent_description>
  <required_arguments><agent_input></agent_input></required_arguments>
  <agent_output_id>reasoning_agent</agent_output_id>
</agent>
<answer>reasoning_agent</answer>"""

MOCK_HIGH = """<thinking>
This problem requires multiple perspectives. I'll use a search agent to gather information,
then have two reasoning agents analyze from different angles, and finally synthesize.
</thinking>
<agent>
  <agent_id>search</agent_id>
  <agent_name>WebSearchAgent</agent_name>
  <agent_description>Searches for relevant information</agent_description>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>
<agent>
  <agent_id>analyzer_1</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Analyzes from a technical perspective</agent_description>
  <required_arguments><agent_input>${search}</agent_input></required_arguments>
</agent>
<agent>
  <agent_id>analyzer_2</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Analyzes from a practical perspective</agent_description>
  <required_arguments><agent_input>${search}</agent_input></required_arguments>
</agent>
<agent>
  <agent_id>synthesizer</agent_id>
  <agent_name>ReflexionAgent</agent_name>
  <agent_description>Synthesizes insights into final answer</agent_description>
  <required_arguments><agent_input>${analyzer_1} ${analyzer_2}</agent_input></required_arguments>
</agent>
<answer>synthesizer</answer>"""
