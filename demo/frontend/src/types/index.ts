export type AgentType = "CoTAgent" | "SCAgent" | "DebateAgent" | "ReflexionAgent" | "WebSearchAgent";
export type AgentStatus = "pending" | "running" | "completed" | "failed";
export type DomLevel = "low" | "high";
export type Stage = "input" | "plan" | "execute" | "result";

export interface Agent {
  id: string;
  type: AgentType;
  description: string;
  input: string;
  depends_on: string[];
}

export interface Edge {
  source: string;
  target: string;
}

export interface Graph {
  agents: Agent[];
  edges: Edge[];
  answer_agent: string;
  direct_solution?: string | null;
}

export interface AgentState {
  id: string;
  status: AgentStatus;
  output?: string;
  error?: string;
}

export interface Plan {
  xml: string;
  graph: Graph;
  thinking: string | null;
}
