import { useState, useCallback } from "react";
import type { Graph, AgentState, DomLevel, Plan, Stage } from "../types";

interface State {
  stage: Stage;
  problem: string;
  expectedAnswer: string;
  domLevel: DomLevel;
  plan: Plan | null;
  graph: Graph | null;
  agentStates: Record<string, AgentState>;
  finalAnswer: string | null;
  error: string | null;
  isLoading: boolean;
}

const initial: State = {
  stage: "input",
  problem: "",
  expectedAnswer: "",
  domLevel: "high",
  plan: null,
  graph: null,
  agentStates: {},
  finalAnswer: null,
  error: null,
  isLoading: false,
};

export function useOrchestration() {
  const [state, setState] = useState<State>(initial);

  const generatePlan = useCallback(async (problem: string, domLevel: DomLevel, expectedAnswer: string) => {
    setState(s => ({ ...s, problem, domLevel, expectedAnswer, isLoading: true, error: null }));

    try {
      const res = await fetch("/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ problem, dom_level: domLevel }),
      });
      if (!res.ok) throw new Error(`Plan failed: ${res.status}`);

      const plan: Plan = await res.json();
      const agentStates = Object.fromEntries(
        plan.graph.agents.map(a => [a.id, { id: a.id, status: "pending" as const }])
      );
      const finalAnswer = plan.graph.direct_solution || null;

      setState(s => ({ ...s, stage: "plan", plan, graph: plan.graph, agentStates, finalAnswer, isLoading: false }));
    } catch (err) {
      setState(s => ({ ...s, error: String(err), isLoading: false }));
    }
  }, []);

  const executePlan = useCallback(async () => {
    if (!state.plan) return;
    setState(s => ({ ...s, stage: "execute", isLoading: true, error: null }));

    try {
      const res = await fetch("/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ problem: state.problem, graph: state.plan.graph }),
      });
      if (!res.ok) throw new Error(`Execute failed: ${res.status}`);

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const data = JSON.parse(line.slice(5).trim());

          if (data.agentId && !data.output && !data.error) {
            setState(s => ({
              ...s,
              agentStates: { ...s.agentStates, [data.agentId]: { id: data.agentId, status: "running" } },
            }));
          } else if (data.agentId && data.output) {
            setState(s => ({
              ...s,
              agentStates: { ...s.agentStates, [data.agentId]: { id: data.agentId, status: "completed", output: data.output } },
            }));
          } else if (data.agentId && data.error) {
            setState(s => ({
              ...s,
              agentStates: { ...s.agentStates, [data.agentId]: { id: data.agentId, status: "failed", error: data.error } },
            }));
          } else if (data.answer) {
            setState(s => ({ ...s, stage: "result", finalAnswer: data.answer, isLoading: false }));
          } else if (data.message) {
            setState(s => ({ ...s, error: data.message }));
          }
        }
      }
    } catch (err) {
      setState(s => ({ ...s, error: String(err), isLoading: false }));
    }
  }, [state.plan, state.problem]);

  const goToStage = useCallback((stage: Stage) => setState(s => ({ ...s, stage, error: null })), []);
  const reset = useCallback(() => setState(initial), []);

  return { ...state, generatePlan, executePlan, goToStage, reset };
}
