# MAS-Orchestra Demo

Interactive demo for Multi-Agent System orchestration visualization.

## Project Structure

```
demo/
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI endpoints
│   │   ├── models.py      # Pydantic data models
│   │   ├── parser.py      # XML parsing (minimal & medium)
│   │   ├── metaagent.py   # Metaagent interface (mock for now)
│   │   └── executor.py    # Agent execution via OpenAI
│   └── requirements.txt
└── frontend/
    └── src/
        ├── components/
        │   ├── ProblemInput.tsx   # Problem input form
        │   ├── GraphViewer.tsx    # React Flow graph visualization
        │   ├── AgentOutputs.tsx   # Agent output display
        │   └── FinalAnswer.tsx    # Final answer display
        ├── hooks/
        │   └── useOrchestration.ts  # SSE handling hook
        └── types/
            └── index.ts           # TypeScript types
```

## Quick Start

### Backend

```bash
cd demo/backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Set your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd demo/frontend
npm install
npm run dev
```

Open http://localhost:3000

## API

### POST /run

Submit a problem and stream execution events via SSE.

**Request:**
```json
{
  "problem": "What is the capital of France?",
  "dom_level": "high"
}
```

**SSE Events:**
- `graph` - Agent graph structure
- `agent_start` - Agent started executing
- `agent_complete` - Agent finished with output
- `agent_error` - Agent failed
- `final_answer` - Final result

## TODO

- [ ] Connect real metaagent (HF backend)
- [ ] Add more agent types
- [ ] Improve graph layout algorithm
- [ ] Add execution history
