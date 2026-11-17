# 🧪 AI Lab

Twenty projects focused on building **AI agents, developer tools, and production-grade agent infrastructure**, organized by technical dependencies and increasing complexity.

> Each project lives in its own subfolder with its own `pyproject.toml` (or `go.mod` for Go projects). Folders are created when the project starts.
> Folder names have no number prefix. The implementation order is tracked here.

---

## 🗺️ Roadmap

### Phase 1 — Foundations

| Project | Folder | Status |
|---|---|---|
| LLM CLI | [`llm-cli/`](llm-cli/) | ✅ Done |
| Strict Pydantic Tool Router | [`tool-router/`](tool-router/) | ✅ Done |
| Local Agent Runtime + Self-Healing ReAct | [`local-agent-runtime/`](local-agent-runtime/) | 🚧 In progress (R0–R2 done) |
| Dynamic Context Budget Assembler | `context-budget-assembler/` | ⬜ Not started |
| Memory & Context Engine | `memory-context-engine/` | ⬜ Not started |

### Phase 2 — Developer Tools

| Project | Folder | Status |
|---|---|---|
| CLI Coding Agent | `cli-coding-agent/` | ⬜ Not started |
| Codebase Intelligence CLI | `codebase-intelligence/` | ⬜ Not started |
| MCP Server & Tool Gateway | `mcp-tool-gateway/` | ⬜ Not started |
| Engineering Workflow Copilot | `workflow-copilot/` | ⬜ Not started |

### Phase 3 — Reliability

| Project | Folder | Status |
|---|---|---|
| Agent Observability & Replay Engine | `agent-observability/` | ⬜ Not started |
| Agent Evaluation Framework + CI Gate | `agent-evals/` | ⬜ Not started |
| Sandboxed Agent Execution | `sandboxed-execution/` | ⬜ Not started |
| Human-in-the-Loop Workflow Engine | `human-in-the-loop/` | ⬜ Not started |

### Phase 4 — Distributed Agents

| Project | Folder | Status |
|---|---|---|
| Async Agent Orchestrator | `async-orchestrator/` | ⬜ Not started |
| Typed Multi-Agent Workflow | `multi-agent-workflow/` | ⬜ Not started |
| Agent Security Gateway | `agent-security-gateway/` | ⬜ Not started |
| Cost & Resource Budget Engine | `cost-budget-engine/` | ⬜ Not started |

### Phase 5 — Platform

| Project | Folder | Status |
|---|---|---|
| AI Inference & Agent Control Plane | `inference-control-plane/` | ⬜ Not started |
| Shadow Execution & Fallback Runtime | `shadow-fallback-runtime/` | ⬜ Not started |
| API-to-Browser Fallback Agent | `browser-fallback-agent/` | ⬜ Not started |

---

## 🎯 Phase Objectives

### Phase 1 — Foundations
Build reusable LLM interfaces, typed tool execution, agent loops, context management, and memory.

**Deliverable:** A functional agent runtime capable of calling models and tools, maintaining context, validating outputs, and recovering from bounded failures.

### Phase 2 — Developer Tools
Build developer-facing applications using the foundational components.

**Deliverable:** A coding agent capable of understanding repositories, modifying files, running tests safely, and integrating with engineering workflows.

### Phase 3 — Reliability
Introduce observability, evaluation, isolation, and human approval.

**Deliverable:** An agent execution environment with measurable performance, reproducible debugging, controlled execution, and approval checkpoints.

### Phase 4 — Distributed Agents
Build concurrent, durable, multi-agent workflows with security and resource controls.

**Deliverable:** A distributed agent system supporting concurrent tasks, durable state, recovery, typed coordination, and policy enforcement.

### Phase 5 — Platform
Build production-oriented infrastructure for running, routing, monitoring, and managing agents.

**Deliverable:** A multi-tenant agent platform with inference routing, quotas, shadow execution, fallback strategies, and operational controls.

---

## 🔭 Current Focus: Project 1 — LLM CLI

| | |
|---|---|
| **Folder** | [`llm-cli/`](llm-cli/) |
| **Language** | Python 3.12+ |
| **Status** | ✅ Done (Ollama provider); see [`PLAN.md`](llm-cli/PLAN.md) |
