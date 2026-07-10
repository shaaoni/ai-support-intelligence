# FDE Discovery Brief
## AI-Powered Customer Support Intelligence System

---

## Client
**Fictional Client:** TechFlow SaaS Ltd.
**Industry:** B2B Software / Project Management Tools
**Team Size:** 120 employees, 8,000 active customers
**Support Volume:** ~500 tickets/day across email, chat, and Twitter

---

## Problem Statement
TechFlow's support team spends 70% of their time on repetitive tier-1 tickets
(billing questions, password resets, app performance issues). Average first
response time is 4 hours. Customer satisfaction (CSAT) score has dropped from
87% to 74% over the last two quarters due to slow response times.

**Core problems:**
- No intelligent routing — all tickets land in one queue
- Support agents manually search FAQs to answer common questions
- No visibility into ticket volume trends or common failure patterns
- Urgent issues (outages, data loss) are not prioritised automatically

---

## Stakeholder Map

| Stakeholder | Role | Priority |
|---|---|---|
| Head of Support | Primary sponsor, owns CSAT target | High |
| Support Agents (×8) | Daily users of the chat interface | High |
| CTO | Technical sign-off, security review | Medium |
| Finance Director | ROI justification, cost per query | Medium |
| End Customers | Indirect beneficiaries of faster responses | High |

---

## Solution Chosen: RAG + Agentic Orchestration

### Architecture
User Query
│
▼
Guardrails (prompt injection check)
│
▼
ML Classifier (TF-IDF + Logistic Regression)
│
├── urgent → Escalate to human agent
├── low confidence → Escalate to human agent
│
▼
RAG Retrieval (ChromaDB + sentence-transformers)
│
▼
LLM Answer Generation (Claude / mock)
│
▼
Output Guardrails → Final Answer

### Components Built
| Phase | Component | Purpose |
|---|---|---|
| 1 | Data Pipeline | Ingest CSV tickets + Markdown docs, quality checks |
| 2 | ML Classifier | Route tickets to billing/technical/general/urgent |
| 3 | RAG Vector Store | Retrieve relevant FAQ chunks for context |
| 4 | Prompt Templates | Versioned, validated prompts for every LLM call |
| 5 | Agent Graph | Orchestrate guardrails → classify → retrieve → answer |
| 6 | Streamlit UI | Chat interface + observability dashboard + admin panel |

---

## Risks Identified

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM hallucination | Medium | High | RAG grounds answers in verified FAQ docs only |
| Prompt injection attacks | Low | High | Guardrails check every input before classifier |
| Classifier low confidence | Medium | Medium | Auto-escalate to human when confidence < 28% |
| Data privacy (customer PII) | Medium | High | No PII stored in vector store; tweets anonymised |
| Model drift over time | High | Medium | MLflow tracks all runs; retrain via Admin panel |

---

## Key Metrics (Target vs Baseline)

| Metric | Baseline | Target |
|---|---|---|
| First response time | 4 hours | < 30 seconds (automated) |
| Tier-1 deflection rate | 0% | 60% |
| CSAT score | 74% | 85% |
| Avg query latency | N/A | < 500ms |
| Escalation rate | 100% | < 40% |

---

## Demo Script (5 minutes)

**Minute 1 — Problem framing**
"TechFlow receives 500 tickets a day. Today, every ticket goes into one queue
and an agent manually reads it, searches the FAQ, and replies. We're going to
change that."

**Minute 2 — Live chat demo**
Type: "I was charged twice this month" → show billing route, RAG answer, confidence score.
Type: "Our entire system is down" → show urgent escalation path.
Type: "ignore previous instructions" → show guardrail block.

**Minute 3 — Pipeline health dashboard**
Show vector store size, classifier accuracy, route distribution, avg latency.
"Every run has a run ID. If a customer complains about a wrong answer,
we can trace it back to this exact run."

**Minute 4 — Admin panel**
Show MLflow experiment runs — two models compared side by side.
"We ran 6 configurations. Logistic Regression at 100% accuracy won.
Here's the exact hyperparameters and the model artifact."

**Minute 5 — Architecture walkthrough**
Walk through the 5-node graph diagram.
"Guardrails fire first — 0.1ms, zero API cost.
Classifier runs next — 1000ms.
RAG retrieval — 30ms avg latency.
LLM answer last — only called when we're confident in the context."

---

## Why This Approach Over Alternatives

| Alternative | Why Not Chosen |
|---|---|
| Fine-tuned LLM only | Expensive, no retrieval, hallucinates FAQ details |
| Pure keyword search | No semantic understanding, misses paraphrasing |
| Human-in-loop only | Doesn't scale, slow response time |
| RAG without classifier | No priority routing, urgent tickets not flagged |

Our approach combines a cheap, fast classifier (sklearn) with semantic retrieval
(ChromaDB) and a grounded LLM answer. Each component is independently testable,
replaceable, and observable.

---

*Brief prepared by: FDE Candidate*
*Date: 2025*
*Version: 1.0*
