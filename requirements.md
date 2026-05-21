# Requirements

Source: `0526_Take-Home Challenge_ LLM Engineer.pdf`

## Challenge Goal

Build a provider-agnostic RAG orchestrator from scratch in a new repository.

The system should improve LLM response speed, accuracy, and cost efficiency by avoiding unnecessary retrieval, unnecessary context injection, unnecessary tool calls, and unnecessary reasoning overhead.

The goal is not to build a polished end-user application. The goal is to demonstrate strong LLM systems engineering judgment around retrieval, orchestration, provider abstraction, latency, cost, multi-turn context, and review/verification.

## Problem Context

The current system retrieves and injects external context too aggressively. Retrieval is applied to nearly every request, even when unnecessary.

The new system should reduce:

- Latency.
- Token usage.
- Response noise.
- Infrastructure cost.

The current system also has these gaps that the new architecture should address:

- No response review layer.
- Limited orchestration intelligence.
- Overly aggressive RAG injection.
- Insufficient MCP/tool orchestration abstraction.
- Heavy provider coupling.

## Response Types To Support

The system should support two major response types.

Fast responses:

- Example use cases:
  - Website chatbot.
  - Interactive product chat.
  - Real-time support interactions.
- Primary goal: extremely fast responses while maintaining strong accuracy.

Slow responses:

- Example use cases:
  - GitHub issue replies.
  - Async technical support.
  - Long-form troubleshooting.
- Primary goal: higher reasoning quality and accuracy, even if latency is slightly higher.

## Performance And Architecture Objectives

The system should improve performance across these dimensions:

- Accuracy: responses should be more reliable and contextually correct.
- Speed: interactive systems should feel fast and responsive.
- Cost efficiency: the orchestrator should reduce unnecessary retrieval, token usage, reasoning overhead, and model calls.
- Architectural flexibility: the system should not depend on one vendor or proprietary workflow, and switching providers should be straightforward.

Advanced orchestration features should be designed in a portable way, including:

- MCP usage.
- Retrieval strategies.
- Thinking/reasoning budgets.
- Structured tool usage.
- Review pipelines.

## Core Requirements

### 1. Use Retrieval Selectively

The system must not retrieve context for every request.

It should:

- Determine whether retrieval is actually necessary.
- Avoid retrieval for generic or unrelated questions.
- Retrieve only when retrieval meaningfully improves answer quality.
- Minimize unnecessary context injection.
- Demonstrate judgment around when retrieval helps, when retrieval hurts, and how much retrieval is appropriate.
- Improve significantly on the current behavior where retrieval context is always injected, even when unnecessary.

### 2. Handle Multi-Turn Conversations

The system should:

- Support follow-up questions.
- Reuse prior conversational context intelligently.
- Determine when a new retrieval is needed.
- Avoid duplicate retrieval and duplicate context injection across turns.
- Avoid repeatedly sending the same documents back to the model.
- Demonstrate thoughtful conversation-state management.

### 3. Support Differentiated Response Modes

Implement at least two orchestration paths.

Fast path:

- Optimized for low latency.
- Uses minimal retrieval.
- Uses lightweight reasoning.
- Suitable for interactive UX.
- Maintains strong accuracy while prioritizing speed.
- Example use cases include website chatbots, interactive product chat, real-time support interactions, live assistants, and quick support questions.

Deep reasoning path:

- Optimized for higher accuracy.
- Optimized for higher reasoning quality, even when latency is slightly higher.
- Uses deeper analysis.
- Supports richer retrieval and tool usage.
- Includes response review or refinement.
- Example use cases include GitHub issue responses, debugging assistance, async technical support, async support workflows, and long-form troubleshooting.

The system should clearly demonstrate:

- How routing decisions are made.
- How orchestration differs between modes.
- How reasoning budgets differ between modes.

### 4. Be Provider Agnostic

The architecture must support multiple providers cleanly and cannot only work with OpenAI.

It must support at least:

- One OpenAI-compatible provider.
- One non-OpenAI provider.

The system must be fully vendor-agnostic and work cleanly across major providers, including but not limited to:

- OpenAI.
- Anthropic.
- Gemini.
- DeepSeek.
- Kimi.

Provider switching should require minimal code or configuration changes.

Provider-specific implementations should be isolated behind a clean abstraction layer.

The implementation should demonstrate:

- Abstraction quality.
- Portability.
- Extensibility.

### 5. Improve MCP and Tool Usage

The system should demonstrate thoughtful MCP/tool orchestration patterns.

It should support or demonstrate:

- Selective tool usage.
- Dynamic tool routing.
- Structured orchestration.
- Cross-provider compatibility.
- Minimizing unnecessary tool calls.

Advanced orchestration concepts should remain portable across providers.

### 6. Retrieve From a Knowledge Source Only When Needed

The system should:

- Query a retrieval layer or RAG database only when beneficial.
- Inject minimal, focused context.
- Explain or demonstrate why specific context was selected.
- Avoid indiscriminate context stuffing.
- Prefer small, targeted context windows over dumping large irrelevant chunks into prompts.

### 7. Add Response Review and Verification

The current system has effectively no review layer. The solution should include some form of review or verification.

Acceptable approaches include:

- Answer validation.
- Self-review.
- Response critique.
- Confidence scoring.
- Verification pass.
- Lightweight evaluator.

This does not need to be overly complex. The goal is to demonstrate practical reasoning about improving answer reliability.

### 8. Clean Implementation

The repository should include:

- Open-source GitHub repository.
- Clear project structure.
- Readable code.
- Modular architecture.
- Extensible design.
- Good engineering practices.

## Deliverables

Submit a GitHub repository containing:

- Code implementing the orchestration system.
- A README with:
  - Setup instructions.
  - Architecture overview.
  - Assumptions.
  - How to run the project.
  - Provider abstraction design.
  - Orchestration design decisions.

A simple Python script that is runnable with some API keys is sufficient.

Use of AI during implementation is allowed.

## Evaluation Priorities

The submission will be judged primarily on:

- Systems thinking.
- Orchestration quality.
- Practical engineering judgment.
- Latency and cost tradeoffs.
- Retrieval quality.
- Provider abstraction design.
- Maintainability.
- Reasoning about real-world production constraints.

Thoughtful architecture and orchestration decisions matter significantly more than UI polish or framework complexity.

## Nice To Have / Bonus

### Seat-Based Tool Cost Optimization

Bonus points are available if the system can intelligently offload suitable tasks to seat-based tools such as Claude Code, Codex, or similar fixed-cost environments instead of always using metered API calls.

The system may demonstrate the ability to:

- Identify tasks better suited for seat-based tools.
- Reduce token/API spend where possible.
- Use coding agents for repo analysis, debugging, refactoring, or implementation tasks.
- Preserve provider-agnostic orchestration principles.
- Avoid unnecessary LLM API calls when a fixed-cost tool can complete the work.

The goal is not to force all work into seat-based tools. The goal is to demonstrate smart cost-aware routing between:

- API-based model calls.
- Retrieval systems.
- MCP/tools.
- Seat-based coding or reasoning environments.

Strong submissions may show how the orchestrator decides when this type of offloading is appropriate and how results are reviewed before being returned to the user.
