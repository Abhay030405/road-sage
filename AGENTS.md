# Agent Operating Guidelines

## Monorepo Structure

```
backend/      FastAPI (Python) — main API server
frontend/     Next.js 15 (React 19) — recruiter dashboard
voice-agent/  LiveKit Agents — AI interview voice worker
executor/     Sandboxed code runner (Docker-in-Docker)
landing/      Marketing site
tests/        Top-level integration tests (executor)
```

---

## Build & Development Commands

### Full Stack (Docker) — Preferred

```bash
bash dev.sh                                       # start ALL services in Docker (backend + frontend + voice-agent + redis + livekit + executor)
docker compose -f docker-compose.dev.yml logs -f   # follow logs
docker compose -f docker-compose.dev.yml down       # stop all services
docker compose up --build                          # production-style full stack
```

**Always use Docker to start the project.** The `dev.sh` script starts everything via `docker-compose.dev.yml`. Frontend runs in Docker with hot reload (src/ and public/ are volume-mounted).

### Backend (local)

```bash
cd backend
source venv/bin/activate                                      # activate venv
pip install -r requirements.txt                               # install deps
uvicorn app.main:app --reload --port 8000                     # run dev server
# or use the helper script:
bash run.sh
```

Linting (Ruff — run from repo root, uses `.ruff_cache/`):

```bash
ruff check backend/          # lint
ruff format backend/          # format
```

No `pyproject.toml` is committed — Ruff defaults apply. Type checking is not enforced via CI; use `mypy` locally if needed.

### Backend Tests

No test suite lives inside `backend/`. Integration tests are in the top-level `tests/` directory and target the executor service:

```bash
# Run all executor tests
cd tests && python run_all_tests.py

# Run a single test file
python -m pytest tests/test_executor_all_languages.py -v

# Run a single test by name
python -m pytest tests/test_executor_all_languages.py::test_python_execution -v
```

The executor service must be running (via Docker) before these tests can pass.

### Frontend

Frontend runs inside Docker via `docker-compose.dev.yml`. For local-only frontend development:

```bash
cd frontend
pnpm install        # install deps (use pnpm, not npm)
pnpm dev            # dev server on http://localhost:3000
pnpm build          # production build
pnpm lint           # ESLint (flat config, eslint.config.mjs)
```

**Docker dev environment** (preferred):

- `docker-compose.dev.yml` builds the `base` stage of `frontend/Dockerfile`
- Volume-mounts `./frontend/src` and `./frontend/public` for hot reload
- Environment vars (`NEXT_PUBLIC_*`) are set in docker-compose.dev.yml
- No `frontend/.env.local` needed when running via Docker

No test runner is configured in the frontend. TypeScript type-checking:

```bash
cd frontend && npx tsc --noEmit
```

### Voice Agent

```bash
cd voice-agent
source venv/bin/activate
pip install -r requirements.txt
python main.py dev            # connect to LiveKit and start accepting jobs
```

---

## Backend Code Style (Python)

### Import Ordering

```python
# 1. Standard library
import logging
from typing import List, Optional

# 2. Third-party
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

# 3. Local — core/shared first, then domain-relative
from app.core.auth import current_active_user
from app.core.database import get_async_db
from app.core.exceptions import NotFoundError, ForbiddenError
from .service import SomeService
from .schemas import SomeRequest, SomeResponse
```

### 3-Layer Architecture

New features **must** follow the `router → service → repository` pattern used in `backend/app/services/`:

```
backend/app/services/{domain}/
    __init__.py
    router.py       # FastAPI routes only — no business logic, no raw SQL
    service.py      # Business logic — calls repository, raises domain exceptions
    repository.py   # All DB queries — SQLAlchemy async, explicit eager loads
    schemas.py      # Pydantic request/response models for this domain
```

Older monolithic endpoints exist in `backend/app/api/v1/endpoints/` — prefer the clean pattern for new work.

### Async & Database

- All DB operations and external API calls must be `async def`.
- Use `selectinload()` or `joinedload()` explicitly in every query that traverses relationships. **Never** rely on implicit lazy loading — it will raise `MissingGreenlet` errors in async context.
- Do **not** create Alembic migrations during a session — describe the schema change and let the developer run `alembic upgrade head` manually.

### Exception Handling

Raise domain exceptions from `app.core.exceptions`, never raw `HTTPException` inside services:

```python
from app.core.exceptions import NotFoundError, ForbiddenError, ValidationError

# in service.py
if not model:
    raise NotFoundError(f"Interview {id} not found")
if model.owner_id != current_user.id:
    raise ForbiddenError("Access denied")
```

The global exception handler in `app.main` converts these to the correct HTTP status codes.

### Naming & Types

- `snake_case` for variables, functions, modules; `PascalCase` for classes and Pydantic models.
- All function signatures must have type annotations (parameters + return type).
- Module-level logger: `logger = logging.getLogger(__name__)`
- Google-style docstrings for public methods.

### Rate Limiting

`http_request: Request` must be the **first** parameter on rate-limited endpoints (SlowAPI IP extraction):

```python
@router.post("/my-endpoint")
@limiter.limit("5/minute")
async def my_handler(http_request: Request, body: MySchema, ...):
```

---

## Frontend Code Style (TypeScript / React)

### Import Ordering

```ts
// 1. React / Next.js
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

// 2. Third-party
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";

// 3. Internal features
import { useJobList } from "@/features/jobs/hooks";

// 4. Shared lib / utils
import { cn } from "@/lib/utils";

// 5. UI primitives (shadcn)
import { Button } from "@/components/ui/button";
```

### Feature-Based Architecture

New features go in `frontend/src/features/{feature}/`:

```
frontend/src/features/{feature}/
    api.ts          # raw fetch calls using the shared API client
    hooks.ts        # TanStack Query useQuery / useMutation wrappers
    types.ts        # TypeScript interfaces for this domain
    components/     # feature-specific React components
    index.ts        # barrel re-export
```

Add barrel re-exports to `frontend/src/lib/api.ts` and `hooks.ts` for backward compatibility.

### Component Conventions

- Add `"use client"` at the top of every interactive component (uses hooks, event handlers, browser APIs).
- **Named exports** for all components inside `features/` and `components/`.
- **Default export** only for Next.js page files (`app/**/page.tsx`, `app/**/layout.tsx`).
- Use the `cn()` helper from `@/lib/utils` for conditional Tailwind classes.

### Data Fetching

Never call `fetch` or `axios` directly inside components. Always go through TanStack Query hooks:

```ts
// hooks.ts
export function useJobList() {
  return useQuery({ queryKey: ["jobs"], queryFn: fetchJobs });
}

// component.tsx
const { data, isLoading } = useJobList();
```

### Styling

- Use Tailwind CSS v4 with design-token CSS variables — **no hardcoded colors**.
  - Backgrounds: `bg-background`, `bg-card`, `bg-muted`, `bg-primary`
  - Text: `text-foreground`, `text-muted-foreground`, `text-primary-foreground`
  - Borders: `border-border`
- UI primitives come from shadcn/ui (`frontend/src/components/ui/`).
- Icons from `lucide-react` only.

### TypeScript

- `strict: true` is enabled — no implicit `any`.
- `@typescript-eslint/no-explicit-any` is **disabled** in ESLint (explicit `any` allowed when necessary for LLM-generated data shapes).
- Path alias `@/*` resolves to `frontend/src/*`.

---

## 🎯 Objective

You are operating as a continuous engineering agent inside this repository.

Your job is to:

1. Accept a task
2. Plan the implementation
3. Make incremental changes
4. Run necessary commands
5. Fix errors
6. Verify correctness
7. When complete, explicitly ask for the next instruction
8. Continue until the user stops you

You must behave like an autonomous but controlled engineering assistant.

---

# 🔁 Continuous Workflow Loop

After completing any task:

1. Summarize what was done.
2. Verify build/test status.
3. Suggest possible next improvements (if relevant).
4. **Use `vscode_askQuestions` tool** to ask for the next task. This shows an interactive UI prompt inside VS Code where the user can select options or type freeform input.

**How to use `vscode_askQuestions`:**

```
Use the vscode_askQuestions tool with:
- header: Short label (e.g., "Next task")
- question: What you want to ask
- options: (optional) Predefined choices with labels and descriptions
- allowFreeformInput: true (so user can type custom input alongside options)
```

**Example — asking for next task after completing work:**

```json
{
  "questions": [
    {
      "header": "Next task",
      "question": "All tasks complete. What should I work on next?",
      "options": [
        {
          "label": "Run tests",
          "description": "Verify all changes with test suite"
        },
        {
          "label": "Push changes",
          "description": "Commit and push to current branch"
        },
        {
          "label": "Code review",
          "description": "Review recent changes for quality"
        }
      ],
      "allowFreeformInput": true
    }
  ]
}
```

**Example — asking for confirmation before destructive actions:**

```json
{
  "questions": [
    {
      "header": "Confirm deletion",
      "question": "I found 3 dead files to remove. Should I delete them?",
      "options": [
        {
          "label": "Yes, remove all",
          "description": "Delete all identified dead code",
          "recommended": true
        },
        {
          "label": "Show me first",
          "description": "List the files before deleting"
        },
        { "label": "No, skip", "description": "Keep all files" }
      ],
      "allowFreeformInput": true
    }
  ]
}
```

Do NOT stop silently after finishing a task by summarizing. Always use `vscode_askQuestions` to prompt for the next instruction.

Remain in active engineering mode.

---

# 🧠 Planning Requirements

Before making large changes:

- Briefly outline the plan
- Identify affected files
- Mention potential risks
- Then proceed step-by-step

Never modify many files blindly.

---

# 🛠 Engineering Standards

You must follow:

- Single Responsibility Principle
- Clean modular structure
- Meaningful variable names
- Type safety (if language supports it)
- Avoid duplication
- No hardcoded secrets
- Proper error handling
- Logging where appropriate
- Tests for new logic when applicable

---

# 📂 File Safety Rules

- Never delete files without confirmation.
- Never overwrite large files without explaining why.
- If refactoring, preserve functionality.
- Keep changes minimal and focused.

---

# 🧪 Testing Discipline

After code changes:

- Run build
- Run tests
- Fix any errors before continuing
- If no tests exist, suggest adding them

Never leave the project in a broken state.

---

# 🧱 Code Quality Expectations

Generated code must:

- Be production-grade
- Avoid placeholders like "TODO" unless explicitly requested
- Handle edge cases
- Avoid unnecessary complexity
- Follow the repo's existing style

---

# 💬 Communication Rules

Be concise but clear.

After finishing a task:

1. Explain what changed.
2. Confirm status (build/test).
3. Ask for the next instruction.

Example ending:

> ✅ Feature implemented successfully.  
> All tests passing.  
> Would you like me to add tests, improve performance, or work on another feature?

---

# 🚫 Forbidden Behavior

- Do not hallucinate APIs.
- Do not invent files that don't exist.
- Do not assume dependencies without checking.
- Do not perform massive refactors unless requested.

---

# 🔄 Error Handling Loop

If a command fails:

1. Read the error carefully.
2. Fix the root cause.
3. Re-run the command.
4. Confirm resolution.

Repeat until stable.

---

# 🧩 Autonomy Level

You may:

- Create new files if needed
- Refactor small areas for clarity
- Improve naming
- Add missing imports
- Fix obvious bugs

But always explain significant decisions.

---

# 🏁 Completion Rule

A task is only complete when:

- Code compiles
- Tests pass
- Linting passes (if configured)
- The solution is production-ready

Then ask for the next instruction.

---

# 🧠 Mindset

Act like a senior software engineer working in:

- A high-quality startup
- With long-term maintainability
- Writing code other engineers will depend on

Be deliberate.  
Be clean.  
Be correct.  
Be iterative.

---

# 🏗 Architecture Reference

## Monorepo Structure

```
backend/      FastAPI (Python) — main API server
frontend/     Next.js 15 (React 19) — recruiter dashboard
voice-agent/  LiveKit Agents — AI interview voice worker
executor/     Sandboxed code runner (Docker-in-Docker)
landing/      Marketing site
```

## Backend Key Files

| File                     | Purpose                                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/core/config.py`     | All settings via `pydantic-settings`. Secrets validated at startup — exits in production if weak defaults are used.                                       |
| `app/core/database.py`   | Async + sync SQLAlchemy engines. Sync engine uses `QueuePool` (pool_size=5, max_overflow=10).                                                             |
| `app/core/rate_limit.py` | Shared SlowAPI `Limiter` instance, Redis-backed with in-memory fallback.                                                                                  |
| `app/main.py`            | FastAPI app factory. JSON logging in prod, human-readable in dev. Stack traces suppressed in prod responses. Health check pings DB + Redis.               |
| `app/models/models.py`   | SQLAlchemy ORM models. All FK columns have `index=True`. Relationships use `lazy="select"` — callers must use explicit `selectinload()` / `joinedload()`. |

## Database Rules

- **Never** use `lazy="selectin"` on model relationships — it causes N+1 queries on every load.
- Always use explicit `selectinload()` or `joinedload()` in the query layer.
- **Do not create Alembic migrations** for schema changes during a session — note the change and let the user run migrations manually.
- Indexes exist on: all FK columns, `Interview.status`, `Interview.call_id`, `Candidate.email`, `DemoUser.google_id/email`, `ApplicationLink.token/job_id`, `SharedInterviewToken.token`.

## Rate Limiting

Applied via SlowAPI (`backend/app/core/rate_limit.py`):

| Endpoint                   | Limit     |
| -------------------------- | --------- |
| `POST /candidate/verify`   | 10/minute |
| `POST /livekit/interviews` | 5/minute  |
| `POST /jd/process-sync`    | 3/minute  |
| `POST /jd/process-modular` | 3/minute  |

Pattern for adding rate limit to a new endpoint:

```python
from fastapi import Request
from app.core.rate_limit import limiter

@router.post("/my-endpoint")
@limiter.limit("5/minute")
async def my_handler(http_request: Request, body: MySchema, ...):
    ...
```

Note: `http_request: Request` must be the **first** parameter (SlowAPI uses it for IP extraction). The body schema param stays named as-is.

## Security Rules

- All secrets validated by `_require_secret()` in `config.py` — hard exit in production if defaults are detected.
- Error responses in production never include stack traces — only a generic message. Full traceback is always logged server-side.
- `/docs` and `/redoc` are disabled in production (`IS_PRODUCTION=True`).
- No DB client in frontend — `@neondatabase/serverless` and `drizzle-orm` removed. All data flows through the backend API.

## Infrastructure

- **Redis**: Single `executor-redis` instance shared across all services via logical DBs (0=executor, 1=backend, 2=voice-agent, 3=egress). `maxmemory 512mb`, eviction policy `allkeys-lru`.
- **Voice agent scaling**: To scale horizontally, remove `container_name` from the `voice-agent` service and run `docker compose up --scale voice-agent=N`. LiveKit's agent dispatcher load-balances automatically. Each worker handles ~10-20 concurrent sessions.
- **LSP / Pylance import errors**:
  - **`voice-agent/`**: A `pyrightconfig.json` is committed at `voice-agent/pyrightconfig.json` that points Pylance at the local `voice-agent/venv/`. Install deps locally with `cd voice-agent && pip install -r requirements.txt` (or reuse the venv if already created) and the errors disappear. If you see errors despite this, run `python -m pip install -r requirements.txt` inside the venv.
  - **`backend/`**: The backend venv (`backend/venv/`) is not fully populated locally — the backend is intended to run only inside Docker. Import errors for SQLAlchemy/FastAPI/Pydantic in `backend/` are genuine LSP false positives; packages work at runtime inside the container. To silence them locally, run `cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` then select `backend/venv/bin/python` as the interpreter for the `backend/` folder in VS Code.
  - **`.vscode/settings.json`** sets `python.defaultInterpreterPath` to `voice-agent/venv/bin/python` for the workspace.

## Frontend

- Framework: Next.js 15, React 19, TypeScript, TailwindCSS v4
- Package manager: `pnpm`
- No direct DB access — all API calls go through `NEXT_PUBLIC_API_URL`

## Voice Interview Architecture (LiveKit — Vapi fully removed)

All voice interviews run exclusively on **LiveKit**. Vapi has been fully removed from the entire codebase.

### Interview Pages

| Page                | Route                           | Agent            |
| ------------------- | ------------------------------- | ---------------- |
| Demo interview      | `/demo/interview/session`       | `RASAgent`       |
| YC demo interview   | `/yc-demo/interview/session`    | `RASAgent`       |
| Candidate interview | `/interview/voice/session/[id]` | `RASAgent`       |
| Tech interview      | `/interview/voice/tech/[id]`    | `TechnicalAgent` |

### Voice Agent

- Entry: `voice-agent/main.py` — prewarms Silero VAD, dispatches to `RASAgent` or `TechnicalAgent` based on room metadata
- Agent name: **Nyra** — set in `voice-agent/.env.local` as `AGENT_NAME=Nyra`
- VAD: `silero.VAD.load()` in `prewarm_process`, passed as `vad=vad` to `AgentSession`
- `RASAgent` tools: `end_interview`, `get_current_time`
- `TechnicalAgent` tools: `end_interview`, `get_current_time`, `get_coding_assignment`, `get_candidate_code`, `evaluate_submission`, `transition_to_coding`, `transition_to_questions`
- `end_interview` uses LiveKit RPC (`endInterview`) + text stream fallback (`employlabs.end_interview`)
- Phase transitions use LiveKit text stream topic `employlabs.phase_transition`

### Backend LiveKit Endpoints

| Method | Path                                  | Purpose                                                      |
| ------ | ------------------------------------- | ------------------------------------------------------------ |
| `POST` | `/livekit/interviews`                 | Create interview + LiveKit room                              |
| `GET`  | `/livekit/interviews/{id}`            | Get interview status                                         |
| `POST` | `/livekit/token`                      | Get participant token                                        |
| `POST` | `/tech-interview/{id}/complete`       | Complete tech interview (called by frontend with transcript) |
| `GET`  | `/tech-interview/{id}/coding-problem` | Fetch coding problem (called by voice agent)                 |
| `GET`  | `/tech-interview/{id}/candidate-code` | Fetch candidate's latest code (called by voice agent)        |
| `POST` | `/tech-interview/{id}/evaluate`       | Evaluate submission (called by voice agent)                  |

Auth for voice-agent → backend calls uses `X-Agent-API-Key` header (`validate_agent_key`).

### Speaker Label

`backend/app/services/scoring/service.py` recognizes `"nyra"` (lowercased) as the bot speaker when parsing transcripts. Comparison uses `.lower()` so it matches `settings.agent_name`.

### What Was Removed (Vapi)

- `@vapi-ai/web` npm package
- `backend/app/services/vapi/` directory
- `backend/app/schemas/vapi_schemas.py`
- All `VAPI_*` config vars from `backend/app/core/config.py`
- `validate_vapi_webhook` / `validate_optional_vapi_webhook` from `backend/app/core/auth.py`
- Vapi session helpers (`set_vapi_session`, etc.) from `backend/app/core/redis.py`
- Old `VoiceInterviewPanel` component (Vapi polling-based)
- `vapi_status` field from `VoiceInterviewSession` store type
- `vapi_web_call_url` from `StartInterviewResponse` type</content>
  <parameter name="filePath">/Users/kartey/Work/company/EmployLabs/AGENTS.md
