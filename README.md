# JobPilot 🧭

[![CI](https://github.com/DimitrisNtaoultzis/jobpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/DimitrisNtaoultzis/jobpilot/actions/workflows/ci.yml)

An AI-powered job-hunting copilot. Upload your CV and a job posting, and JobPilot:

- extracts the required skills from the posting,
- classifies the role and seniority level,
- analyzes the gap between your profile and the job,
- suggests a learning plan to close that gap,
- drafts a tailored cover letter (with human approval before finalizing).

> 🚧 **Work in progress.** This project is being built step by step as a learning and portfolio project.

## Planned tech stack

| Area | Tools |
|---|---|
| Backend | Django, FastAPI, PostgreSQL + pgvector |
| AI / Agents | LangChain, LangGraph, Groq (Llama) |
| Machine Learning | scikit-learn, MLflow |
| Fine-tuning | LoRA / QLoRA (Unsloth, PEFT), Hugging Face Hub |
| Evaluation | DeepEval, promptfoo |
| DevOps | Docker, Kubernetes (kind, Helm), GitHub Actions |
| Cloud | Azure (Container Apps, PostgreSQL, Key Vault), Terraform |

## Development setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/) and Git.

```bash
git clone https://github.com/DimitrisNtaoultzis/jobpilot.git
cd jobpilot
uv sync                      # create .venv and install all dependencies (from uv.lock)
uv run pre-commit install    # enable the git hooks (once per clone)
cp .env.example .env         # then fill in the values
```

**Useful commands:**

```bash
uv run ruff check .                  # lint
uv run ruff format .                 # format
uv run pre-commit run --all-files    # run all checks manually
```

## Roadmap

- [ ] Phase 0: Project foundations (tooling, CI)
- [ ] Phase 1: Database & Django web app
- [ ] Phase 2: FastAPI AI service & LangGraph agent
- [ ] Phase 3: Classical ML (job classification)
- [ ] Phase 4: LoRA fine-tuning
- [ ] Phase 5: LLM evaluation
- [ ] Phase 6: Docker & Kubernetes
- [ ] Phase 7: CI/CD
- [ ] Phase 8: Azure deployment
- [ ] Phase 9: Documentation & demo
