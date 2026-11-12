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

**Prerequisites:** [uv](https://docs.astral.sh/uv/), Git and [Docker](https://docs.docker.com/get-docker/) (with Docker Compose).

```bash
git clone https://github.com/DimitrisNtaoultzis/jobpilot.git
cd jobpilot
uv sync                      # create .venv and install all dependencies (from uv.lock)
uv run pre-commit install    # enable the git hooks (once per clone)
cp .env.example .env         # then fill in the values
docker compose up -d         # start PostgreSQL + pgvector
uv run python manage.py migrate    # create the database tables
```

**Useful commands:**

```bash
uv run python manage.py runserver    # start the dev server at http://127.0.0.1:8000/
uv run ruff check .                  # lint
uv run ruff format .                 # format
uv run pre-commit run --all-files    # run all checks manually
docker compose ps                    # show the status of the database container
docker compose stop                  # stop the database (data is kept)
docker compose down                  # remove the container (data is kept in the volume)
```

## Roadmap

- [x] Phase 0: Project foundations (tooling, CI)
- [ ] Phase 1: Database & Django web app
- [ ] Phase 2: FastAPI AI service & LangGraph agent
- [ ] Phase 3: Classical ML (job classification)
- [ ] Phase 4: LoRA fine-tuning
- [ ] Phase 5: LLM evaluation
- [ ] Phase 6: Docker & Kubernetes
- [ ] Phase 7: CI/CD
- [ ] Phase 8: Azure deployment
- [ ] Phase 9: Documentation & demo
