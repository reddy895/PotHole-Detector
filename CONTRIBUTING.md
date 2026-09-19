# Contributing to AI Pothole Detection System

Thank you for taking the time to contribute! This document covers everything you need to get the project running locally, write good code, and submit quality pull requests.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Project Architecture](#project-architecture)
4. [Running the Application](#running-the-application)
5. [Code Style](#code-style)
6. [Branch Naming Conventions](#branch-naming-conventions)
7. [Commit Message Format](#commit-message-format)
8. [Pull Request Process](#pull-request-process)
9. [Reporting Issues](#reporting-issues)

---

## Prerequisites

| Requirement | Minimum Version |
|---|---|
| Python | 3.10+ |
| pip | 23+ |
| git | 2.x |
| CUDA (optional) | 11.8+ (for GPU acceleration) |

---

## Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/reddy895/PotHole-Detector.git
cd PotHole-Detector

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (GPU only) Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Place model weights
# Download your trained pothole.pt and put it at:
#   models/pothole.pt
# Or run the weight downloader:
python utils/download_weights.py
```

---

## Project Architecture

```
PotHole/
├── config.py          # All system parameters (thresholds, paths, device)
├── detector.py        # YOLO inference engine + OpenCV annotation (core logic)
├── main.py            # CLI entrypoint (webcam / image / video modes)
├── app.py             # Flask web server (MJPEG stream + REST API)
│
├── utils/
│   ├── video_utils.py     # Stream I/O, terminal output, JSONL log writer
│   ├── helpers.py         # Hardware info, base64 image encode/decode
│   ├── logger.py          # Colour-coded structured logger
│   └── download_weights.py # Auto-download model weights
│
├── templates/
│   └── index.html     # Web UI (served by Flask)
│
├── models/            # YOLO .pt weight files (not tracked in git)
└── outputs/           # Annotated images/videos (generated at runtime)
```

**Key design rules:**
- `detector.py` has **no** knowledge of Flask, CLI args, or file I/O. It only takes a numpy frame and returns a `DetectionResult`.
- `config.py` is the **single source of truth** for all defaults. Never hardcode paths or thresholds in other modules.
- `app.py` and `main.py` are thin dispatchers — they validate inputs and call the core engine.

---

## Running the Application

```bash
# CLI — Image
python main.py --source image --input road.jpg

# CLI — Video with detection log + frame-skip optimisation
python main.py --source video --input road.mp4 --save-log --skip-frames 1

# CLI — Webcam
python main.py --source webcam

# Web UI
bash run_web.sh
# or: python app.py
# then open http://localhost:5000

# Query the REST API
curl http://localhost:5000/api/system
curl -X POST http://localhost:5000/api/conf \
     -H "Content-Type: application/json" \
     -d '{"threshold": 0.45}'
```

---

## Code Style

This project uses **[ruff](https://github.com/astral-sh/ruff)** for linting.

```bash
pip install ruff
ruff check . --select E,W,F,I --ignore E501
```

Rules enforced by CI:
- `E` / `W` — pycodestyle errors and warnings
- `F` — pyflakes (undefined names, unused imports)
- `I` — isort (import ordering)
- `E501` is **ignored** — no enforced line length

Additional conventions:
- Type-hint all function signatures.
- Use `%-style` formatting in `logging.*` calls, not f-strings.
- Docstrings follow Google style (Args / Returns / Raises sections).
- No bare `except:` — always catch a specific exception type.

---

## Branch Naming Conventions

| Prefix | Use for |
|---|---|
| `feat/` | New features (`feat/gps-tagging`) |
| `fix/` | Bug fixes (`fix/fps-jitter`) |
| `refactor/` | Internal restructuring without behaviour change |
| `docs/` | Documentation only |
| `ci/` | CI/CD pipeline changes |
| `chore/` | Dependency bumps, tooling config |

Branch names should be lowercase with hyphens. Example: `feat/add-severity-filter`.

---

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short description>

<optional body — explain WHY, not what>

<optional footer: breaking changes, issue refs>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`

Examples:
```
feat(detector): add EMA FPS smoothing (alpha=0.10)

Raw per-frame FPS is jittery due to CUDA scheduling variance.
EMA keeps the HUD readout stable while reacting to genuine throughput changes.
```

```
fix(download_weights): repair broken config.settings import

config.settings does not exist in this codebase; the module was importing
from a non-existent sub-module. Fixed to use the real config.config API.
```

---

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your changes with clean, atomic commits.
3. Ensure `ruff check` passes locally before pushing.
4. Open a PR against `main` — the CI pipeline must be green.
5. Fill in the PR template: describe the change, how to test it, and any screenshots.
6. Request a review. PRs are merged by squash-merge to keep the main history clean.

---

## Reporting Issues

When filing a bug, please include:
- OS and Python version (`python3 --version`)
- GPU model and CUDA version (if applicable)
- Full traceback / error output
- Steps to reproduce

Feature requests: open an issue with the label `enhancement` and describe the use case.
