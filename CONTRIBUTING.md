# Contributing to Fewshoter

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt
```

## Development workflow

1. Create a branch from `main`
2. Make focused changes with clear commit messages
3. Run checks locally before opening a PR

## Quality checks

```bash
python -m pytest
python -m compileall fewshoter
```

## Pull requests

- Keep PRs small and reviewable
- Explain why the change is needed
- Add tests when behavior changes
- Update docs when interfaces change
