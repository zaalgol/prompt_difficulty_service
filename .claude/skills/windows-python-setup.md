# Skill: Windows Python Setup

Use this when helping set up or debug the local Windows environment.

## Preferred Python

Use Python 3.12 or 3.13.

Do not use Python 3.14 with the current locked dependencies unless explicitly requested.

## Setup commands

PowerShell:

```powershell
py -0p
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.12 is unavailable, use:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Known issue

If installation fails on `scikit-learn` with:

```text
ERROR: Compiler cl cannot compile programs.
```

the likely cause is Python 3.14. Recreate the venv with Python 3.12 or 3.13 before changing project code.
