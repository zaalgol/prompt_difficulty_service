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

## Known issues

### scikit-learn compiler error

If installation fails on `scikit-learn` with:

```text
ERROR: Compiler cl cannot compile programs.
```

the likely cause is Python 3.14. Recreate the venv with Python 3.12 or 3.13 before changing project code.

### 504 on the spaCy model

If `pip install -r requirements.txt` fails on the `en_core_web_lg` line with:

```text
ERROR: HTTP error 504 ... Gateway Time-out ... en_core_web_lg-3.8.0-py3-none-any.whl
```

this is a transient GitHub error, **not** a broken environment. pip downloads every
wheel before installing any, so one flaky URL aborts the whole run. Before
reinstalling, check whether it actually installed earlier:

```powershell
pip show en_core_web_lg   # or: pip list | Select-String en_core
```

If it is listed, the install already succeeded and the 504 is just a redundant
re-download — nothing to fix. Otherwise retry, or install the model separately via
`python -m spacy download en_core_web_lg` (a different route than the GitHub wheel).

### `uvicorn` not recognized

`uvicorn`, `pytest`, and `python` live inside the venv. Running them without
activating it gives `'uvicorn' is not recognized ...`. Activate first
(`.\.venv\Scripts\Activate.ps1`) — the prompt shows `(.venv)` — or call the binary
directly: `.\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8081`.
