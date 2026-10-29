# Contributing to ScanTwin

Thanks for your interest in ScanTwin, a NEURA MED AI project.

## Ground rules
- ScanTwin is a **research and educational prototype**. Do not add features that present results as clinical advice, diagnosis, or patient-specific dosimetry.
- **Never commit patient data**, private images, or secrets. Use public, de-identified, licensed data only (see `data/README.md`).

## Getting started
```
git clone https://github.com/NeuraMed-AI-Works/scantwin.git
cd scantwin
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

## Workflow
1. Open an issue to discuss larger changes.
2. Create a branch: `git checkout -b feature/short-description`.
3. Keep changes focused and add or update tests in `tests/`.
4. Run `pytest` before pushing.
5. Open a pull request using the template.

## Code style
- Python 3.10+; follow PEP 8 and keep functions small and documented.
- UI code lives in `ui/`, physics and models in `modalities/`, `simulation/`, `reconstruction/`.
