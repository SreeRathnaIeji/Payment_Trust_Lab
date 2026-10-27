# Payment Trust Lab

A local research prototype for synthetic PaySim fraud classification, a separate experimental node-feature-only GNN, a small selected PaySim test-split network sample, private saved-assessment history, and QR/link warning heuristics. It does **not** connect to bank/UPI accounts, verify payment recipients, or provide calibrated real-world fraud probabilities.

## Run locally (Windows PowerShell)

Install Python 3.11 (64-bit) and make sure the project folder contains `requirements.txt`, `backend/`, `frontend/`, and `Experiments/`. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000. The frontend is plain HTML/CSS/JavaScript served by FastAPI, so **`npm install` and `npm run dev` are not used**. Check API health at http://127.0.0.1:8000/api/health.

If pip fails with `getaddrinfo failed`, fix internet/DNS/VPN/proxy access and retry. If the package install says a file is in use, close other Python/pip processes and retry the install. The optional GNN endpoint requires a signed-in account and successful loading of TensorFlow and its saved artifacts; the app should show the backend error if it cannot run. XGBoost remains the primary model.

## Run with Docker

Requires Docker Desktop:

```powershell
docker compose up --build
```

Open http://127.0.0.1:8000. SQLite data persists in a named Docker volume; `docker compose down -v` **deletes** that volume. The first build can be large/slow because TensorFlow CPU is included. The port is bound to localhost by default.

## Frontend and features

- **Home:** brief overview and workflow cards; navigation is in the hamburger menu.
- **Transaction Analysis:** primary XGBoost result, optional separate node-only GNN, and labelled PaySim examples. Dataset labels are not sent to the prediction endpoint.
- **QR & Link Safety:** locally decodes QR images and checks website URLs using heuristic indicators. It never visits a URL or initiates a payment; UPI payee details are unverified.
- **Transaction Network:** visualizes a selected three-transaction sample from the chronological PaySim test split. The sample relationship inspector counts only those sample rows. The separate signed-in inspector searches only the user's own saved assessments. Neither is a full PaySim account-history lookup.
- **History:** private local assessments for the signed-in demo account.
- **Research & Model Evaluation:** separate offline test results. The graph-feature XGBoost and graph-enabled GNN experiments are not the same as the live node-only GNN endpoint.

## Tests

Frontend behavior tests:

```powershell
node --test tests/frontend_behavior.test.cjs
```

Python tests (install requirements first):

```powershell
python -m pytest tests -q
```

Some offline experiment tests require TensorFlow and may be resource-intensive. Test results must be run in the target environment before claiming the whole app has passed end-to-end testing.

## Model and data boundaries

- The live primary model is the saved chronological PaySim XGBoost artifact with threshold 0.95.
- Model scores are raw outputs and are **not calibrated real-world fraud probabilities**.
- The live optional GNN is node-feature-only and requires sign-in. It does not use graph history. The graph-enabled GNN is an offline research experiment only.
- The network view uses a small selected set of synthetic PaySim test-split transactions (steps 688–716), not the full dataset. Its counts are sample counts, not complete histories.
- URL rules are heuristics, not a trained malicious-URL classifier or a safety guarantee.
- Authentication and SQLite history are for local demonstration, not production financial authentication.

## GitHub checklist

Before committing, inspect `git status`, verify no `.venv/`, real transaction dataset, local database, secrets, or credentials are staged, then create an empty GitHub repository. From the project root:

```powershell
git init
git add .
git status
git commit -m "Prepare Payment Trust Lab research prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

Replace the remote URL with your own repository. No repository is created or pushed automatically by this package.
