# Deployment Steps — GitHub + Streamlit Community Cloud

## 1) Create the GitHub repository

Create a new repository, for example:

`hr-policy-rag-streamlit`

Recommended: **Public** if you want recruiters to view the code. Keep all proprietary policy documents out of it.

## 2) Upload these files

Upload the complete contents of this package to the repository root. Your GitHub root should show `app.py` and `requirements.txt` directly.

## 3) Verify secrets are not committed

The real file `.streamlit/secrets.toml` must **not** appear in GitHub. Only `secrets.toml.example` should be committed.

If a real API key was ever committed, revoke/rotate the key before doing anything else. Removing the line in a later commit is not enough because Git history can retain it.

## 4) Test locally before deployment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

Add your key to `.streamlit/secrets.toml` before running.

## 5) Connect GitHub to Streamlit Community Cloud

Open Streamlit Community Cloud, sign in, and connect your GitHub account. You need admin permission for the repository you deploy.

## 6) Create the app

Choose **Create app** and select:

- Repository: your GitHub repo
- Branch: `main`
- Main file path: `app.py`

## 7) Advanced settings

Choose Python **3.12** and paste this into the Secrets box:

```toml
GROQ_API_KEY = "your-real-groq-key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Do not include quotes around the entire TOML block.

## 8) Deploy and validate

After deployment, test all three sample questions. Confirm:

- the answer displays page citations;
- the retrieved evidence expands correctly;
- no secret/API key is visible in logs or UI;
- the app works with the synthetic demo PDF;
- optional uploaded PDFs work during the current session.

## 9) Add the links to LinkedIn/GitHub

In your GitHub README, add your Streamlit URL near the top. In LinkedIn, share:

- the live Streamlit demo;
- the GitHub repository;
- a short architecture image or screen recording;
- the measured project story: Baseline LLM → Prompt Engineering → Hybrid RAG → Tuned RAG.

## 10) What not to publish

Do not commit or upload:

- the original proprietary handbook;
- course watermarks or personal-use PDFs;
- Groq API keys;
- personal email addresses embedded in source PDFs;
- confidential enterprise HR documents.
