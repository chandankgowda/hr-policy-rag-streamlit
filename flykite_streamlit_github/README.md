# Flykite-style HR Policy Q&A Bot — Hybrid RAG Demo

A public-safe Streamlit demonstration of an enterprise HR policy Q&A pipeline using **MiniLM + FAISS + BM25 + Groq-hosted LLM generation**.

> **Important:** The repository should not contain the original course/company handbook unless you have explicit permission to publish it. The included `data/demo_hr_policy.pdf` is synthetic and created only for demonstration.

## Architecture

`PDF → page extraction → chunking → MiniLM embeddings → FAISS + BM25 → hybrid retrieval → Top-K evidence → grounded LLM → [Page N] citations`

The deployed defaults mirror the best balanced configuration used in the project:

- Chunk size: **1000 characters**
- Chunk overlap: **150 characters**
- Top-K: **4**
- Dense / BM25 weighting: **0.65 / 0.35**
- Reranking: **Off**
- Generation model: **openai/gpt-oss-120b** via Groq
- Embedding model: **sentence-transformers/all-MiniLM-L6-v2**

## Repository structure

```text
.
├── app.py
├── rag_engine.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── data/
    └── demo_hr_policy.pdf
```

## Run locally

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create the local secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

On Windows PowerShell:

```powershell
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
```

4. Edit `.streamlit/secrets.toml` and add your Groq API key.
5. Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app.
3. Select your GitHub repository, branch, and `app.py` as the entrypoint.
4. In **Advanced settings**, select Python **3.12**.
5. Add the following in **Secrets**:

```toml
GROQ_API_KEY = "your-real-key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

6. Deploy.

Do **not** commit `.streamlit/secrets.toml`; it is already ignored by `.gitignore`.

## Suggested LinkedIn demo flow

1. Open with a policy question.
2. Show the grounded answer and page citations.
3. Expand the retrieved evidence panel to demonstrate explainability.
4. Mention that hybrid FAISS + BM25 retrieval was selected after controlled RAG configuration testing.
5. Clarify that the public demo uses synthetic policy content while the project architecture supports private enterprise PDFs.

## Security / publishing checklist

- No API keys in GitHub history.
- No proprietary course/company handbook in the public repository.
- No personal email addresses, watermarks, employee data, or confidential policy data.
- Public demo policy is clearly labelled synthetic.
- LLM answers are framed as policy assistance, not final HR decisions.
