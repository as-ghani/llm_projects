# Webpage Summarizer — GPT & Ollama
 
A small project that scrapes a webpage and generates a short summary of its contents using two different LLM backends: OpenAI's GPT (via API) and a locally-running open-source model through Ollama.
 
## What it does
 
- Scrapes the title and body text of a given URL using `BeautifulSoup`
- Sends the scraped content to an LLM with a system prompt instructing it to summarize, ignoring navigation clutter
- Returns the summary as rendered Markdown
- Supports two interchangeable backends:
  - **GPT (OpenAI API)** — `gpt-4.1-mini` via the OpenAI Python SDK
  - **Ollama (local)** — `llama3.2` running locally, accessed through OpenAI's SDK pointed at Ollama's local server
## How it works
 
1. `scraper.py` fetches the raw HTML of a page and strips out scripts, styles, images, and inputs, returning clean text (truncated to 2,000 characters).
2. `summarizer.ipynb` builds a system + user prompt pair and sends the scraped text to either:
   - the OpenAI API (`https://api.openai.com`), or
   - a local Ollama server (`http://localhost:11434`)
3. The response is rendered as Markdown in the notebook.
## Setup
 
**1. Install dependencies** (from the repo root, using `uv`):
 
```bash
uv sync
```
 
**2. Set your OpenAI API key**
 
Create a `.env` file inside this project folder:
 
```
OPENAI_API_KEY=sk-proj-your-key-here
```
 
**3. (Optional) Run Ollama locally for the local-model version**
 
Make sure [Ollama](https://ollama.com) is installed and running, then pull the model used in this project:
 
```bash
ollama pull llama3.2
```
 
Verify it's running:
 
```bash
curl http://localhost:11434
```
 
## Usage
 
Open `summarizer.ipynb` and run the cells. At the bottom of each section:
 
```python
# GPT-based summary
display_summary("https://www.anthropic.com/")
 
# Ollama-based summary (requires Ollama running locally)
display_summary_ollama("https://www.anthropic.com/")
```
 
Swap in any URL you like.
 
## Files
 
| File | Purpose |
|---|---|
| `scraper.py` | Fetches and cleans webpage content |
| `summarizer.ipynb` | Builds prompts and calls GPT / Ollama to generate summaries |
| `README.md` | This file |
 
## Notes / Learnings
 
- Using the same `OpenAI` client class to talk to both OpenAI's API and a local Ollama server (just by swapping `base_url`) is a neat trick — Ollama exposes an OpenAI-compatible endpoint, so the same code works against a hosted or fully local model with almost no changes.
- Content is truncated to 2,000 characters before being sent to the LLM to keep prompts small and cheap.
- `.env` is gitignored — never commit real API keys.
