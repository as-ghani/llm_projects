# Brochure Generator (GPT)

A small tool that scrapes a company's website and uses OpenAI's GPT models to automatically generate a short marketing brochure in Markdown — highlighting company culture, customers, and careers/jobs when that information is available.

## How It Works

1. **Scrape the landing page** — Fetches the title and visible text content of the given URL using `requests` + `BeautifulSoup`.
2. **Extract links** — Pulls all links from the page.
3. **Classify relevant links (LLM call #1)** — Sends the list of links to GPT (`gpt-4.1-mini`) and asks it to pick out the ones relevant to a brochure (e.g. About, Careers, Company pages), returning structured JSON.
4. **Fetch relevant pages** — Scrapes the content of each relevant link identified in step 3.
5. **Generate the brochure (LLM call #2)** — Combines the landing page content with the content of all relevant pages, and streams a Markdown brochure back from GPT, rendered live in the notebook via `IPython.display`.



## Project Structure

```
brochure-generator-gpt/
├── brochure-generator.ipynb   # Main notebook — run this
├── scraper.py                  # Website scraping helpers
└── README.md
```



## Requirements

- Python 3.12+
- An OpenAI API key
- Dependencies (managed via the parent `uv` workspace, or install manually):
  - `openai`
  - `python-dotenv`
  - `beautifulsoup4`
  - `requests`
  - `ipython`



## Setup

1. Create a `.env` file in the project root with your OpenAI API key:

```
   OPENAI_API_KEY=sk-proj-...
```

1. Install dependencies (from the repo root, if using `uv`):

```bash
   uv sync
```

1. Open `brochure-generator.ipynb` in Jupyter and run the cells in order.



## Usage

At the bottom of the notebook, call `stream_brochure` with a company name and its website URL:

```python
stream_brochure("HuggingFace", "https://huggingface.co")
```

The brochure will stream into the notebook as rendered Markdown.
 

## Notes / Limitations

- Scraping relies on `requests` + `BeautifulSoup`, so JavaScript-rendered pages may not return meaningful content.
- Page content is truncated (2,000 characters per page) and the final combined prompt is truncated to 5,000 characters to stay within reasonable token limits — this may cut off relevant information on content-heavy sites.
- Link classification depends on the LLM correctly identifying relevant pages; results can vary by site.
- No error handling yet for failed requests, malformed JSON responses, or missing links.

