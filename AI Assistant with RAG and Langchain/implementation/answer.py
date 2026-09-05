from pathlib import Path
from typing import Sequence
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
collection_name = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
wait = wait_exponential(multiplier=1, min=10, max=240)

RETRIEVAL_K = 20
FINAL_K = 10

llm = ChatOpenAI(model = MODEL, temperature=0)

embeddings = OpenAIEmbeddings(model = EMBEDDING_MODEL)
vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings,persist_directory=DB_NAME)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}
 
With this context, please answer the user's question. Be accurate, relevant and complete.
"""

class RankOrder(BaseModel):
    order: list[int] = Field(
       description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

@retry(wait=wait)
def rerank(question: str, chunks: Sequence[Document]) ->list[Document]:
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    structured_llm = llm.with_structured_output(RankOrder)
    result: RankOrder = structured_llm.invoke(messages)
    return [chunks[i - 1] for i in result.order]


def make_rag_messages(question: str, history: list[dict], chunks: Sequence[Document]) -> list[dict]:
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )

@retry(wait=wait)
def rewrite_query(question: str, history: list[dict] = None) -> str:
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    if history is None:
        history = []
 
    message = f"""
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.
 
This is the history of your conversation so far with the user:
{history}
 
And this is the user's current question:
{question}
 
Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.
 
EXAMPLE:
user: Who is the founder? -> Query: who is the founder?
assistant: The founder is FooBar
user: What role covers? -> Query: What role FooBar covers?
...
 
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = llm.invoke([{"role": "system", "content": message}])
    return response.content


def merge_chunks(chunks: list[Document], reranked: list[Document]) -> list[Document]:
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

def fetch_context_unranked(question: str) -> list[Document]:
    return vectorstore.similarity_search(question, k=RETRIEVAL_K)

def fetch_context(original_question: str, history: list[dict] = None) -> list[Document]:
    rewritten_question = rewrite_query(original_question, history)
    print(rewritten_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]

@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    chunks = fetch_context(question, history)
    messages = make_rag_messages(question, history, chunks)
    response = llm.invoke(messages)
    return response.content, chunks