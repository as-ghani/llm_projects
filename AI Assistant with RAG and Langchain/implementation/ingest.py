from pathlib import Path
from multiprocessing import Pool
from dotenv import load_dotenv
from langchain_core import embeddings
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm import tqdm 
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100
WORKERS = 3
wait = wait_exponential(multiplier=1, min=10, max=240)

class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def to_document(self , doc_info: dict) -> Document:
        content = f"{self.headline}\n\n{self.summary}\n\n{self.original_text}"
        metadata = {"source": doc_info["source"], "type": doc_info["type"]}
        return Document(page_content = content, metadata = metadata)

class Chunks(BaseModel):
    chunks: list[Chunk]

def fetch_documents() -> list[dict]:
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        loader = DirectoryLoader(str(folder),glob="**/*.md",loader_cls = TextLoader, loader_kwargs={"encoding": "utf-8"},)
        for doc in loader.load():
            documents.append({"type": folder.name, "source": doc.metadata["source"], "text": doc.page_content})
    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.
 
The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.
 
For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.
 
Here is the document:
 
{document["text"]}
 
Respond with the chunks.
"""


@retry(wait = wait )
def process_document(document: dict) -> list[Document]:
    llm = ChatOpenAI(model = MODEL, temperature= 0).with_structured_output(Chunks)
    result : Chunks = llm.invoke(make_prompt(document))
    return [chunk.to_document(document) for chunk in result.chunks]

def create_chunks(documents: list[dict]) -> list[Document]: 
    chunks : list[Document] = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document,documents), total=len(documents)):
            chunks.extend(result)
    return chunks

def create_vectorstore(chunks: list[Document]) -> None:
    embeddings = OpenAIEmbeddings(model= embedding_model)
    vectorstore = Chroma(collection_name= collection_name, embedding_function= embeddings, persist_directory=DB_NAME)
    vectorstore.reset_collection()
    ids = vectorstore.add_documents(chunks)
    print(f"Vectorstore created with {len(ids)} documents")

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_vectorstore(chunks)
    print("Ingestion complete")