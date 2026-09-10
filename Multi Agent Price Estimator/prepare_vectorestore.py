import os
import argparse
 
import chromadb
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from agents.items import Item
 
load_dotenv(override=True)
 
DB_PATH = os.getenv("CHROMA_PATH", "products_vectorstore")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products")
HF_DATASET = os.getenv("HF_DATASET", "ed-donner/items_lite")
BATCH_SIZE = 1000
 
 
def hf_login():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to your .env or export it before running this script."
        )
    login(token=hf_token, add_to_git_credential=False)
 
 
def load_dataset():
    print(f"Loading dataset '{HF_DATASET}' from the Hugging Face Hub...")
    train, val, test = Item.from_hub(HF_DATASET)
    print(
        f"Loaded {len(train):,} training items, {len(val):,} validation items, "
        f"{len(test):,} test items"
    )
    return train
 
 
def build_collection(client, encoder, train, rebuild: bool):
    existing = [c.name for c in client.list_collections()]
 
    if COLLECTION_NAME in existing:
        if not rebuild:
            print(
                f"Collection '{COLLECTION_NAME}' already exists at '{DB_PATH}' - "
                "skipping build. Pass --rebuild to force a fresh build."
            )
            return client.get_collection(COLLECTION_NAME)
        print(f"--rebuild passed: deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
 
    print(f"Creating collection '{COLLECTION_NAME}' and embedding {len(train):,} items...")
    collection = client.create_collection(COLLECTION_NAME)
 
    for i in tqdm(range(0, len(train), BATCH_SIZE)):
        batch = train[i : i + BATCH_SIZE]
        documents = [item.summary for item in batch]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in batch]
        ids = [f"doc_{j}" for j in range(i, i + len(documents))]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
 
    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count():,} items.")
    return collection
 
 
def main():
    parser = argparse.ArgumentParser(description="Build the product vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the collection even if it already exists.",
    )
    args = parser.parse_args()
 
    hf_login()
    train = load_dataset()
 
    print("Loading sentence-transformer encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
 
    print(f"Connecting to persistent Chroma store at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
 
    build_collection(client, encoder, train, rebuild=args.rebuild)
 
 
if __name__ == "__main__":
    main()
 