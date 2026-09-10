import re
from typing import List, Dict
from openai import OpenAI 
from sentence_transformers import SentenceTransformer 
from agents.agent import Agent 

class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.RED 

    MODEL = "gpt-4o-mini"

    def __init__(self, collection):
        self.log("Initializing Frontier Agent")
        self.client = OpenAI()
        self.MODEL = "gpt-5.1"
        self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) ->str:
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n "
        for similar, price in zip(similars, prices):
            message +=f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str,str]]:
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{description}\n\n"
        message += self.make_context(similars, prices)
        return [{"role": "user", "content": message}]
    
    def find_similars(self, description: str):
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products"
        )
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results["documents"][0][:]
        prices = [m["price"] for m in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        documents, prices = self.find_similars(description)
        self.log(
            f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products"
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description, documents, prices),
            seed=42,
            reasoning_effort="none",
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
