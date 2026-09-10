from agents.agent import Agent 
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.preprocessor import Preprocessor

class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.preprocessor = Preprocessor()

    def price(self, description: str) -> float:
        self.log("Running Ensemble Agent: Preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Pre-processed text using {self.preprocessor.model_name}")
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        combined = frontier * 0.8 + specialist * 0.2
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined