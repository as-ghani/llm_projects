from email import message
from typing import Optional, List, Dict 
from agents.agent import Agent 
from agents.deals import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from openai import OpenAI
import json

class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-5.1"

    def __init__(self, collection):
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.openai = OpenAI()
        self.memory = None 
        self.opportunity = None 
        self.log("Autonomous Planning Agent is ready")

    def scan_the_internet_for_bargains(self) -> str:
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        estimate = self.ensemble.price(description)
        return f"The estimated true value of {description} is {estimate}"

    def notify_user_of_deal( self, description: str, deal_price: float, estimate: float, url: str) -> str:

        self.log(
            f"Autonomous Planning Agent is notifying user of a deal: {description[:60]}..."
        )
        discount = estimate - deal_price
        self.opportunity = Opportunity(
            deal=Deal(product_description=description, price=deal_price, url=url),
            estimate=estimate,
            discount=discount,
        )
        return "Notification sent to user successfully"

    scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_true_value",
        "description": "Given the description of an item, estimate how much it is actually worth",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item to be estimated",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_deal",
        "description": (
            "Call this exactly once, for the single best deal you find, once you are "
            "confident the offered price is well below your estimated true value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A clear description of the product",
                },
                "deal_price": {
                    "type": "number",
                    "description": "The price the deal is being offered at",
                },
                "estimate": {
                    "type": "number",
                    "description": "Your estimate of the true value of the product",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the deal",
                },
            },
            "required": ["description", "deal_price", "estimate", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message):
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            if tool:
                result = tool(**arguments)
            else:
                self.log(f"WARNING: model tried to call unknown tool '{tool_name}'")
                result = f"Unknown tool: {tool_name}"
            results.append(
                {"role": "tool", "content": str(result), "tool_call_id": tool_call.id}
            )
        return results

    system_message = "You find great deals on bargain products using your tools, and notify the user of the best bargain by calling notify_user_of_deal."
    user_message = """
    First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
    Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
    Then just reply OK to indicate success.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None
        messages = self.messages[:]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model=self.MODEL, messages=messages, tools=self.get_tools()
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity