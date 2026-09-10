import os 
import json 
import queue
import logging
import threading
from typing import List 
import gradio as gr
import chromadb
from dotenv import load_dotenv
from agents.deals import Opportunity
from agents.autonomous_planning_agent import AutonomousPlanningAgent

load_dotenv(override=True)

MEMORY_FILE = "memory.json"
CHROMA_PATH = os.getenv("CHROMA_PATH", "products_vectorstore")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "products")
MAX_LOG_LINES = 300

log_queue: "queue.Queue[str]" = queue.Queue()
log_lines: List[str] = []

ANSI_COLOR_MAP = {
    "\033[30m": "#6c757d",
    "\033[31m": "#ff6b6b",
    "\033[32m": "#51cf66",
    "\033[33m": "#ffd43b",
    "\033[34m": "#4dabf7",
    "\033[35m": "#cc5de8",
    "\033[36m": "#22b8cf",
    "\033[37m": "#f1f3f5",
}
ANSI_BG = "\033[40m"
ANSI_RESET = "\033[0m"

def ansi_to_html(raw_line: str) -> str:
    text = raw_line
    color = "#f1f3f5"
    for code, hexcolor in ANSI_COLOR_MAP.items():
        if code in text:
            color = hexcolor
            text = text.replace(code, "")
    text = text.replace(ANSI_BG, "").replace(ANSI_RESET, "")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<div style="color:{color};">{text}</div>'

class QueueLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_queue.put(self.format(record))
        except Exception:
            pass

def setup_logging():
    handler = QueueLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, QueueLogHandler) for h in root.handlers):
        root.addHandler(handler)

def load_memory() -> List[Opportunity]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            raw = json.load(f)
        return [Opportunity(**item) for item in raw]
    except Exception as e:
        logging.warning(f"Could not load {MEMORY_FILE}: {e}")
        return []

def save_memory(memory: List[Opportunity]):
    with open(MEMORY_FILE, "w") as f:
        json.dump([opp.model_dump() for opp in memory], f, indent=2)

def opportunities_to_rows(memory: List[Opportunity]):
    return [
        [
            opp.deal.product_description[:150],
            f"${opp.deal.price:,.2f}",
            f"${opp.estimate:,.2f}",
            f"${opp.discount:,.2f}",
            opp.deal.url,
        ]
        for opp in reversed(memory)
    ]

class DealHuntingApp:
    def __init__(self):
        setup_logging()
        self.memory: List[Opportunity] = load_memory()
        self.lock = threading.Lock()
        self.running = False
        self.planner = None
        self._init_error = None
        try:
            logging.info("Connecting to the Chroma vector datastore...")
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_or_create_collection(COLLECTION_NAME)
            logging.info("Initializing the Autonomous Planning Agent (this also connects to Modal)...")
            self.planner = AutonomousPlanningAgent(collection)
            logging.info(f"Ready. Loaded {len(self.memory)} opportunities from memory.")
        except Exception as e:
            self._init_error = str(e)
            logging.error(f"Startup failed: {e}")
            logging.error(
                "Check that CHROMA_PATH points at a built vectorstore and that "
                "'modal deploy modal/pricer_service.py' has been run."
            )
 
    def run_once(self):
        if self.planner is None:
            logging.error(f"Cannot run: agent never initialized ({self._init_error})")
            return
        if self.running:
            logging.info("A scan is already in progress - skipping this request")
            return
        self.running = True
        try:
            with self.lock:
                memory_snapshot = list(self.memory)
            opportunity = self.planner.plan(memory=memory_snapshot)
            if opportunity:
                with self.lock:
                    self.memory.append(opportunity)
                    save_memory(self.memory)
                logging.info(
                    f"New opportunity saved: {opportunity.deal.product_description[:60]}... "
                    f"(${opportunity.discount:.2f} below estimate)"
                )
            else:
                logging.info("Run complete - no opportunity worth flagging this time")
        except Exception as e:
            logging.error(f"Agent framework run failed: {e}")
        finally:
            self.running = False
 
 
app_state = DealHuntingApp()



def run_agent_framework():
    if app_state.running:
        return "A scan is already running..."
    thread = threading.Thread(target=app_state.run_once, daemon=True)
    thread.start()
    return "Agent run started - watch the log below"
 
 
def drain_log_queue() -> None:
    while not log_queue.empty():
        log_lines.append(ansi_to_html(log_queue.get()))
    if len(log_lines) > MAX_LOG_LINES:
        del log_lines[: len(log_lines) - MAX_LOG_LINES]
 
 
def refresh_logs():
    drain_log_queue()
    body = "".join(log_lines) or (
        '<div style="color:#888;">Waiting for the agent to run...</div>'
    )
    return (
        '<div style="height:420px;overflow-y:auto;background:#1e1e1e;padding:12px;'
        'border-radius:8px;font-family:monospace;font-size:13px;">' + body + "</div>"
    )
 
 
def refresh_table():
    return opportunities_to_rows(app_state.memory)

with gr.Blocks(title="Autonomous Deal Hunter", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Autonomous Deal Hunting Agent\n"
        "Scans live deal feeds, estimates true product value with a fine-tuned "
        "specialist model + frontier model ensemble, and flags the best bargains."
    )
 
    with gr.Row():
        with gr.Column(scale=1):
            run_btn = gr.Button("Run Agent Framework", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("**Live agent log**")
            log_html = gr.HTML(value=refresh_logs())
        with gr.Column(scale=2):
            table = gr.Dataframe(
                headers=["Description", "Deal Price", "Estimated Value", "Discount", "URL"],
                value=opportunities_to_rows(app_state.memory),
                label="Opportunities Found",
                wrap=True,
            )
 
    run_btn.click(fn=run_agent_framework, outputs=status)
 
    timer = gr.Timer(1.0)
    timer.tick(fn=refresh_logs, outputs=log_html)
    timer.tick(fn=refresh_table, outputs=table)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
