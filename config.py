import os
from dotenv import load_dotenv

load_dotenv()

# --- vLLM (OpenAI-compatible API) ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct")

# --- Exa MCP ---
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
EXA_MCP_BASE = "https://mcp.exa.ai/mcp"

def exa_mcp_url():
    url = EXA_MCP_BASE
    if EXA_API_KEY:
        url += f"?exaApiKey={EXA_API_KEY}"
    return url

# --- ForecastBench ---
ORGANIZATION = os.getenv("ORGANIZATION", "MyOrg")
MODEL_ORGANIZATION = os.getenv("MODEL_ORGANIZATION", "Meta")

QUESTION_SET_RAW_BASE = (
    "https://raw.githubusercontent.com/forecastingresearch/"
    "forecastbench-datasets/main/datasets/question_sets"
)
RESOLUTION_SET_RAW_BASE = (
    "https://raw.githubusercontent.com/forecastingresearch/"
    "forecastbench-datasets/main/datasets/resolution_sets"
)

MARKET_SOURCES = {"infer", "manifold", "metaculus", "polymarket"}
DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}

# --- Pipeline tuning ---
MAX_SEARCH_RESULTS = 5
LLM_TEMPERATURE = 0.0
# Qwen3.5 and other thinking models need high max_tokens because
# the reasoning/thinking phase consumes tokens before the actual answer.
LLM_MAX_TOKENS = 4096
