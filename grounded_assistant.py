"""
grounded_assistant.py
=====================
Grounded Reply Assistant — Week 1 Homework (Mini-Project)

Tasks covered:
  Task 1 · Retrieve (Embeddings RAG)
  Task 2 · Decide  (Function / Tool Calling)
  Task 3 · Answer  (Structured + Grounded)

Backend : Groq API  llama-3.1-8b-instant  (tool-calling)
Embeddings : nomic-embed-text-v1.5 via SentenceTransformer (local)
Structured output : instructor (MD_JSON mode) wraps the Groq client
"""

import sys, pathlib, json, uuid, os, re
import chromadb
import instructor
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# This file lives at:  <repo>/week1/homework/grounded_assistant.py
# masar_utils is now local
_REPO_ROOT = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Step 0 — Credentials & Clients
# ---------------------------------------------------------------------------
import sys, pathlib, json, uuid, os, re
import chromadb
import instructor
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import openai

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

# Ensure masar_utils is in path (it's done in path setup block above)
from masar_utils import pretty, count_tokens, estimate_cost, get_client

# -- 1. Primary Client (HuggingFace Router) --
_HF_BASE_URL = os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1")
_HF_TOKEN    = os.environ.get("HF_TOKEN", "")
PRIMARY_MODEL = os.environ.get("HF_MODEL", "openai/gpt-oss-20b:groq")

if not _HF_TOKEN:
    print("Warning: HF_TOKEN not found, falling back directly to Groq")
    _primary_raw_client = None
    primary_client = None
else:
    _primary_raw_client = OpenAI(base_url=_HF_BASE_URL, api_key=_HF_TOKEN)
    primary_client = instructor.from_openai(_primary_raw_client, mode=instructor.Mode.MD_JSON)

# -- 2. Fallback Client (Groq directly) --
os.environ.setdefault("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
FALLBACK_MODEL = "llama-3.1-8b-instant"
_fallback_raw_client = get_client()
fallback_client = instructor.from_openai(_fallback_raw_client, mode=instructor.Mode.MD_JSON)

MODEL = PRIMARY_MODEL
print(f"Primary model : {PRIMARY_MODEL}")
print(f"Fallback model: {FALLBACK_MODEL}")

# -- 3. Embeddings --
os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
print("Loading embedding model...")
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
print("Clients and embedding model ready!")

# Global variable to track which model was actually used
_last_model_used = None

def get_last_model_used():
    global _last_model_used
    return _last_model_used

# Helper for fallback execution
def execute_with_fallback(*args, **kwargs):
    global _last_model_used
    """Executes a chat completion call with the primary client, falling back if it fails."""
    client_inst_from_kwargs = kwargs.pop('client_instance', None)
    primary_inst = client_inst_from_kwargs if client_inst_from_kwargs is not None else _primary_raw_client
    
    try:
        if _primary_raw_client is None:
            raise openai.APIStatusError("No HF token", response=None, body=None) # type: ignore
        
        kwargs['model'] = PRIMARY_MODEL
        print(f"  [Model Executing] {PRIMARY_MODEL}")
        _last_model_used = PRIMARY_MODEL
        return primary_inst.chat.completions.create(*args, **kwargs)
    except Exception as e:
        print(f"  [Fallback Triggered] Primary failed: {type(e).__name__} - {e}")
        if hasattr(primary_inst, 'mode'):
            fallback_inst = fallback_client
        else:
            fallback_inst = _fallback_raw_client
            
        kwargs['model'] = FALLBACK_MODEL
        print(f"  [Model Executing] {FALLBACK_MODEL}")
        _last_model_used = FALLBACK_MODEL
        return fallback_inst.chat.completions.create(*args, **kwargs)

# ---------------------------------------------------------------------------
# Task 1 · Retrieve (Embeddings RAG)
# ---------------------------------------------------------------------------

# 1.1 — Load the handbook corpus
DOC_PATH = pathlib.Path(__file__).parent / "data" / "company_handbook.md"
raw_text = DOC_PATH.read_text(encoding="utf-8")


# 1.2 — Chunk the text (Header-based Chunking)
def chunk_by_headers(text: str) -> list[str]:
    """Split markdown into chunks; bullet-only sections get one chunk per bullet."""
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    chunks = []

    if sections[0].strip():
        chunks.append(sections[0].strip())

    for section in sections[1:]:
        section = section.strip()
        lines = section.split("\n")
        header = lines[0]
        body_lines = [l for l in lines[1:] if l.strip()]
        bullets = [l for l in body_lines if l.strip().startswith("-")]

        # if the whole body is a bullet list, emit one chunk per bullet
        if bullets and len(bullets) == len(body_lines):
            for b in bullets:
                chunks.append(f"{header}\n{b}")
        else:
            chunks.append(section)

    return chunks


chunks = chunk_by_headers(raw_text)


# 1.3 — ChromaDB Vector Database
chroma_client = chromadb.Client()
vector_db = chroma_client.get_or_create_collection(
    name="handbook",
    metadata={"hnsw:space": "cosine"},
)


# 1.4 — Embed all chunks & build the index
def embed(texts) -> list:
    """Embed a list of strings using SentenceTransformer."""
    if isinstance(texts, str):
        texts = [texts]
    return embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()


chunk_vecs = embed(chunks)
vector_db.add(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=chunk_vecs,
)


# 1.5 — Retrieve top-k chunks (with similarity threshold)
SIMILARITY_THRESHOLD = 0.45


def retrieve(question: str, k: int = 3) -> list[tuple[str, float]]:
    """
    Embed the question, query ChromaDB, return top-k chunks.
    Returns EMPTY list when best score < SIMILARITY_THRESHOLD.
    ChromaDB cosine distance: 0=identical.  sim = 1 - distance.
    """
    q_emb = embed([question])[0]
    results = vector_db.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "distances"],
    )
    docs  = results["documents"][0]
    dists = results["distances"][0]
    hits  = [(doc, 1.0 - dist) for doc, dist in zip(docs, dists)]
    if not hits or hits[0][1] < SIMILARITY_THRESHOLD:
        return []
    return hits


# ---------------------------------------------------------------------------
# Task 2 · Decide (Function / Tool Calling)
# ---------------------------------------------------------------------------

# 2.1 — Dummy store data
CATALOG: dict = {
    "sku-tee": {"name": "Acme T-Shirt", "price": 20.0, "stock": 10},
    "sku-mug": {"name": "Acme Mug",     "price": 12.0, "stock": 5},
    "sku-cap": {"name": "Acme Cap",     "price": 15.0, "stock": 0},
}
ORDERS: dict = {}


# 2.2 — Tool function implementations
def search_handbook(query: str) -> dict:
    """Look up Acme Cloud policy/pricing/support facts via RAG."""
    hits = retrieve(query)
    if not hits:
        return {
            "context": "",
            "found": False,
            "message": "No relevant handbook content found for this question.",
        }
    context = "\n\n".join(f"[{i+1}] {doc}" for i, (doc, _) in enumerate(hits))
    return {"context": context, "found": True}


def buy_item(item_id: str, quantity: int) -> dict:
    """Create an order. Returns error if item unknown, quantity < 1, or out of stock."""
    if quantity < 1:
        return {"error": "InvalidQuantity", "message": "Quantity must be at least 1."}
    if item_id not in CATALOG:
        return {
            "error": "ItemNotFound",
            "message": f"'{item_id}' not in catalog. Available: {list(CATALOG.keys())}",
        }
    item = CATALOG[item_id]
    if item["stock"] < quantity:
        return {
            "error": "OutOfStock",
            "message": f"Only {item['stock']} unit(s) of '{item['name']}' in stock (requested {quantity}).",
        }
    item["stock"] -= quantity
    order_id = "ord-" + uuid.uuid4().hex[:8]
    total     = round(item["price"] * quantity, 2)
    ORDERS[order_id] = {
        "item_id": item_id, "item_name": item["name"],
        "quantity": quantity, "total": total, "status": "confirmed",
    }
    return {"order_id": order_id, "item": item["name"],
            "quantity": quantity, "total": total, "status": "confirmed"}


def return_item(order_id: str) -> dict:
    """Mark an existing order as returned."""
    if order_id not in ORDERS:
        return {"error": "OrderNotFound", "message": f"Order '{order_id}' does not exist."}
    order = ORDERS[order_id]
    if order["status"] == "returned":
        return {"error": "AlreadyReturned", "message": f"Order '{order_id}' was already returned."}
    CATALOG[order["item_id"]]["stock"] += order["quantity"]
    order["status"] = "returned"
    return {
        "order_id": order_id, "status": "returned", "refunded": order["total"],
        "message": f"Order '{order_id}' returned, ${order['total']:.2f} refunded.",
    }


# 2.3 — Tool schemas (OpenAI-compatible format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_handbook",
            "description": (
                "Look up Acme Cloud policy/pricing/support facts. "
                "Use when the user asks about the company, plans, SLA, backups, refunds, or support."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language question to look up in the handbook.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buy_item",
            "description": "Purchase an item from the Acme store. Use when the user wants to buy something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Catalog SKU. Must be exactly one of: sku-tee, sku-mug, sku-cap.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of units (>= 1).",
                    },
                },
                "required": ["item_id", "quantity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_item",
            "description": "Return a previously placed order. Use when the user wants to return or cancel an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to return.",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

# 2.4 — safe_call — error-safe tool runner
def safe_call(func, args: dict) -> str:
    """Run a tool returning JSON string. On failure return structured error."""
    try:
        return json.dumps(func(**args))
    except Exception as e:
        return json.dumps({"error": type(e).__name__, "message": str(e)})


TOOL_IMPLS: dict = {
    "search_handbook": search_handbook,
    "buy_item":        buy_item,
    "return_item":     return_item,
}


# 2.5 — run_agent() — the multi-round tool-call loop
def run_agent(
    user_msg: str,
    tools: list = TOOLS,
    tool_impls: dict = TOOL_IMPLS,
    system: str = "You are a helpful assistant. Use tools when useful.",
    max_rounds: int = 5,
    verbose: bool = True,
) -> str:
    """
    Multi-round tool-call loop.

    Parameters
    ----------
    user_msg    : str   The user message to process.
    tools       : list  Tool JSON schemas (defaults to TOOLS).
    tool_impls  : dict  {name -> callable} map (defaults to TOOL_IMPLS).
    system      : str   System prompt.
    max_rounds  : int   Safety cap on the number of model calls.
    verbose     : bool  Print each round and tool call.

    Returns
    -------
    str  The model's final text answer.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]

    for round_i in range(1, max_rounds + 1):
        resp  = _raw_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            parallel_tool_calls=False,
        )
        msg   = resp.choices[0].message
        calls = msg.tool_calls or []

        if resp.choices[0].finish_reason == "stop":
            if verbose:
                print(f"[round {round_i}] final answer")
            return msg.content

        if verbose:
            print(f"[round {round_i}] model requested {len(calls)} tool call(s): "
                  + ", ".join(c.function.name for c in calls))

        messages.append(msg.model_dump(exclude_none=True))
        for c in calls:
            name    = c.function.name
            args    = json.loads(c.function.arguments or "{}")
            impl    = tool_impls.get(name)
            content = json.dumps({"error": "UnknownTool", "message": name}) if impl is None else safe_call(impl, args)
            if verbose:
                print(f"    {name}({args}) -> {content[:200]}")
            messages.append({"role": "tool", "tool_call_id": c.id, "content": content})

    return "(stopped: hit max_rounds without a final answer)"


# ---------------------------------------------------------------------------
# Task 3 · Answer (Structured + Grounded)
# ---------------------------------------------------------------------------

# 3.1 — Pydantic GroundedReply model
class GroundedReply(BaseModel):
    """Structured response model for the Grounded Reply Assistant."""
    answer:     str       = Field(description="The assistant's answer to the user.")
    sources:    list[str] = Field(description="Titles or identifiers of the handbook sections used to answer the question.")
    confidence: float     = Field(description="Confidence score 0.0-1.0.", ge=0.0, le=1.0)
    answered:   bool      = Field(description="True ONLY if the user's request was fulfilled using the handbook context or store tools; False for greetings, off-topic, or if no grounded answer was provided.")


# 3.2 — System prompts
TOOL_SYSTEM_PROMPT = """\
Role: You are a precise, grounded support assistant for Acme Cloud.

Task:
1. If the user asks about Acme Cloud topics (company info, pricing, plans, SLA, backups,
   refunds, security, regions, connection limits, data export, support, passwords),
   you MUST call search_handbook — even for short phrases like "reset password".
2. If the user wants to buy or return an item, use the appropriate store tool.
3. For anything unrelated to Acme Cloud (world events, geography, general knowledge),
   do NOT call any tool — just reply in plain text that you can only help with Acme topics.

Constraints:
- ONLY use the tools provided: search_handbook, buy_item, return_item. Never invent or call other tools.
- Never guess or hallucinate facts.
- Use EXACTLY the quantity the user specified for each item. Never assume, round, or change a quantity.
- ONLY call return_item if the user explicitly asks to return, cancel, or refund an order. NEVER call return_item right after a purchase unless the user separately and explicitly requested a return.
- Only call each tool for actions the user actually asked for. Do not perform extra, unrequested actions."""


STRUCTURED_SYSTEM_PROMPT = """\
Role: You are a precise, grounded support assistant for Acme Cloud formatting the final response.

Constraints for GroundedReply:
- The user's message may contain MULTIPLE requests (e.g. a question + a purchase).
  You MUST address every part of it in `answer`. Never drop a part just because
  it wasn't the last thing discussed.
- For handbook questions, answer ONLY from the raw tool context provided.
- For store actions, confirm the result (order id, total, or error) from the raw tool context.
- Populate `sources` ONLY with handbook section titles actually used (from search_handbook results).
  If no handbook lookup contributed, sources = [].
- Never invent handbook section titles; use ONLY titles that appear exactly in the retrieved context.
- If a tool never returned a result for part of the request, say so — don't guess."""


# 3.3 — grounded_assistant() — the full pipeline
def grounded_assistant(
    user_msg: str,
    max_rounds: int = 5,
    verbose: bool = True,
) -> tuple["GroundedReply", list[str]]:
    """
    Run the full Grounded Reply pipeline for one user message.

    Returns
    -------
    reply        : GroundedReply  - the parsed, structured answer
    tools_called : list[str]      - names of every tool that was invoked
    """
    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    tools_called: list[str] = []
    tool_results_log: list[str] = []

    # Agent loop (uses _raw_client so instructor does not intercept)
    for round_i in range(1, max_rounds + 1):
        resp = execute_with_fallback(
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            parallel_tool_calls=False,
            client_instance=_primary_raw_client
        )
        msg   = resp.choices[0].message
        calls = msg.tool_calls or []

        if resp.choices[0].finish_reason == "stop":
            if verbose:
                print(f"  [round {round_i}] no tool calls -> structured parse")
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        if verbose:
            print(f"  [round {round_i}] {len(calls)} tool call(s): "
                  + ", ".join(c.function.name for c in calls))
        messages.append(msg.model_dump(exclude_none=True))

        for tc in calls:
            name  = tc.function.name
            args  = json.loads(tc.function.arguments or "{}")
            impl  = TOOL_IMPLS.get(name)
            if impl is None:
                content = json.dumps({"error": "UnknownTool", "message": name})
            else:
                content = safe_call(impl, args)
                tools_called.append(name)
                tool_results_log.append(f"[{name}] {content}")
            if verbose:
                print(f"    {name}({args}) -> {content[:200]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    else:
        if verbose:
            print(f"  [warning] hit max_rounds={max_rounds}")

    context_block = "\n\n".join(tool_results_log) if tool_results_log else "(no tools were called)"

    structured_messages = [
        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
        {"role": "system", "content": f"Raw tool results (ground truth — cover ALL of these):\n{context_block}"},
    ]

    reply: GroundedReply = execute_with_fallback(
        messages=structured_messages,
        response_model=GroundedReply,
        temperature=0,
        client_instance=primary_client
    )

    if verbose:
        prompt_text = "".join(m.get("content", "") for m in structured_messages if isinstance(m, dict) and "content" in m)
        comp_text   = reply.answer if hasattr(reply, "answer") else ""
        p_toks = count_tokens(prompt_text)
        c_toks = count_tokens(comp_text)
        cost   = estimate_cost(p_toks, c_toks)
        print(f"  [Cost Estimate] ~{p_toks} prompt tokens, ~{c_toks} completion tokens => ${cost:.6f}")

    return reply, tools_called





if __name__ == "__main__":
    print("=" * 60)
    print("grounded_assistant.py — Demo Loop")
    print("=" * 60)

    # Reset store state for demo
    CATALOG["sku-tee"]["stock"] = 10
    CATALOG["sku-mug"]["stock"] = 5
    CATALOG["sku-cap"]["stock"] = 0
    ORDERS.clear()

    demo_questions = [
        "Hi there!",
        "How long are Pro backups kept?",
        "What is the Scale plan SLA?",
        "What is the GDP of Saudi Arabia?",
        "I'd like to buy 2 mugs.",
        "I want to buy the cap.",
    ]

    for q in demo_questions:
        print("\n" + "=" * 70)
        print(f"USER: {q}")
        reply, tools = grounded_assistant(q, verbose=True)
        pretty(reply.model_dump(), title="GroundedReply")
        print(f"Tools called: {tools}")
