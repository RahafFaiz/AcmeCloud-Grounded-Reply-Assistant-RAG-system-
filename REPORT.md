# REPORT — Grounded Reply Assistant · Week 1 Homework

## Eval Results
## Multi-Model Evaluation (9 Cases)
You can see the 9 cases results here 👇
`@eval_results.csv`

---

## Grounded vs. Ungrounded

When grounding is **on** (RAG enabled), the assistant retrieves exact text from the
company handbook before answering; it cites the source section and refuses to answer
when no relevant chunk is found (similarity < 0.45), giving `answered=False` for
off-topic questions. When grounding is **off** (raw model, no tools), the model answers
from its pre-training knowledge, which may be outdated, incorrect, or hallucinated —
and it never refuses, so it will confidently fabricate policy details it cannot
actually know.

**Example (Query: "What is the refund window for annual plans?"):**
- **Grounded (ON):** Retrieves the exact policy from the handbook and answers accurately: *"The refund window for annual plans is within the first 30 days, prorated; after that no refund is available."*
- **Ungrounded (OFF):** Relies on pre-trained knowledge and confidently hallucinates a fake policy: *"According to our 'Fair Cloud Practices' policy, which is outlined in Section 3... For cancellations received between Day 1-Day 30 of the annual term, we offer a pro-rated refund minus a 20% administrative fee... For cancellations after Day 180, we do not offer any refund."*

---

## API Summary

| Feature | Value |
|---------|-------|
| Chat model | `openai/gpt-oss-20b:groq` (Primary) / `llama-3.1-8b-instant` (Fallback) |
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` (SentenceTransformer, local) |
| Structured output | `instructor` (MD_JSON mode) |
| Vector DB | ChromaDB in-memory, cosine similarity |
| Similarity threshold | 0.45 (below → refuse) |


---

## Bonus: Architectural Improvements

During testing, we encountered and resolved two major limitations regarding reasoning capacity and retrieval precision. 

### 1. Multi-part Query Handling (Model Upgrade)
- **Before (`llama-3.1-8b-instant`)**: The smaller 8B model struggled to process multi-part queries (e.g., *"What is the GDP of Saudi Arabia? I want to buy a t-shirt"*). It often suffered from "attention collapse", fixating on only one part of the prompt, ignoring the second action, or failing to populate the structured output correctly.
- **After (Upgraded to `openai/gpt-oss-20b:groq`)**: By switching to a larger, more capable model (with fallback architecture in place), the agent now effortlessly isolates multiple intents. It successfully refuses the off-topic question (GDP) while simultaneously executing the store tool (buying a t-shirt) and formatting everything perfectly into the `GroundedReply` schema.

### 2. Precision Chunking (RAG Improvement)
- **Before (Coarse Chunking)**: Text was split exclusively by `##` headers. This created massive chunks (e.g., the entire Account Management section). When a user asked a short, specific question like *"Reset password"*, the dense chunk diluted the embedding vector, causing it to drop below the `0.45` similarity threshold, resulting in retrieval failure.
- **After (Granular Bullet Chunking)**: The `chunk_by_headers()` logic was updated to detect bulleted lists and split them into individual `[Header] + [Bullet]` chunks. This drastically improves semantic density and precision, allowing the vector database to accurately catch and retrieve the exact policy line for very short queries.


