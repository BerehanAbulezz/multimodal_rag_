"""
Corrective RAG loop.

After generating an answer, a "grader" call asks the LLM whether the
answer is actually supported by the retrieved context and answers the
question. If not:
  - the query is rewritten (for text queries) to search again,
  - top_k is widened a bit on each retry,
  - we regenerate and re-grade,
until `max_retries` is used up. The last attempt is always returned, with
a `warning` flag if it never passed grading.
"""
from . import config
from .generator import build_context_block, build_prompt, generate_answer

GRADER_PROMPT_TEMPLATE = """You are a strict grader for a retrieval-augmented QA system.
Given a question, the retrieved context, and a generated answer, decide whether
the answer is fully supported by the context AND actually answers the question.
Respond with exactly one word: "correct" or "incorrect". Do not explain.

Question: {query}

Context:
{context}

Answer:
{answer}

Verdict:"""

REWRITE_PROMPT_TEMPLATE = """Rewrite the following question to be clearer and more
specific for a semantic search system, while keeping the same intent and topic.
Return ONLY the rewritten question, nothing else.

Original question: {query}
Rewritten question:"""


def grade_answer(query: str, context: str, answer: str) -> bool:
    prompt = GRADER_PROMPT_TEMPLATE.format(query=query, context=context, answer=answer)
    interaction = config.client.interactions.create(model=config.GEMINI_MODEL, input=prompt)
    verdict = interaction.output_text.strip().lower()
    return "incorrect" not in verdict and "correct" in verdict


def rewrite_query(query: str) -> str:
    prompt = REWRITE_PROMPT_TEMPLATE.format(query=query)
    interaction = config.client.interactions.create(model=config.GEMINI_MODEL, input=prompt)
    return interaction.output_text.strip()


def corrective_rag_answer(
    query: str,
    retriever,
    top_k: int = 3,
    max_retries: int = 2,
    query_image=None,
):
    """
    query: the question text (always required, even for image queries -
           e.g. "What animal is this and what does it eat?")
    query_image: optional path/PIL.Image - if given, retrieval is done in
                 image mode against the shared CLIP space.
    Returns a dict: {answer, results, attempts, trace, warning?}
    """
    query_type = "image" if query_image is not None else "text"
    search_input = query_image if query_image is not None else query
    current_grading_query = query

    trace = []
    answer, results, context = None, None, None

    for attempt in range(max_retries + 1):
        results = retriever.retrieve(search_input, query_type=query_type, top_k=top_k + attempt)
        context = build_context_block(results)
        prompt = build_prompt(query, results)
        answer = generate_answer(prompt)
        is_correct = grade_answer(current_grading_query, context, answer)

        trace.append(
            {
                "attempt": attempt + 1,
                "search_query": search_input if query_type == "text" else "<image>",
                "answer": answer,
                "graded_correct": is_correct,
            }
        )

        if is_correct:
            return {"answer": answer, "results": results, "attempts": attempt + 1, "trace": trace}

        # Corrective step before the next try: only text queries can be
        # meaningfully rewritten - an image query is re-run as-is with a
        # wider top_k instead.
        if query_type == "text":
            search_input = rewrite_query(search_input)
            current_grading_query = search_input

    return {
        "answer": answer,
        "results": results,
        "attempts": max_retries + 1,
        "trace": trace,
        "warning": "low_confidence: answer did not pass grading after all retries",
    }
