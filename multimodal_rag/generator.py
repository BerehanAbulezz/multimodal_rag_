"""
Turns retrieved (text + image) results into a prompt, and calls Gemini
to get the final answer. Retrieved images are converted to captions via
BLIP first, since the LLM here is text-only.
"""
from . import config
from .captioning import generate_caption


def build_context_block(results) -> str:
    """results: list[(MultimodalItem, score)] -> formatted context string."""
    lines = []
    for i, (item, score) in enumerate(results, 1):
        animal = item.source.get("animal") or "?"
        if item.type == "text":
            topic = item.source.get("topic", "")
            lines.append(f"[Text {i}] ({animal} - {topic}): {item.content}")
        else:
            caption = generate_caption(item.content)
            lines.append(f"[Image {i}] ({animal}, file={item.content}): {caption}")
    return "\n".join(lines)


def build_prompt(query: str, results) -> str:
    context = build_context_block(results)
    return f"""Answer the question using only the context below. If the answer isn't in the context, say so explicitly.

Context:
{context}

Question: {query}
Answer:"""


def generate_answer(prompt: str) -> str:
    if config.client is None:
        raise RuntimeError(
            "Gemini client not initialized - call multimodal_rag.init_client(api_key) first."
        )
    interaction = config.client.interactions.create(model=config.GEMINI_MODEL, input=prompt)
    return interaction.output_text
