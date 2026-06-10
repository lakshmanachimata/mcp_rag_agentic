"""Web search helpers for grounding chat responses."""

from __future__ import annotations


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return title, url, and snippet for each result."""
    from ddgs import DDGS

    results: list[dict] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title") or "Untitled",
                    "url": item.get("href") or "",
                    "snippet": item.get("body") or "",
                }
            )
    return [result for result in results if result["url"]]


def format_search_context(results: list[dict]) -> str:
    if not results:
        return ""

    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        sections.append(
            f"[{index}] {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Snippet: {result['snippet']}"
        )
    return "\n\n".join(sections)


def augment_prompt_with_web_search(user_prompt: str, results: list[dict]) -> str:
    context = format_search_context(results)
    if not context:
        return (
            "No web search results were found for this query. "
            "Answer from your own knowledge and note that no live web sources were available.\n\n"
            f"Question: {user_prompt}"
        )

    return (
        "Answer the user's question using the web search results below. "
        "Write a concise, well-structured summary like a cloud assistant with browsing. "
        "Cite sources inline using [1], [2], etc. "
        "If the results are insufficient or conflicting, say so clearly.\n\n"
        f"Web search results:\n\n{context}\n\n"
        f"Question: {user_prompt}"
    )
