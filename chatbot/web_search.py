"""Web search helpers for grounding chat responses."""

from __future__ import annotations

import re

_WEB_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:please\s+)?(?:only\s+)?(?:do\s+a\s+)?web\s+search(?:\s+(?:the\s+)?web)?(?:\s+for)?\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:only\s+)?search(?:\s+(?:the\s+)?web|\s+online|\s+the\s+internet)?(?:\s+for)?\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:please\s+)?(?:only\s+)?google\s+(.+)$", re.IGNORECASE),
    re.compile(
        r"^(?:please\s+)?(?:only\s+)?look\s+up\s+(.+?)(?:\s+online|\s+on\s+the\s+web)?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:only\s+)?find\s+(.+?)\s+on\s+the\s+(?:web|internet)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:only\s+)?browse(?:\s+the\s+web)?(?:\s+for)?\s+(.+)$",
        re.IGNORECASE,
    ),
)


def extract_web_search_query(prompt: str) -> str | None:
    """Return the search query when the prompt asks for web search only."""
    text = prompt.strip()
    if not text:
        return None

    for pattern in _WEB_ONLY_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        query = match.group(1).strip().rstrip(".?!")
        if query:
            return query
    return None


def is_web_search_only_request(prompt: str) -> bool:
    """Return True when the user wants results from the web only."""
    return extract_web_search_query(prompt) is not None


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
