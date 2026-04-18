import time
import xml.etree.ElementTree as ET

import httpx

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}
_PAGE_SIZE = 100


def _fetch_page(category: str, start: int, page_size: int) -> list[dict[str, str]]:
    url = (
        f"{_ARXIV_API}?search_query=cat:{category}"
        f"&start={start}&max_results={page_size}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers: list[dict[str, str]] = []

    for entry in root.findall("atom:entry", _NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=_NS)
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]

        title_el = entry.find("atom:title", _NS)
        title = (title_el.text or "").strip() if title_el is not None else ""

        authors = ", ".join(
            (name_el.text or "").strip()
            for author in entry.findall("atom:author", _NS)
            if (name_el := author.find("atom:name", _NS)) is not None
        )

        summary_el = entry.find("atom:summary", _NS)
        abstract = (summary_el.text or "").strip() if summary_el is not None else ""

        published_el = entry.find("atom:published", _NS)
        published_at = (published_el.text or "").strip() if published_el is not None else ""

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "published_at": published_at,
            }
        )

    return papers


def fetch_arxiv_papers(
    category: str,
    max_results: int,
    start: int = 0,
) -> list[dict[str, str]]:
    papers: list[dict[str, str]] = []
    offset = start

    while len(papers) < max_results:
        page_size = min(_PAGE_SIZE, max_results - len(papers))
        page = _fetch_page(category, offset, page_size)
        if not page:
            break
        papers.extend(page)
        offset += len(page)
        if len(page) < page_size:
            break
        if len(papers) < max_results:
            time.sleep(3)

    return papers[:max_results]
