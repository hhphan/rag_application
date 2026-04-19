import io

import httpx
import pdfplumber


def download_pdf_bytes(pdf_url: str) -> bytes:
    response = httpx.get(pdf_url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    return response.content


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def parse_pdf_from_url(pdf_url: str) -> str:
    try:
        pdf_bytes = download_pdf_bytes(pdf_url)
        return extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        print(f"[parser] Failed to parse {pdf_url}: {exc}")
        return ""
