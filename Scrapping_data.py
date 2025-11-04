import logging
from typing import List, Optional
from langchain_community.document_loaders import PlaywrightURLLoader, WebBaseLoader
from bs4 import SoupStrainer
from langchain_core.documents import Document
import re

logging.basicConfig(level=logging.INFO)

# List of URLs to load documents from
urls = [
    "https://modelcontextprotocol.io/docs/getting-started/intro",
    "https://modelcontextprotocol.io/docs/learn/architecture",
    "https://modelcontextprotocol.io/docs/learn/server-concepts",
    "https://modelcontextprotocol.io/docs/learn/client-concepts",
    "https://modelcontextprotocol.io/specification/versioning",
    "https://modelcontextprotocol.io/docs/develop/connect-local-servers",
    "https://modelcontextprotocol.io/docs/develop/connect-remote-servers",
    "https://modelcontextprotocol.io/docs/develop/build-server",
    "https://modelcontextprotocol.io/docs/develop/build-client",
    "https://modelcontextprotocol.io/docs/sdk",
    "https://modelcontextprotocol.io/docs/tutorials/security/authorization",
    "https://modelcontextprotocol.io/docs/tools/inspector",
]

# Function to clean content
def clean_content(content: str) -> str:
    """Remove HTML tags, extra whitespace, and small fragments."""
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\b\.\s*\b', ' ', content)
    content = re.sub(r'\n\s*\n', '\n', content)
    return content

def load_documents(urls_to_load: Optional[List[str]] = None) -> List[Document]:
    if urls_to_load is None:
        urls_to_load = urls
    # Limit for testing: keep first N (caller can pass a slice instead)
    test_urls = urls_to_load
    # Try PlaywrightURLLoader
    logging.info("Trying PlaywrightURLLoader...")
    docs: List[Document] = []
    try:
        playwright_loader = PlaywrightURLLoader(
            urls=test_urls,
            wait_until="networkidle",
            remove_selectors=["header", "nav", "footer", ".sidebar"],
        )
        docs = playwright_loader.load()
    except Exception as e:
        logging.warning(f"PlaywrightURLLoader error: {e}")
        logging.info("Falling back to WebBaseLoader...")
        parse_only = SoupStrainer("body")
        web_loader = WebBaseLoader(
            web_paths=test_urls,
            bs_kwargs={"parse_only": parse_only},
            requests_kwargs={
                "headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/91.0.4472.124 Safari/537.36"
                    )
                }
            },
        )
        try:
            docs = web_loader.load()
        except Exception as e2:
            logging.error(f"WebBaseLoader error: {e2}")
            docs = []
    # Clean documents in-place
    for doc in docs:
        if hasattr(doc, "page_content") and doc.page_content:
            doc.page_content = clean_content(doc.page_content)

    logging.info(f"Number of documents loaded: {len(docs)}")
    return docs


def merge_documents(docs: List[Document]) -> List[Document]:
    """Combine all document contents into a single merged document."""
    if not docs:
        return []
    merged_text = "\n\n".join([clean_content(doc.page_content) for doc in docs])
    merged_doc = Document(page_content=merged_text, metadata={"source": "merged_document"})
    return [merged_doc]