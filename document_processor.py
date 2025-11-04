# document_processor.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Scrapping_data import load_documents, merge_documents

class DocumentProcessor:
    # Initialize the document processor
    def __init__(self, chunk_size=300, chunk_overlap=80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            is_separator_regex=False,
            keep_separator=False,
            length_function=len
        )

    # Load and split documents
    def load_and_split_documents(self):
        """Load and split documents into chunks."""
        documents = load_documents()
        merged_docs = merge_documents(documents)
        split_docs = self.text_splitter.split_documents(merged_docs)
        print(f"Number of documents after splitting: {len(split_docs)}")
        return split_docs