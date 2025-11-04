# src/app.py
import streamlit as st
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager

def initialize_system():
    """Initialize document processor and vector store manager."""
    try:
        # Initialize document processor and load documents
        doc_processor = DocumentProcessor(chunk_size=600, chunk_overlap=100)
        split_docs = doc_processor.load_and_split_documents()

        # Initialize vector store manager
        vector_manager = VectorStoreManager(index_name="hybrid-index")
        vector_manager.setup_index()
        vector_manager.create_vector_store(split_docs)

        return vector_manager
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return None

def main():
    st.title("MCP Knowledge Assistant")
    st.write("Enter a query here.")

    # Initialize system only once using session state
    if "vector_manager" not in st.session_state:
        with st.spinner("Initializing search system..."):
            st.session_state.vector_manager = initialize_system()

    if st.session_state.vector_manager:
        # Query input
        query = st.text_input("Enter your query:", placeholder="e.g., What is MCP?")
        
        if query:
            with st.spinner("Generating answer..."):
                try:
                    # Generate answer using LLM
                    answer = st.session_state.vector_manager.generate_answer(query, top_k=5)
                    st.subheader("Generated Answer")
                    st.write(answer)

                    # Display retrieved documents in an expander
                    with st.expander("View Retrieved Documents"):
                        results = st.session_state.vector_manager.search(query, top_k=5)
                        if results:
                            for i, (doc, score) in enumerate(results, 1):
                                st.write(f"**Document {i}** (Score: {score:.4f})")
                                st.write(f"Content: {doc.page_content[:200]}...")
                                st.write("---")
                        else:
                            st.info("No documents found.")
                except Exception as e:
                    st.error(f"Error during generation: {e}")

if __name__ == "__main__":
    main()