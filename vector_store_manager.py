# vector_store_manager.py
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import google.generativeai as genai


class VectorStoreManager:
    def __init__(self, index_name="Dense-index", dimension=384, metric="cosine"):
        load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Configure Google GenAI
        genai.configure(api_key=self.google_api_key)

        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("Hugging Face embeddings loaded successfully!")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            raise

        # Initialize Gemma 3 12B Instruct via Google API
        try:
            print("Initializing Gemma 3 12B Instruct via Google API...")
            self.model_name = "gemma-3-12b-it"
            self.model = genai.GenerativeModel(self.model_name)
            print("Gemma 3 12B Instruct ready using Google API.")
        except Exception as e:
            print(f"Error initializing Gemma model: {e}")
            raise

        self.index = None
        self.vector_store = None

    # ---------- Pinecone setup ----------
    def setup_index(self):
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
            print(f"Deleted existing index: {self.index_name}")

        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Serverless index '{self.index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

        self.index = self.pc.Index(self.index_name)

    # ---------- Vector store ----------
    def create_vector_store(self, documents):
        try:
            self.vector_store = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name,
            )
            print(f"Documents added to Pinecone index '{self.index_name}' successfully.")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def search(self, query, top_k=5):
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call create_vector_store first.")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            raise

    # ---------- Generation ----------
    def generate_answer(self, query, top_k=5):
        try:
            results = self.search(query, top_k=top_k)
            if not results:
                return "No relevant documents found to generate an answer."

            # Merge context for inference
            context = "\n\n".join([
                f"Doc {i+1}: {doc.page_content}"
                for i, (doc, score) in enumerate(results)
            ])[:6000]
            
            prompt = f"""
            You are a precise and factual assistant.
            Use only the provided context to answer factually. 
            If the context is unrelated to the question, reply: "I do not have relevant information."

            Context:
            {context}

            Question:
            {query}

            Answer (respond below this line and keep it concise):
            ###ANSWER_START###
            """
            
           
            '''
            prompt = f"""
                You are a precise and factual assistant that follows the Model Context Protocol (MCP) principles.
                Always answer using only the given context — do not make assumptions or add extra information.

                ### Example
                Context:
                The Model Context Protocol (MCP) defines a standardized way for clients and servers to share and exchange contextual data. 
                It allows models to access relevant resources or tools dynamically via context objects.

                Question:
                What is the primary purpose of the Model Context Protocol (MCP)?

                Answer:
                The Model Context Protocol enables structured communication between clients and servers so that language models can use external tools and data safely and contextually.

                ---

                Now follow the same style for the new input below.

                ### Context:
                {context}

                ### Question:
                {query}

                ### Instruction:
                Answer concisely and accurately **based only on the above context**. 
                If the context does not contain the answer, reply with "The context does not provide that information."
                """
            '''
            
            '''
            prompt = f"""
                You are a precise and factual assistant that follows the Model Context Protocol (MCP) principles.
                Use only the given context to answer; do not add assumptions or add extra information. Think step by step from reasoning steps.

                ### Context:
                {context}

                ### Question:
                {query}

                ### Reasoning Steps:
                1. Carefully read and understand the context.
                2. Identify key facts or statements relevant to the question.
                3. Exclude any information not explicitly present in the context.
                4. If the context lacks sufficient information, conclude that it is unavailable.
                5. Formulate a short, factual answer based strictly on the verified content.

                ### Final Answer:
                Provide your final, concise answer here.
                """
                '''
            

            response = self.model.generate_content(prompt)
            answer = getattr(response, "text", None)
            if not answer and hasattr(response, "candidates"):
                answer = response.candidates[0].content.parts[0].text

            return answer.strip() if answer else "No clear answer generated."
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error generating answer: {e}"







