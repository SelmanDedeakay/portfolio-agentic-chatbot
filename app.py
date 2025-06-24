import streamlit as st
import os
import numpy as np
from typing import List, Dict, Any
import PyPDF2
from google import genai
from google.genai import types
import re

class GeminiEmbeddingRAG:
    """RAG using Gemini embeddings instead of SentenceTransformers"""
    
    def __init__(self, cv_path: str = "selman-cv.pdf"):
        self.cv_path = cv_path
        self.cv_chunks = []
        self.cv_embeddings = []
        
        # Initialize Gemini client
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
            if api_key:
                self.client = genai.Client(api_key=api_key)
                self.configured = True
                st.success("✅ Chatbot configured for embeddings and generation")
            else:
                self.configured = False
                st.error("❌ We are having trouble connecting to Chatbot.")
        except Exception as e:
            st.error(f"❌ Chatbot setup failed: {e}")
            self.configured = False
        
        if self.configured:
            self.load_cv()
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Gemini embedding model"""
        if not self.configured:
            return []
        
        try:
            embeddings = []
            for text in texts:
                # Use Gemini embedding model with correct API
                response = self.client.models.embed_content(
                    model="models/text-embedding-004",
                    contents=[text]  # Pass as list
                )
                embeddings.append(response.embeddings[0].values)
            return embeddings
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return []
    
    def load_cv(self):
        """Load CV and create embeddings"""
        try:
            if os.path.exists(self.cv_path):
                # Extract text from PDF
                with open(self.cv_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                
                if text.strip():
                    # Chunk the text
                    self.cv_chunks = self._chunk_text(text)
                    
                    # Generate embeddings using Gemini
                    with st.spinner(f"Generating embeddings for {len(self.cv_chunks)} chunks..."):
                        self.cv_embeddings = self.get_embeddings(self.cv_chunks)
                    
                    if self.cv_embeddings:
                        st.success("✅ Loaded Selman's recent CV.")         
                    else:
                        st.error("❌ Failed to generate embeddings")
                else:
                    st.error("❌ PDF file is empty or unreadable")
                    self.cv_chunks = ["CV file is empty or unreadable"]
            else:
                st.error(f"❌ CV file '{self.cv_path}' not found. Please upload it to the app directory.")
                
        except Exception as e:
            st.error(f"Error loading CV: {e}")
            self.cv_chunks = ["Error loading CV"]
    
    def _chunk_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """Split text into chunks"""
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # Find natural break
            if end < len(text):
                for delimiter in ['\n\n', '\n', '. ', '! ', '? ']:
                    last_delimiter = text.rfind(delimiter, start, end)
                    if last_delimiter != -1:
                        end = last_delimiter + len(delimiter)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        return chunks
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar chunks using Gemini embeddings"""
        if not self.configured or not self.cv_embeddings:
            return [{"text": "Embeddings not available", "similarity": 0}]
        
        # Get query embedding
        query_embeddings = self.get_embeddings([query])
        if not query_embeddings:
            return [{"text": "Could not process query", "similarity": 0}]
        
        query_embedding = query_embeddings[0]
        
        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.cv_embeddings):
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append({
                "text": self.cv_chunks[i],
                "similarity": similarity,
                "index": i
            })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def generate_response(self, query: str) -> str:
        """Generate response using retrieved context"""
        if not self.configured:
            return "Gemini API not configured"
        
        # Retrieve relevant chunks
        relevant_chunks = self.search_similar_chunks(query)
        
        # Create context
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Generate response
        prompt = f"""You are Selman Dedeakayoğulları's AI portfolio assistant. Answer questions about Selman based on his CV.

Rules:
- His name IS NOT Selman Dedeakayoğulları Jr. It is Selman Dedeakayoğulları.
- Only use information from the provided context
- If information isn't available, say so politely
- Be professional and helpful
- Keep responses informative.
- Use markdown formatting for clarity.

CV Context:
{context}

User Question: {query}

Response:"""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=400
                )
            )
            return response.text if response.text else "No response generated"
        except Exception as e:
            return f"Error generating response: {e}"

# Streamlit App
def main():
    st.set_page_config(
        page_title="Selman DEDEAKAYOĞULLARI Portfolio RAG Chatbot",
        page_icon="🔍",
        layout="centered"
    )
    
    st.title("Welcome!")
    st.caption("I'm Selman's AI portfolio assistant, what would you like to know about him?")
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        with st.spinner("Initializing Chatbot"):
            st.session_state.rag_system = GeminiEmbeddingRAG()
    
    rag_system = st.session_state.rag_system
    
    # Only proceed if configured
    if not rag_system.configured:
        st.error("Please configure GEMINI_API_KEY to continue")
        st.stop()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm here to answer questions about Selman. What would you like to know?"}
        ]
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Selman's background..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching in Selman's CV..."):
                response = rag_system.generate_response(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 🔍 So you are a curious one :)")
        st.markdown("- **Embeddings**: text-embedding-004")
        st.markdown("- **Generation**: gemini-1.5-flash")
        st.markdown("- **Vector dims**: 3072")
        st.markdown("- **Search**: Cosine similarity")
        
        if rag_system.configured and rag_system.cv_chunks:
            st.markdown(f"- **Chunks loaded**: {len(rag_system.cv_chunks)}")
            st.markdown(f"- **Embeddings**: {'✅' if rag_system.cv_embeddings else '❌'}")

if __name__ == "__main__":
    main()