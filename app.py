import streamlit as st
import os
import smtplib
import json
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM:
    """Wrapper for Google Gemini API"""
    
    def __init__(self):
        try:
            # Try to get API key from Streamlit secrets first, then environment
            api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.configured = True
                logger.info("Gemini API configured successfully")
            else:
                self.configured = False
                logger.warning("Gemini API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.configured = False
    
    def generate(self, prompt: str, context: str) -> str:
        """Generate response using Gemini"""
        if not self.configured:
            return "Gemini API not configured. Please set GEMINI_API_KEY in your environment or Streamlit secrets."
        
        full_prompt = f"""You are an AI assistant for Selman's portfolio. Your role is to answer questions about Selman based on the provided CV context.

Important instructions:
1. Only use information from the provided context
2. If the answer is not in the context, politely say so and suggest contacting Selman directly
3. Be professional and friendly
4. Keep responses concise but informative
5. When mentioning specific experiences or skills, quote directly from the CV when possible

Context from Selman's CV:
{context}

User Question: {prompt}

Please provide a helpful and accurate response based on the CV context above:"""
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=512,
                    top_p=0.8,
                    top_k=40
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"I apologize, but I encountered an error while generating a response. Please try again."

class CVRAGTool:
    """RAG tool for CV question answering with Gemini"""
    
    def __init__(self, cv_path: str = "selman-cv.pdf"):
        self.cv_path = cv_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cv_chunks = []
        self.cv_embeddings = None
        self.full_text = ""
        self.llm = GeminiLLM()
        self.metadata = {}  # Store metadata about chunks
        self.load_cv()
    
    def load_cv(self):
        """Load and process CV into chunks with metadata"""
        try:
            if os.path.exists(self.cv_path):
                with open(self.cv_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    page_texts = []
                    
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        page_texts.append((i + 1, page_text))
                        text += page_text + "\n"
                
                self.full_text = text
                self.cv_chunks, self.metadata = self._chunk_text_with_metadata(page_texts)
                self.cv_embeddings = self.embedder.encode(self.cv_chunks)
                logger.info(f"Loaded CV with {len(self.cv_chunks)} chunks from {len(page_texts)} pages")
            else:
                logger.warning(f"CV file {self.cv_path} not found")
                self.cv_chunks = ["CV file not available. Please upload selman-cv.pdf to provide information about Selman."]
                self.cv_embeddings = self.embedder.encode(self.cv_chunks)
        except Exception as e:
            logger.error(f"Error loading CV: {e}")
            self.cv_chunks = ["Error loading CV file."]
            self.cv_embeddings = self.embedder.encode(self.cv_chunks)
    
    def _chunk_text_with_metadata(self, page_texts: List[tuple], chunk_size: int = 400, overlap: int = 100) -> tuple:
        """Split text into overlapping chunks with metadata"""
        chunks = []
        metadata = []
        
        for page_num, page_text in page_texts:
            # Clean text
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            
            # Split page into chunks
            start = 0
            while start < len(page_text):
                end = start + chunk_size
                
                # Try to find a natural break point
                if end < len(page_text):
                    # Look for paragraph or sentence endings
                    for delimiter in ['\n\n', '\n', '. ', '! ', '? ']:
                        last_delimiter = page_text.rfind(delimiter, start, end)
                        if last_delimiter != -1 and last_delimiter > start:
                            end = last_delimiter + len(delimiter)
                            break
                
                chunk = page_text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                    metadata.append({
                        'page': page_num,
                        'start_char': start,
                        'end_char': end,
                        'chunk_id': len(chunks) - 1
                    })
                
                # Move start with overlap
                start = end - overlap if end < len(page_text) else end
        
        return chunks, metadata
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks using semantic search"""
        if not self.cv_chunks or self.cv_embeddings is None:
            return [{"text": "CV information not available.", "metadata": {}}]
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.cv_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by similarity threshold and prepare results
        threshold = 0.25
        relevant_chunks = []
        
        for idx in top_indices:
            if similarities[idx] > threshold:
                relevant_chunks.append({
                    'text': self.cv_chunks[idx],
                    'similarity': float(similarities[idx]),
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                })
        
        # Sort by similarity
        relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        return relevant_chunks if relevant_chunks else [{"text": "No relevant information found in CV.", "metadata": {}}]
    
    def __call__(self, query: str) -> str:
        """Perform RAG: retrieve relevant chunks and generate answer using Gemini"""
        # Retrieve relevant chunks with metadata
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        # Create context from chunks with page information
        context_parts = []
        for i, chunk_data in enumerate(relevant_chunks):
            chunk_text = chunk_data['text']
            metadata = chunk_data.get('metadata', {})
            
            if metadata and 'page' in metadata:
                context_parts.append(f"[From page {metadata['page']}]: {chunk_text}")
            else:
                context_parts.append(chunk_text)
        
        context = "\n\n".join(context_parts)
        
        # Add similarity scores for debugging (optional)
        if relevant_chunks and 'similarity' in relevant_chunks[0]:
            logger.info(f"Top chunk similarity: {relevant_chunks[0]['similarity']:.3f}")
        
        # Generate response using Gemini
        response = self.llm.generate(query, context)
        
        return response

class EmailTool:
    """SMTP Gmail email tool"""
    
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        try:
            self.sender_email = st.secrets["GMAIL_EMAIL"]
            self.sender_password = st.secrets["GMAIL_APP_PASSWORD"]
        except:
            self.sender_email = os.getenv("GMAIL_EMAIL", "")
            self.sender_password = os.getenv("GMAIL_APP_PASSWORD", "")
    
    def __call__(self, sender_name: str, sender_email: str, subject: str, message: str) -> str:
        """Send email via Gmail SMTP"""
        if not self.sender_email or not self.sender_password:
            return "Email configuration not available. Please set GMAIL_EMAIL and GMAIL_APP_PASSWORD environment variables."
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.sender_email
            msg['Subject'] = f"Portfolio Contact: {subject}"
            
            body = f"""
New message from portfolio website:

From: {sender_name}
Email: {sender_email}
Subject: {subject}

Message:
{message}

---
This message was sent via your portfolio chatbot.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return "Email sent successfully! Selman will receive your message and get back to you soon."
            
        except Exception as e:
            logger.error(f"Email error: {e}")
            return "Sorry, there was an error sending the email. Please try again later."

class ToolCallingAgent:
    """Agent with RAG and tool calling capabilities using Gemini"""
    
    def __init__(self):
        # Initialize tools
        self.tools = {
            "search_cv": CVRAGTool(),
            "send_email": EmailTool()
        }
        self.llm = GeminiLLM()
    
    def detect_intent(self, user_message: str) -> Dict[str, Any]:
        """Detect user intent from message using pattern matching and keyword analysis"""
        message_lower = user_message.lower()
        
        # Define intent patterns
        cv_keywords = [
            'experience', 'background', 'skills', 'education', 'work', 'projects', 
            'resume', 'cv', 'about', 'who', 'qualification', 'study', 'studied',
            'university', 'degree', 'expertise', 'specializ', 'proficient', 'what',
            'where', 'when', 'how', 'which', 'tell', 'describe', 'explain', 'know',
            'learn', 'working', 'worked', 'role', 'position', 'job', 'career',
            'technical', 'programming', 'language', 'framework', 'tool', 'technology'
        ]
        
        contact_keywords = [
            'email', 'contact', 'reach', 'hire', 'message', 'send', 'get in touch',
            'talk', 'discuss', 'opportunity', 'collaboration', 'connect', 'interested',
            'proposal', 'inquiry', 'question for selman'
        ]
        
        greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
        
        # Check for patterns
        is_question = any(message_lower.startswith(q) for q in ['what', 'where', 'when', 'who', 'how', 'which', 'can', 'could', 'would', 'is', 'are', 'do', 'does', 'tell', 'explain'])
        has_question_mark = '?' in user_message
        
        # Count keyword matches
        cv_score = sum(1 for keyword in cv_keywords if keyword in message_lower)
        contact_score = sum(1 for keyword in contact_keywords if keyword in message_lower)
        greeting_score = sum(1 for keyword in greeting_keywords if keyword in message_lower)
        
        # Email detection
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
        
        # Determine intent
        if greeting_score > 0 and len(message_lower.split()) < 6:
            return {"intent": "greeting", "confidence": 1.0}
        elif email_match or (contact_score > 2):
            return {
                "intent": "contact", 
                "confidence": min(contact_score / 3, 1.0),
                "email": email_match.group(0) if email_match else None
            }
        elif cv_score > 0 or is_question or has_question_mark:
            confidence = min((cv_score / 5) + (0.3 if is_question else 0) + (0.2 if has_question_mark else 0), 1.0)
            return {"intent": "cv_query", "confidence": confidence}
        else:
                       # Default to general/cv_query for ambiguous cases
            return {"intent": "cv_query", "confidence": 0.5}
    
    def generate_response(self, user_message: str) -> str:
        """Generate response based on intent detection and tool calling"""
        try:
            # Detect intent
            intent_result = self.detect_intent(user_message)
            intent = intent_result["intent"]
            confidence = intent_result.get("confidence", 0)
            
            logger.info(f"Detected intent: {intent} (confidence: {confidence:.2f})")
            
            if intent == "greeting":
                responses = [
                    "Hello! I'm Selman's AI assistant powered by Gemini. I can help you learn about his background, experience, and skills from his CV. What would you like to know?",
                    "Hi there! I have access to Selman's CV and can answer questions about his education, work experience, projects, and skills. How can I help you today?",
                    "Hey! Welcome to Selman's portfolio. I can provide detailed information about his professional background using AI-powered search. What interests you?"
                ]
                return responses[hash(user_message) % len(responses)]
            
            elif intent == "cv_query":
                # Use RAG tool for CV queries
                with st.spinner("Searching Selman's CV and generating response..."):
                    response = self.tools["search_cv"](user_message)
                
                # Add a follow-up suggestion if response seems incomplete
                if "not found" in response.lower() or "not available" in response.lower():
                    response += "\n\nWould you like to contact Selman directly for more information? Just let me know and I can help you send a message."
                
                return response
            
            elif intent == "contact":
                email = intent_result.get("email")
                if email:
                    # Extract name if possible
                    name_patterns = [
                        r"(?:i'm|i am|my name is|this is)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)",
                        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+here",
                        r"- ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*$"
                    ]
                    
                    sender_name = "Portfolio Visitor"
                    for pattern in name_patterns:
                        name_match = re.search(pattern, user_message, re.IGNORECASE)
                        if name_match:
                            sender_name = name_match.group(1).strip().title()
                            break
                    
                    # Extract subject/topic if mentioned
                    subject = "Contact from Portfolio"
                    subject_keywords = ['about', 'regarding', 'for', 'discuss', 'opportunity', 'project', 'position', 'role']
                    for keyword in subject_keywords:
                        if keyword in user_message.lower():
                            # Try to extract the part after the keyword
                            pattern = f"{keyword}\\s+(.+?)(?:\\.|,|$)"
                            match = re.search(pattern, user_message.lower())
                            if match:
                                subject = f"Portfolio: {match.group(1).strip().title()}"
                                break
                    
                    # Send email
                    email_result = self.tools["send_email"](
                        sender_name,
                        email,
                        subject,
                        user_message
                    )
                    return email_result
                else:
                    return ("I'd be happy to help you contact Selman! To send him a message, please include:\n"
                           "- Your email address\n"
                           "- Your message or inquiry\n\n"
                           "For example: 'I'd like to contact Selman about a project opportunity. My email is john@example.com'")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request. Please try rephrasing your question or try again later."

# Cache the agent to avoid reinitialization
@st.cache_resource
def load_agent():
    return ToolCallingAgent()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Selman's Portfolio Assistant",
        page_icon="🤖",
        layout="centered"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stChat {
            max-width: 800px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Selman's AI Portfolio Assistant")
    st.caption("Powered by Gemini 1.5 Flash and RAG")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 About this Assistant")
        st.markdown("I'm an AI assistant that can help you learn about Selman's professional background using Retrieval-Augmented Generation (RAG).")
        
        st.markdown("### 💡 What I can do:")
        st.markdown("- 📄 **Answer questions** about Selman's education, experience, and skills")
        st.markdown("- 🔍 **Search his CV** for specific information")
        st.markdown("- 💼 **Explain his projects** and technical expertise")
        st.markdown("- 📧 **Help you contact him** directly")
        
        st.markdown("---")
        
        st.markdown("### 🛠️ How it works:")
        st.markdown("1. I use **semantic search** to find relevant information from Selman's CV")
        st.markdown("2. **Gemini AI** analyzes the context and generates accurate responses")
        st.markdown("3. All answers are based on actual CV content")
        
        st.markdown("---")
        
        # Debug info (optional)
        if st.checkbox("Show debug info"):
            agent = load_agent()
            st.markdown("### Debug Information")
            st.markdown(f"- CV Loaded: {'Yes' if agent.tools['search_cv'].cv_chunks else 'No'}")
            st.markdown(f"- Number of chunks: {len(agent.tools['search_cv'].cv_chunks)}")
            st.markdown(f"- Gemini configured: {'Yes' if agent.llm.configured else 'No'}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_message = (
            "Hello! I'm Selman's AI portfolio assistant, powered by Google's Gemini 1.5 Flash and RAG technology. "
            "I can answer questions about his education, work experience, skills, and projects based on his CV. "
            "What would you like to know?"
        )
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_message
        })
    
    # Load agent
    agent = load_agent()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Selman's background or request to contact him..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            try:
                # Show thinking indicator
                with st.spinner("Thinking..."):
                    response = agent.generate_response(prompt)
                
                # Display response
                st.markdown(response)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                logger.error(f"Chat error: {e}")
    
    # Footer
    st.markdown("---")
    st.caption("💡 Tip: Try asking about specific skills, projects, or educational background!")

if __name__ == "__main__":
    # Check for required packages
    required_packages = {
        'google-generativeai': 'google-generativeai',
        'sentence-transformers': 'sentence-transformers',
        'PyPDF2': 'PyPDF2',
        'scikit-learn': 'scikit-learn',
        'numpy': 'numpy'
    }
    
    # Display installation instructions if needed
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        st.error(f"Missing required packages: {', '.join(missing_packages)}")
        st.code(f"pip install {' '.join(missing_packages)}")
        st.stop()
    
    # Run the app
    main()