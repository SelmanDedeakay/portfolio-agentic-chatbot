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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVRAGTool:
    """RAG tool for CV question answering"""
    
    def __init__(self, cv_path: str = "selman-cv.pdf"):
        self.cv_path = cv_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cv_chunks = []
        self.cv_embeddings = None
        self.load_cv()
    
    def load_cv(self):
        """Load and process CV into chunks"""
        try:
            if os.path.exists(self.cv_path):
                with open(self.cv_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                
                self.cv_chunks = self._chunk_text(text)
                self.cv_embeddings = self.embedder.encode(self.cv_chunks)
                logger.info(f"Loaded CV with {len(self.cv_chunks)} chunks")
            else:
                logger.warning(f"CV file {self.cv_path} not found")
                self.cv_chunks = ["CV file not available. Please upload selman-cv.pdf to provide information about Selman."]
                self.cv_embeddings = self.embedder.encode(self.cv_chunks)
        except Exception as e:
            logger.error(f"Error loading CV: {e}")
            self.cv_chunks = ["Error loading CV file."]
            self.cv_embeddings = self.embedder.encode(self.cv_chunks)
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def __call__(self, query: str) -> str:
        """Search CV for relevant information"""
        if not self.cv_chunks:
            return "CV information not available."
        
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.cv_embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        relevant_chunks = [self.cv_chunks[i] for i in top_indices]
        return "\n\n".join(relevant_chunks)

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
    """Simple Agent with tool calling capabilities"""
    
    def __init__(self):
        # Initialize tools
        self.tools = {
            "search_cv": CVRAGTool(),
            "send_email": EmailTool()
        }
    
    def detect_intent(self, user_message: str) -> Dict[str, Any]:
        """Detect user intent from message"""
        message_lower = user_message.lower()
        
        # Check for CV-related queries
        cv_keywords = ['experience', 'background', 'skills', 'education', 'work', 'projects', 
                       'resume', 'cv', 'about', 'who', 'qualification', 'study', 'studied',
                       'university', 'degree', 'expertise', 'specializ', 'proficient']
        
        # Check for contact intent
        contact_keywords = ['email', 'contact', 'reach', 'hire', 'message', 'send', 'get in touch',
                           'talk', 'discuss', 'opportunity', 'collaboration', 'connect']
        
        # Check for greeting
        greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        
        # Count keyword matches
        cv_score = sum(1 for keyword in cv_keywords if keyword in message_lower)
        contact_score = sum(1 for keyword in contact_keywords if keyword in message_lower)
        greeting_score = sum(1 for keyword in greeting_keywords if keyword in message_lower)
        
        # Determine primary intent
        if cv_score > 0 and cv_score >= contact_score:
            return {"intent": "cv_query", "confidence": cv_score / len(cv_keywords)}
        elif contact_score > 0:
            # Check for email in message
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
            return {
                "intent": "contact", 
                "confidence": contact_score / len(contact_keywords),
                "email": email_match.group(0) if email_match else None
            }
        elif greeting_score > 0:
            return {"intent": "greeting", "confidence": 1.0}
        else:
            # Default to cv_query if unclear
            return {"intent": "general", "confidence": 0.5}
    
    def generate_response(self, user_message: str) -> str:
        """Generate response based on intent detection and tool calling"""
        try:
            # Detect intent
            intent_result = self.detect_intent(user_message)
            intent = intent_result["intent"]
            
            if intent == "greeting":
                responses = [
                    "Hello! I'm Selman's portfolio assistant. How can I help you today?",
                    "Hi there! I can help you learn about Selman's background or help you contact him. What would you like to know?",
                    "Hey! Welcome to Selman's portfolio. I'm here to assist you with any questions about his experience or to help you get in touch."
                ]
                return responses[hash(user_message) % len(responses)]
            
            elif intent == "cv_query":
                # Use CV search tool
                cv_result = self.tools["search_cv"](user_message)
                if "CV file not available" in cv_result:
                    return "I apologize, but I don't have access to Selman's CV at the moment. However, I can help you contact him directly if you'd like to learn more about his background."
                else:
                    return f"Based on Selman's CV, here's what I found:\n\n{cv_result}\n\nIs there anything specific you'd like to know more about?"
            
            elif intent == "contact":
                email = intent_result.get("email")
                if email:
                    # Extract name if possible
                    name_match = re.search(r"(i'm|i am|my name is|this is)\s+([a-z]+)", user_message.lower())
                    sender_name = name_match.group(2).title() if name_match else "Portfolio Visitor"
                    
                    # Send email
                    email_result = self.tools["send_email"](
                        sender_name,
                        email,
                        "Contact from Portfolio",
                        user_message
                    )
                    return email_result
                else:
                    return ("I'd be happy to help you contact Selman! To send him a message, please include:\n"
                           "- Your email address\n"
                           "- Your message\n\n"
                           "For example: 'I'd like to contact Selman. My email is john@example.com and I'm interested in discussing a project.'")
            
            else:  # general intent
                return ("I'm here to help you learn about Selman's background and experience, or help you contact him. "
                       "You can ask me about:\n"
                       "- His education and qualifications\n"
                       "- Work experience and projects\n"
                       "- Technical skills and expertise\n"
                       "- How to get in touch with him\n\n"
                       "What would you like to know?")
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request. Please try rephrasing your question."

# Initialize the agent
@st.cache_resource
def load_agent():
    return ToolCallingAgent()

def main():
    st.set_page_config(
        page_title="Selman's Portfolio Assistant",
        page_icon="💬",
        layout="centered"
    )
    
    st.title("💬 Selman's Portfolio Assistant")
    
    # Add a sidebar with information
    with st.sidebar:
        st.markdown("### About this Assistant")
        st.markdown("I can help you:")
        st.markdown("- 📄 Learn about Selman's background")
        st.markdown("- 💼 Explore his work experience")
        st.markdown("- 🎓 Discover his education")
        st.markdown("- 💻 Understand his technical skills")
        st.markdown("- 📧 Contact him directly")
        
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("Simply type your question or request in the chat!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm Selman's portfolio assistant. I can help you learn about his background and experience, or help you contact him. What would you like to know?"
        })
    
    # Load agent
    agent = load_agent()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Selman or request to contact him..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent.generate_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = "I apologize, but I encountered an error. Please try again."
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Chat error: {e}")

if __name__ == "__main__":
    main()