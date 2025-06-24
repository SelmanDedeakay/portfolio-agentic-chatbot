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
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
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
    """LLM Agent with tool calling capabilities"""
    
    def __init__(self):
        # Initialize tools
        self.tools = {
            "search_cv": CVRAGTool(),
            "send_email": EmailTool()
        }
        
        # Initialize LLM
        self.model_name = "microsoft/DialoGPT-medium"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            self.tokenizer = None
            self.model = None
    
    def get_tool_descriptions(self) -> str:
        """Get descriptions of available tools"""
        return """
Available tools:
1. search_cv(query: str) - Search Selman's CV/resume for information about his background, experience, skills, education, projects
2. send_email(sender_name: str, sender_email: str, subject: str, message: str) - Send an email to Selman for contact/collaboration requests

Tool calling format:
TOOL_CALL: tool_name(param1="value1", param2="value2")
"""
    
    def extract_tool_calls(self, text: str) -> List[Dict]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for tool_name, params_str in matches:
            if tool_name in self.tools:
                try:
                    # Parse parameters
                    params = {}
                    param_pattern = r'(\w+)="([^"]*)"'
                    param_matches = re.findall(param_pattern, params_str)
                    for key, value in param_matches:
                        params[key] = value
                    
                    tool_calls.append({
                        "tool": tool_name,
                        "params": params
                    })
                except Exception as e:
                    logger.error(f"Error parsing tool call: {e}")
        
        return tool_calls
    
    def execute_tool_call(self, tool_call: Dict) -> str:
        """Execute a tool call"""
        tool_name = tool_call["tool"]
        params = tool_call["params"]
        
        try:
            if tool_name == "search_cv":
                return self.tools[tool_name](params.get("query", ""))
            elif tool_name == "send_email":
                return self.tools[tool_name](
                    params.get("sender_name", "Portfolio Visitor"),
                    params.get("sender_email", ""),
                    params.get("subject", "Contact from Portfolio"),
                    params.get("message", "")
                )
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing {tool_name}: {str(e)}"
        
        return f"Unknown tool: {tool_name}"
    
    def generate_response(self, user_message: str) -> str:
        """Generate response with tool calling"""
        system_prompt = f"""You are Selman's portfolio assistant chatbot. Your job is to help visitors learn about Selman and contact him.

{self.get_tool_descriptions()}

Instructions:
- If someone asks about Selman's background, experience, skills, education, or projects, use search_cv tool
- If someone wants to contact Selman or send a message, use send_email tool
- Extract email addresses from user messages when handling contact requests
- Always use tool calls when appropriate, don't try to answer from memory
- Be helpful and professional

User message: {user_message}

Response (use tools when needed):"""

        if self.model and self.tokenizer:
            try:
                encoded = self.tokenizer(system_prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
                inputs = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_length=inputs.shape[1] + 150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                
                # Check for tool calls in response
                tool_calls = self.extract_tool_calls(response)
                
                if tool_calls:
                    # Execute tool calls and format response
                    tool_results = []
                    for tool_call in tool_calls:
                        result = self.execute_tool_call(tool_call)
                        tool_results.append(result)
                    
                    # Generate final response with tool results
                    final_prompt = f"""Based on the tool results below, provide a helpful response to the user.

User question: {user_message}
Tool results: {' '.join(tool_results)}

Response:"""
                    
                    final_encoded = self.tokenizer(final_prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
                    final_inputs = final_encoded['input_ids']
                    final_attention_mask = final_encoded['attention_mask']
                    
                    with torch.no_grad():
                        final_outputs = self.model.generate(
                            final_inputs,
                            attention_mask=final_attention_mask,
                            max_length=final_inputs.shape[1] + 100,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    final_response = self.tokenizer.decode(final_outputs[0][final_inputs.shape[1]:], skip_special_tokens=True)
                    return final_response.strip()
                
                return response.strip()
                
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
        
        # Fallback: Rule-based tool calling
        return self.fallback_response(user_message)
    
    def fallback_response(self, user_message: str) -> str:
        """Fallback response when LLM is not available"""
        message_lower = user_message.lower()
        
        # Check for CV-related queries
        cv_keywords = ['experience', 'background', 'skills', 'education', 'work', 'projects', 'resume', 'cv', 'about', 'who']
        if any(keyword in message_lower for keyword in cv_keywords):
            cv_result = self.tools["search_cv"](user_message)
            return f"Based on Selman's CV:\n\n{cv_result}"
        
        # Check for email intent
        email_keywords = ['email', 'contact', 'reach', 'hire', 'message', 'send']
        if any(keyword in message_lower for keyword in email_keywords):
            # Try to extract email
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
            if email_match:
                email_result = self.tools["send_email"](
                    "Portfolio Visitor",
                    email_match.group(0),
                    "Contact from Portfolio",
                    user_message
                )
                return email_result
            else:
                return "I'd be happy to help you contact Selman! Please provide your email address and message, and I'll send it to him."
        
        # General response
        return "Hello! I'm Selman's portfolio assistant. I can help you learn about his background and experience, or help you contact him. What would you like to know?"

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