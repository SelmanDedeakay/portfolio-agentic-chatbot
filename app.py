import streamlit as st
import os
import numpy as np
from typing import List, Dict, Any
import PyPDF2
from google import genai
from google.genai import types
import re
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

load_dotenv()

def detect_language_from_messages(messages: List[Dict]) -> str:
    """Detect if user is speaking Turkish based on recent messages"""
    if not messages:
        return "en"
    
    # Check last few user messages for Turkish keywords/patterns
    recent_user_messages = [msg['content'] for msg in messages[-5:] if msg['role'] == 'user']
    turkish_keywords = ['hakkında', 'nedir', 'kimdir', 'nasıl', 'merhaba', 'teşekkür', 'iletişim', 'mesaj', 'gönder']
    
    for message in recent_user_messages:
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in turkish_keywords):
            return "tr"
    
    return "en"

class EmailTool:
    """SMTP email functionality"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = st.secrets.get("GMAIL_EMAIL", os.getenv("GMAIL_EMAIL"))
        self.email_password = st.secrets.get("GMAIL_APP_PASSWORD", os.getenv("GMAIL_APP_PASSWORD"))
        self.recipient_email = st.secrets.get("RECIPIENT_EMAIL", os.getenv("RECIPIENT_EMAIL"))
        
    def send_email(self, sender_name: str, sender_email: str, subject: str, message: str) -> Dict[str, Any]:
        """Send email via SMTP"""
        try:
            # Debug info
            print(f"SMTP Server: {self.smtp_server}")
            print(f"SMTP Port: {self.smtp_port}")
            print(f"Email User: {self.email_user}")
            print(f"Recipient: {self.recipient_email}")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_user  # Use authenticated email as sender
            msg['To'] = self.recipient_email
            msg['Subject'] = subject  # Already includes prefix
            msg['Reply-To'] = sender_email  # Set reply-to as the user's email
            
            # Email body
            body = f"""
New contact from portfolio chatbot:

From: {sender_name}
Email: {sender_email}

Message:
{message}

---
Sent via Portfolio RAG Chatbot
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.set_debuglevel(1)  # Enable debug output
            server.starttls()
            server.login(self.email_user, self.email_password)
            
            # Send the message
            text = msg.as_string()
            server.sendmail(self.email_user, self.recipient_email, text)
            server.quit()
            
            return {
                "success": True,
                "message": f"Email sent successfully to {self.recipient_email}! Selman will get back to you soon."
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                "success": False,
                "message": "Email authentication failed. Please check EMAIL_USER and EMAIL_PASSWORD (use App Password for Gmail)."
            }
        except smtplib.SMTPException as e:
            return {
                "success": False,
                "message": f"SMTP error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send email: {str(e)}"
            }

class GeminiEmbeddingRAG:
    """Enhanced RAG with tool calling for email"""
    
    def __init__(self, cv_path: str = "selman-cv.pdf"):
        self.cv_path = cv_path
        self.cv_chunks = []
        self.cv_embeddings = None
        self.email_tool = EmailTool()
        
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
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using Gemini embedding model"""
        if not self.configured:
            return np.array([])
        
        try:
            embeddings = []
            for text in texts:
                response = self.client.models.embed_content(
                    model="models/text-embedding-004",
                    contents=[text]
                )
                embeddings.append(response.embeddings[0].values)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return np.array([])
    
    def load_cv(self):
        """Load CV and create embeddings"""
        try:
            if os.path.exists(self.cv_path):
                with open(self.cv_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                
                if text.strip():
                    self.cv_chunks = self._chunk_text(text)
                    
                    with st.spinner(f"Generating embeddings for {len(self.cv_chunks)} chunks..."):
                        self.cv_embeddings = self.get_embeddings(self.cv_chunks)
                    
                    if self.cv_embeddings.size > 0:
                        st.success(f"✅ Loaded Selman's recent CV ({len(self.cv_chunks)} chunks)")
                    else:
                        st.error("❌ Failed to generate embeddings")
                else:
                    st.error("❌ PDF file is empty or unreadable")
            else:
                st.error(f"❌ CV file '{self.cv_path}' not found.")
                
        except Exception as e:
            st.error(f"Error loading CV: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 350) -> List[str]:
        """Split text into chunks"""
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            
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
    
    def search_similar_chunks(self, query: str, top_k: int = 4) -> List[Dict]:
        """Search for similar chunks using Gemini embeddings"""
        if not self.configured or self.cv_embeddings is None or self.cv_embeddings.size == 0:
            return [{"text": "Embeddings not available", "similarity": 0}]
        
        query_embedding = self.get_embeddings([query])
        if query_embedding.size == 0:
            return [{"text": "Could not process query", "similarity": 0}]
        
        query_vec = query_embedding[0]
        query_norm = np.linalg.norm(query_vec)
        
        similarities = []
        for i, chunk_vec in enumerate(self.cv_embeddings):
            chunk_norm = np.linalg.norm(chunk_vec)
            if query_norm > 0 and chunk_norm > 0:
                similarity = np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm)
            else:
                similarity = 0
            
            similarities.append({
                "text": self.cv_chunks[i],
                "similarity": float(similarity),
                "index": i
            })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def _get_tool_definitions(self) -> List[Any]:
        """Define available tools for function calling"""
        from google.genai.types import Tool, FunctionDeclaration
        
        prepare_email_func = FunctionDeclaration(
            name="prepare_email",
            description="Prepare an email to Selman when someone wants to contact him. This function prepares the email for review before sending.",
            parameters={
                "type": "object",
                "properties": {
                    "sender_name": {
                        "type": "string",
                        "description": "Name of the person sending the email"
                    },
                    "sender_email": {
                        "type": "string",
                        "description": "Email address of the sender"
                    },
                    "message": {
                        "type": "string",
                        "description": "The message content to send to Selman"
                    }
                },
                "required": ["sender_name", "sender_email", "message"]
            }
        )
        
        return [Tool(function_declarations=[prepare_email_func])]
    
    def _execute_tool(self, tool_name: str, tool_args: Dict) -> Dict[str, Any]:
        """Execute the requested tool"""
        if tool_name == "prepare_email":
            # Add default subject
            tool_args['subject'] = "New Message from Portfolio Bot"
            # Store email data for verification
            st.session_state.pending_email = tool_args
            return {
                "success": True,
                "message": "Email prepared for review",
                "data": tool_args
            }
        else:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}"
            }
    
    def generate_response(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Generate response with tool calling capability and Turkish support"""
        if not self.configured:
            return "Gemini API not configured"
        
        # Get conversation context for better email handling
        recent_context = ""
        if conversation_history and len(conversation_history) > 1:
            recent_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-4:]])
        
        # Regular RAG response
        relevant_chunks = self.search_similar_chunks(query)
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Enhanced prompt with Turkish support and smart email handling
        prompt = f"""You are Selman Dedeakayoğulları's AI portfolio assistant. You can respond in both English and Turkish based on the user's language preference.

Rules:
- His name IS NOT Selman Dedeakayoğulları Jr. It is Selman Dedeakayoğulları.
- Respond in the same language as the user's query
- Only use information from the provided context for CV questions
- Be professional and helpful
- Use markdown formatting for clarity

TOOL USAGE:
- Use prepare_email tool when someone wants to contact Selman and you have ALL required information
- Only ask for: sender name, sender email, and message content
- DO NOT ask for email subject - it will be automatically set
- Extract information naturally from conversation context
- If someone wants to contact Selman but you don't have complete info, ask for missing details conversationally
- Don't repeat requests for information already provided in the conversation

Recent Conversation Context:
{recent_context}

CV Context:
{context}

User Question: {query}

Response:"""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=400,
                    tools=self._get_tool_definitions()
                )
            )
            
            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Execute the prepare_email tool
                        tool_name = part.function_call.name
                        tool_args = {k: v for k, v in part.function_call.args.items()}
                        
                        result = self._execute_tool(tool_name, tool_args)
                        
                        if result["success"] and tool_name == "prepare_email":
                            # Return a message indicating email is ready for review
                            return "EMAIL_PREPARED_FOR_REVIEW"
            
            return response.text if response.text else "No response generated"
            
        except Exception as e:
            return f"Error generating response: {e}"

def get_ui_text(language: str) -> Dict[str, str]:
    """Get UI text based on language"""
    if language == "tr":
        return {
            "email_review_title": "📧 **Lütfen e-postanızı göndermeden önce kontrol edin:**",
            "from_label": "**Gönderen:**",
            "email_label": "**E-posta:**",
            "message_label": "**Mesaj:**",
            "send_button": "✅ E-postayı Gönder",
            "cancel_button": "❌ İptal Et",
            "edit_button": "✏️ Mesajı Düzenle",
            "edit_title": "✏️ **E-postanızı düzenleyin:**",
            "name_field": "Adınız",
            "email_field": "E-posta Adresiniz",
            "message_field": "Mesaj",
            "save_button": "💾 Değişiklikleri Kaydet",
            "cancel_edit_button": "❌ Düzenlemeyi İptal Et",
            "email_sent": "✅ E-posta başarıyla gönderildi! Selman size yakında dönüş yapacak.",
            "email_failed": "❌ E-posta gönderilemedi: ",
            "email_cancelled": "E-posta iptal edildi. Başka bir konuda yardımcı olabileceğim bir şey var mı?",
            "email_prepared": "E-postanız Selman'a hazırlandı. Lütfen göndermeden önce aşağıdaki detayları kontrol edin."
        }
    else:  # English
        return {
            "email_review_title": "📧 **Please review your email before sending:**",
            "from_label": "**From:**",
            "email_label": "**Email:**",
            "message_label": "**Message:**",
            "send_button": "✅ Send Email",
            "cancel_button": "❌ Cancel",
            "edit_button": "✏️ Edit Message",
            "edit_title": "✏️ **Edit your email:**",
            "name_field": "Your Name",
            "email_field": "Your Email",
            "message_field": "Message",
            "save_button": "💾 Save Changes",
            "cancel_edit_button": "❌ Cancel Editing",
            "email_sent": "✅ Email sent successfully! Selman will get back to you soon.",
            "email_failed": "❌ Failed to send email: ",
            "email_cancelled": "Email cancelled. Is there anything else I can help you with?",
            "email_prepared": "I've prepared your email to Selman. Please review the details below before sending."
        }

def render_email_verification_card(email_data: Dict[str, str], language: str):
    """Render email verification card within the chat message"""
    ui_text = get_ui_text(language)
    
    with st.container():
        st.info(ui_text["email_review_title"])
        
        # Display email details in a nice format
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(ui_text["from_label"])
            st.markdown(ui_text["email_label"])
            st.markdown(ui_text["message_label"])
        
        with col2:
            st.markdown(f"{email_data['sender_name']}")
            st.markdown(f"{email_data['sender_email']}")
            st.markdown(f"{email_data['message']}")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button(ui_text["send_button"], type="primary", key="send_email_btn"):
                st.session_state.email_action = "send"
                st.rerun()
        
        with col2:
            if st.button(ui_text["cancel_button"], key="cancel_email_btn"):
                st.session_state.email_action = "cancel"
                st.rerun()
        
        with col3:
            if st.button(ui_text["edit_button"], key="edit_email_btn"):
                st.session_state.email_action = "edit"
                st.rerun()

def render_email_editor_card(email_data: Dict[str, str], language: str):
    """Render email editor card within the chat message"""
    ui_text = get_ui_text(language)
    
    with st.container():
        st.info(ui_text["edit_title"])
        
        # Editable fields
        with st.form("email_editor", clear_on_submit=False):
            sender_name = st.text_input(ui_text["name_field"], value=email_data['sender_name'])
            sender_email = st.text_input(ui_text["email_field"], value=email_data['sender_email'])
            message = st.text_area(ui_text["message_field"], value=email_data['message'], height=150)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.form_submit_button(ui_text["save_button"], type="primary"):
                    # Update email data
                    st.session_state.pending_email = {
                        'sender_name': sender_name,
                        'sender_email': sender_email,
                        'subject': 'New Message from Portfolio Bot',
                        'message': message
                    }
                    st.session_state.editing_email = False
                    st.session_state.email_action = None
                    st.rerun()
            
            with col2:
                if st.form_submit_button(ui_text["cancel_edit_button"]):
                    st.session_state.editing_email = False
                    st.session_state.email_action = None
                    st.rerun()

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
    
    # Check email configuration
    if not rag_system.email_tool.email_user or not rag_system.email_tool.email_password:
        st.warning("⚠️ Email functionality is not configured. Please set EMAIL_USER and EMAIL_PASSWORD environment variables.")
    
    # Detect language from conversation
    language = detect_language_from_messages(st.session_state.get("messages", []))
    ui_text = get_ui_text(language)
    
    # Handle email actions
    if "email_action" in st.session_state and st.session_state.email_action:
        if st.session_state.email_action == "send":
            # Send the email
            email_data = st.session_state.pending_email
            with st.spinner("Sending email..." if language == "en" else "E-posta gönderiliyor..."):
                result = rag_system.email_tool.send_email(
                    email_data['sender_name'],
                    email_data['sender_email'],
                    email_data['subject'],
                    email_data['message']
                )
            
            # Clear pending email and action
            del st.session_state.pending_email
            del st.session_state.email_action
            
            # Add result to messages
            if result["success"]:
                message_content = ui_text["email_sent"]
            else:
                message_content = ui_text["email_failed"] + result['message']
            
            st.session_state.messages.append({"role": "assistant", "content": message_content})
            st.rerun()
        
        elif st.session_state.email_action == "cancel":
            del st.session_state.pending_email
            del st.session_state.email_action
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ui_text["email_cancelled"]
            })
            st.rerun()
        
        elif st.session_state.email_action == "edit":
            st.session_state.editing_email = True
            del st.session_state.email_action
            st.rerun()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm here to answer questions about Selman. What would you like to know? I can also help you get in touch with him directly if needed! 📧\n\nMerhaba! Selman hakkında sorularınızı yanıtlayabilirim. Onunla doğrudan iletişime geçmenize de yardımcı olabilirim! 📧"}
        ]
    
    # Display messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Check if this is the last message and we have a pending email
            if (i == len(st.session_state.messages) - 1 and 
                message.get("content") in [ui_text["email_prepared"], "I've prepared your email to Selman. Please review the details below before sending."] and
                "pending_email" in st.session_state):
                
                st.write(message["content"])
                
                # Show email card within the message
                if st.session_state.get("editing_email", False):
                    render_email_editor_card(st.session_state.pending_email, language)
                else:
                    render_email_verification_card(st.session_state.pending_email, language)
            else:
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Selman's background or request to contact him..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Detect language again after new message
        language = detect_language_from_messages(st.session_state.messages)
        ui_text = get_ui_text(language)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..." if language == "en" else "İsteğiniz işleniyor..."):
                response = rag_system.generate_response(prompt, st.session_state.messages)
            
            # Check if email was prepared
            if response == "EMAIL_PREPARED_FOR_REVIEW":
                # Show a message and the email card
                message = ui_text["email_prepared"]
                st.write(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
                
                # Show the email verification card
                if "pending_email" in st.session_state:
                    render_email_verification_card(st.session_state.pending_email, language)
            else:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 🔍 So you are a curious one :)")
        st.markdown("- **Embeddings**: text-embedding-004")
        st.markdown("- **Generation**: gemini-2.0-flash-exp")
        st.markdown("- **Vector dims**: 3072")
        st.markdown("- **Search**: Cosine similarity")
        st.markdown("- **Tools**: Email with verification 📧")
        
        if rag_system.configured and rag_system.cv_chunks:
            st.markdown(f"- **Chunks loaded**: {len(rag_system.cv_chunks)}")
            st.markdown(f"- **Embeddings**: {'✅' if rag_system.cv_embeddings is not None else '❌'}")
        

if __name__ == "__main__":
    main()