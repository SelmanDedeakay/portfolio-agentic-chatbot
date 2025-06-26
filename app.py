# app.py

import streamlit as st
import os
import numpy as np
from typing import List, Dict, Any
from google import genai
from google.genai import types
import re
from dotenv import load_dotenv
import json

# Import ayrıştırılmış araçlar ve bileşenler
from tools.email_tool import EmailTool
from tools.social_media_tool import SocialMediaAggregator
from tools.tool_definitions import ToolDefinitions
from ui.email_components import get_ui_text, render_email_verification_card, render_email_editor_card

load_dotenv()


def detect_language_from_messages(messages: List[Dict]) -> str:
    """Geliştirilmiş dil tespiti - her mesajdan anlık tespit yapar"""
    if not messages:
        return "en"
    
    # SON KULLANICI MESAJINI AL (anlık tespit için)
    last_user_message = None
    for msg in reversed(messages):
        if msg['role'] == 'user':
            last_user_message = msg['content']
            break
    
    if not last_user_message:
        return "en"
    
    message_lower = last_user_message.lower().strip()
    
    # Çok kısa mesajlar için özel kontrol
    if len(message_lower) <= 3:
        if message_lower in ['hi', 'hey']:
            return "en"
        elif message_lower in ['selam', 'merhaba']:
            return "tr"
    
    # Türkçe anahtar kelimeler (genişletilmiş)
    turkish_keywords = [
        'hakkında', 'nedir', 'kimdir', 'nasıl', 'merhaba', 'teşekkür', 'iletişim', 'mesaj', 'gönder',
        'anlat', 'söyle', 'nerede', 'ne zaman', 'hangi', 'proje', 'projeler', 'deneyim', 'eğitim',
        'çalışma', 'iş', 'üniversite', 'okul', 'mezun', 'değil', 'yok', 'var', 'olan', 'yapan',
        'merhabalar', 'selam', 'günaydın', 'teşekkürler', 'sağol', 'kariyer', 'bilgi', 'selamlar',
        'anladım', 'bilmiyorum', 'istiyorum', 'isterim', 've', 'bir', 'bu', 'şu', 'o', 'ben', 'sen',
        'ile', 'için', 'ama', 'fakat', 'lakin', 'çünkü', 'ki', 'da', 'de', 'ta', 'te'
    ]
    
    # İngilizce belirleyici kelimeler
    english_keywords = [
        'hello', 'hi', 'what', 'who', 'when', 'where', 'why', 'how', 'about', 'thank', 'thanks',
        'tell', 'show', 'project', 'experience', 'work', 'education', 'university', 'job', 'i', 'you',
        'know', 'dont', "don't", 'want', 'need', 'can', 'could', 'would', 'should', 'the', 'and',
        'with', 'for', 'but', 'because', 'that', 'this', 'they', 'we', 'he', 'she', 'it', 'my', 'your'
    ]
    
    turkish_score = 0
    english_score = 0
    
    # Türkçe karakterler kontrolü (çok güçlü gösterge)
    if any(char in message_lower for char in ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü']):
        turkish_score += 10
    
    # Özel durumlar - tam eşleşme
    if message_lower in ['selamlar', 'selam', 'merhaba', 'merhabalar']:
        return "tr"
    elif message_lower in ['hello', 'hi', 'hey']:
        return "en"
        
    # "I dont know" gibi özel İngilizce ifadeler
    if any(phrase in message_lower for phrase in ["i dont", "i don't", "i want", "i need", "i can", "i would"]):
        return "en"
    
    # Türkçe özel ifadeler
    if any(phrase in message_lower for phrase in ["bilmiyorum", "istiyorum", "yapabilir", "söyleyebilir"]):
        return "tr"
    
    # Anahtar kelime sayımı
    for keyword in turkish_keywords:
        if keyword in message_lower:
            turkish_score += 2
    
    for keyword in english_keywords:
        if keyword in message_lower:
            english_score += 1
    
    # Eğer hiç puan yoksa, İngilizce default
    if turkish_score == 0 and english_score == 0:
        return "en"
    
    return "tr" if turkish_score > english_score else "en"


class GeminiEmbeddingRAG:
    """Enhanced RAG with tool calling for email using JSON data"""
    
    def __init__(self, json_path: str = "selman-cv.json"):
        self.json_path = json_path
        self.cv_data = {}
        self.cv_chunks = []
        self.cv_embeddings = None
        self.email_tool = EmailTool()
        self.tool_definitions = ToolDefinitions()
        self.social_media_aggregator = SocialMediaAggregator()
        
        # Initialize Gemini client
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
            if api_key:
                self.client = genai.Client(api_key=api_key)
                self.configured = True

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
    
    def json_to_chunks(self, data: Dict) -> List[str]:
        """Convert JSON data to searchable text chunks"""
        chunks = []
        
        # Basic information chunk
        basic_info = f"""Name: {data['name']}
    Title: {data['title']}
    Location: {data['location']}
    Email: {data['email']}
    Phone: {data['phone']}
    Profile: {data['profile']}"""
        chunks.append(basic_info)
        
        # Links chunk
        links_text = "Links and Social Media:\n"
        for platform, url in data['links'].items():
            links_text += f"- {platform.capitalize()}: {url}\n"
        chunks.append(links_text)
        
        # Education chunks - HER BİRİ İÇİN AYRI CHUNK
        for edu in data['education']:
            edu_text = f"Education / Eğitim: {edu.get('institution', 'N/A')}\n"
            
            # Degree veya program bilgisi
            degree_info = edu.get('degree') or edu.get('program', 'N/A')
            edu_text += f"Degree/Program/Derece: {degree_info}\n"
            
            # Yıl bilgisi - years veya year
            year_info = edu.get('years') or edu.get('year', 'N/A')
            edu_text += f"Years/Duration/Süre: {year_info}\n"
            
            # Lokasyon bilgisi - güvenli erişim
            location_info = edu.get('location', 'N/A')
            edu_text += f"Location/Konum: {location_info}\n"
            
            # Üyelikler varsa ekle
            if 'memberships' in edu and edu['memberships']:
                edu_text += f"Memberships/Üyelikler: {', '.join(edu['memberships'])}\n"
            
            # Anahtar kelimeler ekle - daha kapsamlı
            keywords = [
                "education", "eğitim", "university", "üniversite", "degree", "derece", 
                "diploma", "bachelor", "lisans", "graduate", "mezun", "student", "öğrenci",
                edu.get('institution', '').lower().replace(' ', '_')
            ]
            edu_text += f"Keywords: {', '.join(keywords)}"
            
            chunks.append(edu_text)

        # Tüm eğitim bilgilerini özetleyen ek chunk
        all_education_text = "Complete Education Background / Tüm Eğitim Geçmişi:\n"
        for i, edu in enumerate(data['education'], 1):
            degree_info = edu.get('degree') or edu.get('program', 'Program')
            year_info = edu.get('years') or edu.get('year', '')
            all_education_text += f"{i}. {degree_info} - {edu.get('institution', 'N/A')} ({year_info})\n"

        all_education_text += "\nKeywords: complete education, tüm eğitim, educational background, eğitim geçmişi, all degrees, tüm dereceler"
        chunks.append(all_education_text)
        
        # Experience chunks - DAHA DETAYLI VE ANAHTAR KELİMELER EKLE
        for exp in data['experience']:
            exp_text = f"""Work Experience / İş Deneyimi:
    Position/Pozisyon: {exp['title']}
    Company/Şirket: {exp['company']}
    Duration/Süre: {exp['duration']}
    Job Description/İş Tanımı: {exp['description']}
    Keywords: work experience, iş deneyimi, {exp['company'].lower()}, {exp['title'].lower()}"""
            chunks.append(exp_text)
        
        # Tüm deneyimleri tek bir chunk'ta da topla
        all_exp_text = "All Work Experience / Tüm İş Deneyimleri:\n"
        for exp in data['experience']:
            all_exp_text += f"- {exp['title']} at {exp['company']} ({exp['duration']})\n"
        chunks.append(all_exp_text)
        
        # Skills chunk
        skills_text = "Technical Skills:\n"
        for category, skills in data['skills'].items():
            skills_text += f"{category}: {', '.join(skills)}\n"
        chunks.append(skills_text)
        
        # Projects chunks - DAHA DETAYLI VE ANAHTAR KELİMELER EKLE
        for project in data['projects']:
            proj_text = f"""Project / Proje:
    Project Name/Proje Adı: {project['name']}
    Technology Used/Kullanılan Teknoloji: {project['technology']}
    Project Description/Proje Açıklaması: {project['description']}
    Keywords: project, proje, {project['technology'].lower()}, {project['name'].lower()}"""
            if 'link' in project:
                proj_text += f"\nProject Link/Proje Linki: {project['link']}"
            chunks.append(proj_text)
        
        # Tüm projeleri tek bir chunk'ta da topla
        all_proj_text = "All Projects / Tüm Projeler:\n"
        for project in data['projects']:
            all_proj_text += f"- {project['name']} ({project['technology']})\n"
        chunks.append(all_proj_text)
        
        # Awards chunks
        for award in data['awards']:
            award_text = f"Award: {award['name']}\n"
            award_text += f"Organization: {award['organization']}\n"
            award_text += f"Description: {award['description']}"
            chunks.append(award_text)
        
        # Languages chunk
        lang_text = "Languages:\n"
        for lang, level in data['languages'].items():
            lang_text += f"- {lang}: {level}\n"
        chunks.append(lang_text)
        
        # Organizations chunk
        for org in data['organizations']:
            org_text = f"Organization: {org['name']}\n"
            org_text += f"Role: {org['role']}\n"
            org_text += f"Duration: {org['duration']}"
            chunks.append(org_text)
        
        # References chunks
        ref_text = "References:\n"
        for ref in data['references']:
            ref_text += f"- {ref['name']} ({ref['title']} at {ref['organization']})"
        chunks.append(ref_text)
        
        return chunks
        
    def load_cv(self):
        """Load CV from JSON and create embeddings"""
        try:
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r', encoding='utf-8') as file:
                    self.cv_data = json.load(file)
                
                if self.cv_data:
                    self.cv_chunks = self.json_to_chunks(self.cv_data)
                    
                    with st.spinner(f"Generating embeddings for {len(self.cv_chunks)} chunks..."):
                        self.cv_embeddings = self.get_embeddings(self.cv_chunks)
                    
                    if self.cv_embeddings.size > 0:

                        
                        # Initialize job compatibility analyzer with CV data
                        self.tool_definitions.initialize_job_analyzer(self.client, self.cv_data,self)
                        
                    else:
                        st.error("❌ Failed to generate embeddings")
                else:
                    st.error("❌ JSON file is empty or unreadable")
            else:
                st.error(f"❌ CV file '{self.json_path}' not found.")
                
        except Exception as e:
            st.error(f"Error loading CV: {e}")
    
    def search_similar_chunks(self, query: str, top_k: int = 6) -> List[Dict]:
        """Enhanced search with keyword matching"""
        if not self.configured or self.cv_embeddings is None or self.cv_embeddings.size == 0:
            return [{"text": "Embeddings not available", "similarity": 0}]
        
        # Sorguyu normalize et
        query_lower = query.lower()
        
        # Anahtar kelime eşleştirmesi için özel ağırlıklar
        keyword_boosts = {
        'proje': ['project', 'proje'],
        'projects': ['project', 'proje'],
        'deneyim': ['experience', 'deneyim', 'work', 'iş'],
        'experience': ['experience', 'deneyim', 'work', 'iş'],
        'work': ['experience', 'deneyim', 'work', 'iş'],
        'iş': ['experience', 'deneyim', 'work', 'iş'],
        'çalış': ['experience', 'deneyim', 'work', 'iş'],
        # EĞİTİM ANAHTAR KELİMELERİ - EXPANDED
        'eğitim': ['education', 'eğitim', 'university', 'üniversite', 'degree', 'derece'],
        'education': ['education', 'eğitim', 'university', 'üniversite', 'degree', 'derece'],
        'university': ['education', 'eğitim', 'university', 'üniversite'],
        'üniversite': ['education', 'eğitim', 'university', 'üniversite'],
        'okul': ['education', 'eğitim', 'school', 'university', 'üniversite'],
        'mezun': ['graduate', 'education', 'eğitim', 'degree', 'diploma'],
        'diploma': ['degree', 'diploma', 'education', 'eğitim'],
        'lisans': ['bachelor', 'degree', 'education', 'eğitim'],
        'bachelor': ['bachelor', 'lisans', 'degree', 'education'],
        'erasmus': ['erasmus', 'exchange', 'değişim'],
        'eskişehir': ['eskişehir', 'technical', 'university'],
        'vorarlberg': ['vorarlberg', 'austria', 'erasmus'],
    }
        
        # Embedding hesapla
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
            
            # Keyword boost - anahtar kelime varsa skoru artır
            chunk_lower = self.cv_chunks[i].lower()
            boost = 0
            for key, keywords in keyword_boosts.items():
                if key in query_lower:
                    for keyword in keywords:
                        if keyword in chunk_lower:
                            boost += 0.2  # %20 boost
                            break
            
            similarities.append({
                "text": self.cv_chunks[i],
                "similarity": float(similarity + boost),
                "index": i
            })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def generate_response(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Generate response with tool calling capability and Turkish support"""
        if not self.configured:
            return "Gemini API not configured"
        
        # Dil tespiti - hem sorgu hem de konuşma geçmişi
        language = detect_language_from_messages((conversation_history or []) + [{"role": "user", "content": query}])
        
        # Get conversation context for better tool handling
        recent_context = ""
        if conversation_history and len(conversation_history) > 1:
            recent_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-4:]])
        
        # Detect query types
        query_lower = query.lower()
        social_keywords = ['post', 'article', 'medium', 'linkedin', 'social media', 'paylaşım', 'makale', 'yazı']
        is_social_query = any(keyword in query_lower for keyword in social_keywords)
        
        # Job compatibility keywords
        job_keywords = ['job', 'position', 'role', 'hiring', 'recruit', 'vacancy', 'opening', 'career', 'employment', 
                    'iş', 'pozisyon', 'işe alım', 'kariyer', 'istihdam', 'açık pozisyon']
        is_job_query = any(keyword in query_lower for keyword in job_keywords)
        
        # Regular RAG search (still important for CV info)
        is_project_query = any(word in query_lower for word in ['proje', 'project', 'yaptığı', 'geliştirdiği'])
        is_experience_query = any(word in query_lower for word in ['deneyim', 'experience', 'çalış', 'work', 'iş'])
        
        # Eğitim sorularını tespit etmek için
        is_education_query = any(word in query_lower for word in [
            'eğitim', 'education', 'university', 'üniversite', 'okul', 'school', 
            'mezun', 'graduate', 'diploma', 'degree', 'lisans', 'bachelor',
            'erasmus', 'exchange', 'öğrenci', 'student'
        ])
        
        # top_k değerini eğitim sorularında artır
        top_k = 8 if is_education_query else (6 if (is_project_query or is_experience_query) else 4)
        relevant_chunks = self.search_similar_chunks(query, top_k=top_k)
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Dile göre prompt oluştur
        if language == "tr":
            prompt = f"""Sen Selman Dedeakayoğulları'nın yapay zeka portföy asistanısın. Onun portföy web sitesinde gömülüsün. Ziyaretçiler sana sorular soracak.

    KURALLAR:
    - SADECE TÜRKÇE yanıt ver
    - Sadece verilen bağlam bilgilerini kullan
    - Profesyonel ve yardımsever ol
    - Markdown formatını kullan
    - Referans sorulduğunda göster ve iletişim bilgilerinin talep üzerine verildiğini belirt
    - Proje veya deneyim sorularında, bağlamdan tüm ilgili öğeleri listele
    - Proje sorularında: proje adları, kullanılan teknolojiler ve açıklamalar dahil et
    - Deneyim sorularında: şirket adları, pozisyonlar, süreler ve açıklamalar dahil et
    - SELMAN MEZUN OLMUŞTUR

    ARAÇ KULLANIMI:
    - Biri Selman'la iletişime geçmek istediğinde ve tüm bilgiler mevcutsa prepare_email aracını kullan
    - Son paylaşımlar, makaleler sorulduğunda get_recent_posts kullan
    - İş ilanı verildiğinde analyze_job_compatibility kullan
    - PDF istendiğinde generate_compatibility_pdf kullan

    Son Konuşma Bağlamı:
    {recent_context}

    CV Bağlamı:
    {context}

    Kullanıcı Sorusu: {query}

    Yanıt:"""
        else:
            prompt = f"""You are Selman Dedeakayoğulları's AI portfolio assistant. You are embedded in his portfolio website. Visitors will ask questions to you.

    Rules:
    - Respond ONLY in ENGLISH
    - Only use information from the provided context for CV questions
    - Be professional and helpful
    - Use markdown formatting for clarity and readability
    - If the user asks for references, display them and add a note that contact information is available upon request
    - When asked about projects or work experience, list ALL relevant items from the context
    - For project questions, include project names, technologies used, and descriptions. Do not give links unless asked specifically. When talking about "Agentic Portfolio Bot" make a joke about it, since it is you.
    - For experience questions, include company names, positions, durations, and descriptions
    - SELMAN IS GRADUATED

    TOOL USAGE:
    - Use prepare_email tool when someone wants to contact Selman and you have ALL required information
    - Use get_recent_posts tool when someone asks about Selman's recent posts, articles, Medium content, LinkedIn activity, or social media
    - Use analyze_job_compatibility tool when someone provides a job description or asks about fit for a specific role
    - Use generate_compatibility_pdf tool when user asks for PDF, download, or wants to save the job compatibility report

    Recent Conversation Context:
    {recent_context}

    CV Context:
    {context}

    User Question: {query}

    Let's work this out in a step-by-step way to be sure we have the right answer.
    Response:"""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1200,
                    tools=self.tool_definitions.get_all_tools()
                )
            )
            
            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        tool_name = part.function_call.name
                        tool_args = {k: v for k, v in part.function_call.args.items()}
                        
                        result = self.tool_definitions.execute_tool(tool_name, tool_args)
                        
                        if result["success"]:
                            if tool_name == "prepare_email":
                                return "EMAIL_PREPARED_FOR_REVIEW"
                            elif tool_name == "get_recent_posts":
                                return result["data"]["formatted_response"]
                            elif tool_name == "analyze_job_compatibility":
                                st.session_state.last_compatibility_report = result["data"]["report"]
                                st.session_state.last_job_title = result["data"]["job_title"]
                                
                                # Dile göre PDF mesajı
                                if language == "tr":
                                    pdf_msg = "\n\n📄 *Bu raporun PDF versiyonunu indirmek isterseniz söyleyebilirsiniz!*"
                                else:
                                    pdf_msg = "\n\n📄 *You can ask for a PDF version of this report if you'd like to download it!*"
                                
                                return result["data"]["report"] + pdf_msg
                            elif tool_name == "generate_compatibility_pdf":
                                return "PDF_GENERATED"
                        else:
                            return f"❌ {result['message']}"
                    
            return response.text if response.text else "No response generated"
            
        except Exception as e:
            if language == "tr":
                return f"Yanıt oluşturulurken hata: {e}"
            else:
                return f"Error generating response: {e}. The API response might have been empty or invalid."

# ... (rest of the app.py file remains the same)
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
            {"role": "assistant", "content": "Hello! I'm here to answer questions about Selman. What would you like to know? I can also help you get in touch with him directly if needed! 📧\n\nI can also analyze job compatibility if you have a job description you'd like me to review against Selman's profile! 💼\n\nMerhaba! Selman hakkında sorularınızı yanıtlayabilirim. Onunla doğrudan iletişime geçmenize de yardımcı olabilirim! 📧\n\nAyrıca bir iş ilanınız varsa, Selman'ın profiliyle uyumluluğunu analiz edebilirim! 💼"}
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
    if prompt := st.chat_input("Ask about Selman's background, request to contact him, or paste a job description for compatibility analysis..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # HER YENİ MESAJDA DİL TESPİTİ YAP (bu satır çok önemli)
        language = detect_language_from_messages(st.session_state.messages)
        ui_text = get_ui_text(language)
        
        # Generate response
        with st.chat_message("assistant"):
            # Dile göre spinner mesajı
            spinner_msg = "İsteğiniz işleniyor..." if language == "tr" else "Processing your request..."
            with st.spinner(spinner_msg):
                response = rag_system.generate_response(prompt, st.session_state.messages)
            
            # Check if email was prepared
            if response == "EMAIL_PREPARED_FOR_REVIEW":
                message = ui_text["email_prepared"]
                st.write(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
                if "pending_email" in st.session_state:
                    render_email_verification_card(st.session_state.pending_email, language)
            elif response == "PDF_GENERATED":
                # Dile göre PDF mesajı
                if language == "tr":
                    message = "✅ PDF raporu başarıyla oluşturuldu! Aşağıdaki butona tıklayarak indirebilirsiniz."
                else:
                    message = "✅ PDF report generated successfully! You can download it using the button below."
                st.write(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
            else:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    if "pdf_data" in st.session_state and "pdf_filename" in st.session_state:
        pdf_data = st.session_state.pdf_data
        pdf_filename = st.session_state.pdf_filename
        
        # Create download button that clears data after download
        st.download_button(
            label="📄 Download PDF Report / PDF Raporu İndir",
            data=pdf_data,
            file_name=pdf_filename,
            mime="application/pdf",
            key="download_pdf",
            on_click=lambda: [
                st.session_state.pop("pdf_data", None),
                st.session_state.pop("pdf_filename", None)
            ]
        ) 
    # Sidebar info
    with st.sidebar:
        st.markdown("### 🔍 So you are a curious one :)")
        st.markdown("- **Embeddings**: text-embedding-004")
        st.markdown("- **Generation**: gemini-2.5-flash-lite-preview-06-17")
        st.markdown("- **Vector dims**: 768")
        st.markdown("- **Search**: Cosine similarity")
        st.markdown("- **Data Source**: JSON")
        if st.button("🔍 View Generated Chunks"):
            st.session_state.show_chunks = not st.session_state.get("show_chunks", False)
        
        if st.session_state.get("show_chunks", False):
            st.markdown("### 📋 Generated Chunks")
            if rag_system.configured and rag_system.cv_chunks:
                for i, chunk in enumerate(rag_system.cv_chunks):
                    with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                        st.text(chunk)
            else:
                st.warning("No chunks available")

        if rag_system.configured and rag_system.cv_chunks:
            st.markdown(f"- **Chunks loaded**: {len(rag_system.cv_chunks)}")
            st.markdown(f"- **Embeddings**: {'✅' if rag_system.cv_embeddings is not None else '❌'}")
            st.markdown(f"- **Job Analyzer**: {'✅' if rag_system.tool_definitions.job_compatibility_analyzer else '❌'}")


if __name__ == "__main__":
    main()