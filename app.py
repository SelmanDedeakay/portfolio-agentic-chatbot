import streamlit as st
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import tools and components
from tools.email_tool import EmailTool
from tools.social_media_tool import SocialMediaAggregator
from tools.tool_definitions import ToolDefinitions
from ui.email_components import get_ui_text, render_email_verification_card, render_email_editor_card

load_dotenv()


class AppConstants:
    """Application-wide constants"""
    MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
    EMBEDDING_MODEL = "models/text-embedding-004"
    DEFAULT_TEMPERATURE = 0.1
    MAX_OUTPUT_TOKENS = 1200
    
    # Search parameters
    DEFAULT_TOP_K = 4
    PROJECT_TOP_K = 6
    EDUCATION_TOP_K = 8
    
    # UI defaults
    DEFAULT_LANGUAGE = "en"
    
    # Chunk boost scores
    KEYWORD_BOOST_SCORE = 0.2


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    TURKISH = "tr"


@dataclass
class QueryType:
    """Query type detection result"""
    is_social_query: bool = False
    is_job_query: bool = False
    is_project_query: bool = False
    is_experience_query: bool = False
    is_education_query: bool = False
    is_contact_query: bool = False


class LanguageDetector:
    """Enhanced language detection with caching and optimization"""
    
    # Language-specific indicators
    TURKISH_CHARS = set('çğıöşü')
    
    TURKISH_KEYWORDS = {
        'hakkında', 'nedir', 'kimdir', 'nasıl', 'merhaba', 'teşekkür', 'iletişim', 'mesaj', 'gönder',
        'anlat', 'söyle', 'nerede', 'ne zaman', 'hangi', 'proje', 'projeler', 'deneyim', 'eğitim',
        'çalışma', 'iş', 'üniversite', 'okul', 'mezun', 'değil', 'yok', 'var', 'olan', 'yapan',
        'merhabalar', 'selam', 'günaydın', 'teşekkürler', 'sağol', 'kariyer', 'bilgi', 'selamlar',
        'anladım', 'bilmiyorum', 'istiyorum', 'isterim', 've', 'bir', 'bu', 'şu', 'o', 'ben', 'sen',
        'ile', 'için', 'ama', 'fakat', 'lakin', 'çünkü', 'ki', 'da', 'de', 'ta', 'te'
    }
    
    ENGLISH_KEYWORDS = {
        'hello', 'hi', 'what', 'who', 'when', 'where', 'why', 'how', 'about', 'thank', 'thanks',
        'tell', 'show', 'project', 'experience', 'work', 'education', 'university', 'job', 'i', 'you',
        'know', 'dont', "don't", 'want', 'need', 'can', 'could', 'would', 'should', 'the', 'and',
        'with', 'for', 'but', 'because', 'that', 'this', 'they', 'we', 'he', 'she', 'it', 'my', 'your'
    }
    
    # Phrase patterns
    TURKISH_PHRASES = {'bilmiyorum', 'istiyorum', 'yapabilir', 'söyleyebilir', 'eder misin', 'var mı'}
    ENGLISH_PHRASES = {"i dont", "i don't", "i want", "i need", "i can", "i would", "could you", "can you"}
    
    # Quick lookup greetings
    TURKISH_GREETINGS = {'selam', 'merhaba', 'merhabalar', 'selamlar', 'günaydın', 'iyi günler'}
    ENGLISH_GREETINGS = {'hello', 'hi', 'hey', 'greetings', 'good morning', 'good day'}
    
    @classmethod
    def detect_from_text(cls, text: str) -> Language:
        """Detect language from a single text"""
        if not text:
            return Language.ENGLISH
        
        text_lower = text.lower().strip()
        
        # Quick checks for very short messages
        if len(text_lower) <= 3:
            if text_lower in {'hi', 'hey'}:
                return Language.ENGLISH
            elif text_lower in {'selam', 'mrb'}:
                return Language.TURKISH
        
        # Check greetings first (fastest)
        if text_lower in cls.TURKISH_GREETINGS:
            return Language.TURKISH
        elif text_lower in cls.ENGLISH_GREETINGS:
            return Language.ENGLISH
        
        # Turkish character detection (very strong indicator)
        if any(char in text_lower for char in cls.TURKISH_CHARS):
            return Language.TURKISH
        
        # Phrase detection
        for phrase in cls.ENGLISH_PHRASES:
            if phrase in text_lower:
                return Language.ENGLISH
        
        for phrase in cls.TURKISH_PHRASES:
            if phrase in text_lower:
                return Language.TURKISH
        
        # Keyword scoring
        text_words = set(text_lower.split())
        turkish_score = len(text_words & cls.TURKISH_KEYWORDS) * 2
        english_score = len(text_words & cls.ENGLISH_KEYWORDS)
        
        if turkish_score == 0 and english_score == 0:
            return Language.ENGLISH
        
        return Language.TURKISH if turkish_score > english_score else Language.ENGLISH
    
    @classmethod
    def detect_from_messages(cls, messages: List[Dict[str, str]]) -> Language:
        """Detect language from conversation history"""
        if not messages:
            return Language.ENGLISH
        
        # Get last user message
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                return cls.detect_from_text(msg.get('content', ''))
        
        return Language.ENGLISH


class QueryClassifier:
    """Classify user queries into different types"""
    
    # Keywords for different query types
    SOCIAL_KEYWORDS = {
        'post', 'article', 'medium', 'linkedin', 'social media', 
        'paylaşım', 'makale', 'yazı', 'sosyal medya'
    }
    
    JOB_KEYWORDS = {
        'job', 'position', 'role', 'hiring', 'recruit', 'vacancy', 
        'opening', 'career', 'employment', 'iş', 'pozisyon', 
        'işe alım', 'kariyer', 'istihdam', 'açık pozisyon'
    }
    
    PROJECT_KEYWORDS = {
        'proje', 'project', 'yaptığı', 'geliştirdiği', 'built', 
        'developed', 'created', 'portfolio'
    }
    
    EXPERIENCE_KEYWORDS = {
        'deneyim', 'experience', 'çalış', 'work', 'iş', 'worked', 
        'job history', 'employment history'
    }
    
    EDUCATION_KEYWORDS = {
        'eğitim', 'education', 'university', 'üniversite', 'okul', 
        'school', 'mezun', 'graduate', 'diploma', 'degree', 'lisans', 
        'bachelor', 'erasmus', 'exchange', 'öğrenci', 'student'
    }
    
    CONTACT_KEYWORDS = {
        'contact', 'email', 'reach', 'get in touch', 'message', 
        'iletişim', 'ulaş', 'mesaj', 'e-posta', 'mail'
    }
    
    @classmethod
    def classify(cls, query: str) -> QueryType:
        """Classify query into different types"""
        query_lower = query.lower()
        
        return QueryType(
            is_social_query=any(kw in query_lower for kw in cls.SOCIAL_KEYWORDS),
            is_job_query=any(kw in query_lower for kw in cls.JOB_KEYWORDS),
            is_project_query=any(kw in query_lower for kw in cls.PROJECT_KEYWORDS),
            is_experience_query=any(kw in query_lower for kw in cls.EXPERIENCE_KEYWORDS),
            is_education_query=any(kw in query_lower for kw in cls.EDUCATION_KEYWORDS),
            is_contact_query=any(kw in query_lower for kw in cls.CONTACT_KEYWORDS)
        )


class ChunkBuilder:
    """Build searchable chunks from CV data"""
    
    @staticmethod
    def build_basic_info(data: Dict[str, Any]) -> str:
        """Build basic information chunk"""
        return f"""Name: {data.get('name', 'N/A')}
Title: {data.get('title', 'N/A')}
Location: {data.get('location', 'N/A')}
Email: {data.get('email', 'N/A')}
Phone: {data.get('phone', 'N/A')}
Profile: {data.get('profile', 'N/A')}"""
    
    @staticmethod
    def build_links_chunk(links: Dict[str, str]) -> str:
        """Build social links chunk"""
        links_text = "Links and Social Media:\n"
        for platform, url in links.items():
            links_text += f"- {platform.capitalize()}: {url}\n"
        return links_text
    
    @staticmethod
    def build_education_chunk(edu: Dict[str, Any]) -> str:
        """Build individual education chunk"""
        edu_text = f"Education / Eğitim: {edu.get('institution', 'N/A')}\n"
        
        degree_info = edu.get('degree') or edu.get('program', 'N/A')
        edu_text += f"Degree/Program/Derece: {degree_info}\n"
        
        year_info = edu.get('years') or edu.get('year', 'N/A')
        edu_text += f"Years/Duration/Süre: {year_info}\n"
        
        location_info = edu.get('location', 'N/A')
        edu_text += f"Location/Konum: {location_info}\n"
        
        if memberships := edu.get('memberships'):
            edu_text += f"Memberships/Üyelikler: {', '.join(memberships)}\n"
        
        keywords = [
            "education", "eğitim", "university", "üniversite", "degree", "derece", 
            "diploma", "bachelor", "lisans", "graduate", "mezun", "student", "öğrenci",
            edu.get('institution', '').lower().replace(' ', '_')
        ]
        edu_text += f"Keywords: {', '.join(keywords)}"
        
        return edu_text
    
    @staticmethod
    def build_experience_chunk(exp: Dict[str, Any]) -> str:
        """Build work experience chunk"""
        return f"""Work Experience / İş Deneyimi:
Position/Pozisyon: {exp.get('title', 'N/A')}
Company/Şirket: {exp.get('company', 'N/A')}
Duration/Süre: {exp.get('duration', 'N/A')}
Job Description/İş Tanımı: {exp.get('description', 'N/A')}
Keywords: work experience, iş deneyimi, {exp.get('company', '').lower()}, {exp.get('title', '').lower()}"""
    
    @staticmethod
    def build_project_chunk(project: Dict[str, Any]) -> str:
        """Build project chunk"""
        proj_text = f"""Project / Proje:
Project Name/Proje Adı: {project.get('name', 'N/A')}
Technology Used/Kullanılan Teknoloji: {project.get('technology', 'N/A')}
Project Description/Proje Açıklaması: {project.get('description', 'N/A')}
Keywords: project, proje, {project.get('technology', '').lower()}, {project.get('name', '').lower()}"""
        
        if link := project.get('link'):
            proj_text += f"\nProject Link/Proje Linki: {link}"
        
        return proj_text
    
    @staticmethod
    def build_skills_chunk(skills: Dict[str, List[str]]) -> str:
        """Build skills chunk"""
        skills_text = "Technical Skills:\n"
        for category, skill_list in skills.items():
            if isinstance(skill_list, list):
                skills_text += f"{category}: {', '.join(skill_list)}\n"
        return skills_text


class GeminiEmbeddingRAG:
    """Enhanced RAG with tool calling for email using JSON data"""
    
    def __init__(self, json_path: str = "selman-cv.json"):
        self.json_path = json_path
        self.cv_data: Dict[str, Any] = {}
        self.cv_chunks: List[str] = []
        self.cv_embeddings: Optional[np.ndarray] = None
        self.configured = False
        
        # Initialize tools
        self.email_tool = EmailTool()
        self.tool_definitions = ToolDefinitions()
        self.social_media_aggregator = SocialMediaAggregator()
        
        # Initialize helpers
        self.chunk_builder = ChunkBuilder()
        self.query_classifier = QueryClassifier()
        
        # Initialize client
        self._initialize_client()
        
        if self.configured:
            self.load_cv()
    
    def _initialize_client(self) -> None:
        """Initialize Gemini client with proper error handling"""
        try:
            api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                self.configured = True
            else:
                self.configured = False
                st.error("❌ We are having trouble connecting to Chatbot.")
        except Exception as e:
            st.error(f"❌ Chatbot setup failed: {e}")
            self.configured = False
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using Gemini embedding model with batch processing"""
        if not self.configured or not texts:
            return np.array([])
        
        try:
            embeddings = []
            
            # Process in batches for better performance
            batch_size = 5
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                for text in batch:
                    response = self.client.models.embed_content(
                        model=AppConstants.EMBEDDING_MODEL,
                        contents=[text]
                    )
                    embeddings.append(response.embeddings[0].values)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return np.array([])
    
    def json_to_chunks(self, data: Dict[str, Any]) -> List[str]:
        """Convert JSON data to searchable text chunks"""
        chunks = []
        
        # Basic information
        chunks.append(self.chunk_builder.build_basic_info(data))
        
        # Links
        if links := data.get('links', {}):
            chunks.append(self.chunk_builder.build_links_chunk(links))
        
        # Education
        if education := data.get('education', []):
            for edu in education:
                chunks.append(self.chunk_builder.build_education_chunk(edu))
            
            # Summary chunk
            summary = "Complete Education Background / Tüm Eğitim Geçmişi:\n"
            for i, edu in enumerate(education, 1):
                degree_info = edu.get('degree') or edu.get('program', 'Program')
                year_info = edu.get('years') or edu.get('year', '')
                summary += f"{i}. {degree_info} - {edu.get('institution', 'N/A')} ({year_info})\n"
            summary += "\nKeywords: complete education, tüm eğitim, educational background, eğitim geçmişi"
            chunks.append(summary)
        
        # Experience
        if experience := data.get('experience', []):
            for exp in experience:
                chunks.append(self.chunk_builder.build_experience_chunk(exp))
            
            # Summary chunk
            summary = "All Work Experience / Tüm İş Deneyimleri:\n"
            for exp in experience:
                summary += f"- {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('duration', 'N/A')})\n"
            chunks.append(summary)
        
        # Skills
        if skills := data.get('skills', {}):
            chunks.append(self.chunk_builder.build_skills_chunk(skills))
        
        # Projects
        if projects := data.get('projects', []):
            for project in projects:
                chunks.append(self.chunk_builder.build_project_chunk(project))
            
            # Summary chunk
            summary = "All Projects / Tüm Projeler:\n"
            for project in projects:
                summary += f"- {project.get('name', 'N/A')} ({project.get('technology', 'N/A')})\n"
            chunks.append(summary)
        
        # Awards
        for award in data.get('awards', []):
            award_text = f"Award: {award.get('name', 'N/A')}\n"
            award_text += f"Organization: {award.get('organization', 'N/A')}\n"
            award_text += f"Description: {award.get('description', 'N/A')}"
            chunks.append(award_text)
        
        # Languages
        if languages := data.get('languages', {}):
            lang_text = "Languages:\n"
            for lang, level in languages.items():
                lang_text += f"- {lang}: {level}\n"
            chunks.append(lang_text)
        
        # Organizations
        for org in data.get('organizations', []):
            org_text = f"Organization: {org.get('name', 'N/A')}\n"
            org_text += f"Role: {org.get('role', 'N/A')}\n"
            org_text += f"Duration: {org.get('duration', 'N/A')}"
            chunks.append(org_text)
        
        # References
        if references := data.get('references', []):
            ref_text = "References:\n"
            for ref in references:
                ref_text += f"- {ref.get('name', 'N/A')} ({ref.get('title', 'N/A')} at {ref.get('organization', 'N/A')})"
            chunks.append(ref_text)
        
        return chunks
    
    def load_cv(self) -> None:
        """Load CV from JSON and create embeddings"""
        try:
            if not os.path.exists(self.json_path):
                st.error(f"❌ CV file '{self.json_path}' not found.")
                return
            
            with open(self.json_path, 'r', encoding='utf-8') as file:
                self.cv_data = json.load(file)
            
            if not self.cv_data:
                st.error("❌ JSON file is empty or unreadable")
                return
            
            # Convert to chunks
            self.cv_chunks = self.json_to_chunks(self.cv_data)
            
            # Generate embeddings
            with st.spinner(f"Generating embeddings for {len(self.cv_chunks)} chunks..."):
                self.cv_embeddings = self.get_embeddings(self.cv_chunks)
            
            if self.cv_embeddings.size > 0:
                # Initialize job compatibility analyzer
                self.tool_definitions.initialize_job_analyzer(
                    self.client, 
                    self.cv_data, 
                    self
                )
            else:
                st.error("❌ Failed to generate embeddings")
                
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
        except Exception as e:
            st.error(f"Error loading CV: {e}")
    
    def _calculate_keyword_boost(self, query: str, chunk: str) -> float:
        """Calculate keyword boost score for a chunk"""
        query_lower = query.lower()
        chunk_lower = chunk.lower()
        
        # Keyword boost mappings
        keyword_mappings = {
            'proje': ['project', 'proje'],
            'projects': ['project', 'proje'],
            'deneyim': ['experience', 'deneyim', 'work', 'iş'],
            'experience': ['experience', 'deneyim', 'work', 'iş'],
            'work': ['experience', 'deneyim', 'work', 'iş'],
            'iş': ['experience', 'deneyim', 'work', 'iş'],
            'çalış': ['experience', 'deneyim', 'work', 'iş'],
            'eğitim': ['education', 'eğitim', 'university', 'üniversite', 'degree', 'derece'],
            'education': ['education', 'eğitim', 'university', 'üniversite', 'degree', 'derece'],
            'university': ['education', 'eğitim', 'university', 'üniversite'],
            'üniversite': ['education', 'eğitim', 'university', 'üniversite'],
        }
        
        boost = 0.0
        for key, keywords in keyword_mappings.items():
            if key in query_lower:
                for keyword in keywords:
                    if keyword in chunk_lower:
                        boost += AppConstants.KEYWORD_BOOST_SCORE
                        break
        
        return boost
    
    def search_similar_chunks(self, query: str, top_k: int = AppConstants.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Enhanced search with keyword matching and caching"""
        if not self.configured or self.cv_embeddings is None or self.cv_embeddings.size == 0:
            return [{"text": "Embeddings not available", "similarity": 0.0, "index": -1}]
        
        # Get query embedding
        query_embedding = self.get_embeddings([query])
        if query_embedding.size == 0:
            return [{"text": "Could not process query", "similarity": 0.0, "index": -1}]
        
        query_vec = query_embedding[0]
        query_norm = np.linalg.norm(query_vec)
        
        # Calculate similarities with boost
        similarities = []
        for i, chunk_vec in enumerate(self.cv_embeddings):
            # Cosine similarity
            chunk_norm = np.linalg.norm(chunk_vec)
            if query_norm > 0 and chunk_norm > 0:
                similarity = np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm)
            else:
                similarity = 0.0
            
            # Apply keyword boost
            boost = self._calculate_keyword_boost(query, self.cv_chunks[i])
            
            similarities.append({
                "text": self.cv_chunks[i],
                "similarity": float(similarity + boost),
                "index": i
            })
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def _build_prompt(self, query: str, context: str, language: Language, recent_context: str) -> str:
        """Build appropriate prompt based on language"""
        if language == Language.TURKISH:
            return f"""Siz Selman Dedeakayoğulları'nın AI portföy asistanısınız. Portföy web sitesine yerleştiriliyorsunuz. Ziyaretçiler size sorular soracak.

    Kurallar:
    - SADECE TÜRKÇE yanıtlayın
    - CV soruları için yalnızca sağlanan bağlamdan bilgi kullanın
    - Profesyonel ve yardımsever olun
    - Netlik ve okunabilirlik için markdown biçimlendirmesini kullanın
    - Kullanıcı referans isterse, bunları görüntüleyin ve talep üzerine iletişim bilgilerinin mevcut olduğuna dair bir not ekleyin
    - Projeler veya iş deneyimleri hakkında sorulduğunda, bağlamdan TÜM ilgili öğeleri listeleyin
    - Proje soruları için, proje adlarını, kullanılan teknolojileri ve açıklamaları ekleyin. Özel olarak istenmediği sürece bağlantı vermeyin. "Agentic Portfolio Bot" hakkında konuşurken, siz olduğunuz için bununla ilgili bir şaka yapın. 
    - Deneyim soruları için şirket adlarını, pozisyonları, süreleri ve açıklamaları ekleyin

    EMAIL KURALLARI - ÇOK ÖNEMLİ:
    - Birisi Selman ile iletişime geçmek istediğinde, prepare_email aracını KULLANMADAN ÖNCE şu bilgilerin TAMAMINI toplayın:
    1. Gönderenin tam adı (ad ve soyad gerekli)
    2. Gönderenin e-posta adresi
    3. Mesaj içeriği
    - Bu bilgilerden HERHANGİ BİRİ eksikse, önce eksik bilgileri isteyin
    - Örnek: "E-posta gönderebilmem için adınızı ve e-posta adresinizi öğrenebilir miyim?"
    - TÜM bilgiler toplandıktan SONRA prepare_email aracını kullanın

    DİĞER ARAÇLAR:
    - Birisi Selman'ın son gönderileri, makaleleri, Medium içeriği, LinkedIn etkinliği veya sosyal medyası hakkında soru sorduğunda get_recent_posts aracını kullanın
    - Birisi bir iş tanımı sağladığında veya belirli bir rol için uygunluk hakkında soru sorduğunda analyze_job_compatibility aracını kullanın
    - Kullanıcı PDF istediğinde, indirdiğinde veya iş uyumluluk raporunu kaydetmek istediğinde generate_compatibility_pdf aracını kullanın

    Son Konuşma Bağlamı:
    {recent_context}

    CV Bağlamı:
    {context}

    Kullanıcı Sorusu: {query}
    Yanıt:"""
        else:
            return f"""You are Selman Dedeakayoğulları's AI portfolio assistant. You are embedded in his portfolio website. Visitors will ask questions to you.

    Rules:
    - Respond ONLY in ENGLISH
    - Only use information from the provided context for CV questions
    - Be professional and helpful
    - Use markdown formatting for clarity and readability
    - If the user asks for references, display them and add a note that contact information is available upon request
    - When asked about projects or work experience, list ALL relevant items from the context
    - For project questions, include project names, technologies used, and descriptions. Do not give links unless asked specifically. When talking about "Agentic Portfolio Bot" make a joke about it, since it is you.
    - For experience questions, include company names, positions, durations, and descriptions

    EMAIL RULES - VERY IMPORTANT:
    - When someone wants to contact Selman, BEFORE using prepare_email tool, collect ALL of the following:
    1. Sender's full name (first and last name required)
    2. Sender's email address
    3. Message content
    - If ANY of this information is missing, ask for the missing details first
    - Example: "I'd be happy to help you contact Selman. Could you please provide your full name and email address?"
    - ONLY use prepare_email tool after ALL information is collected

    OTHER TOOLS:
    - Use get_recent_posts tool when someone asks about Selman's recent posts, articles, Medium content, LinkedIn activity, or social media
    - Use analyze_job_compatibility tool when someone provides a job description or asks about fit for a specific role
    - Use generate_compatibility_pdf tool when user asks for PDF, download, or wants to save the job compatibility report

    Recent Conversation Context:
    {recent_context}

    CV Context:
    {context}

    User Question: {query}


    Response:"""
    
    def _get_recent_context(self, conversation_history: List[Dict[str, str]]) -> str:
        """Extract recent conversation context"""
        if not conversation_history or len(conversation_history) <= 1:
            return ""
        
        # Get last 4 messages
        recent_messages = conversation_history[-4:]
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    
    def _determine_top_k(self, query_type: QueryType) -> int:
        """Determine optimal top_k based on query type"""
        if query_type.is_education_query:
            return AppConstants.EDUCATION_TOP_K
        elif query_type.is_project_query or query_type.is_experience_query:
            return AppConstants.PROJECT_TOP_K
        else:
            return AppConstants.DEFAULT_TOP_K
    
    def _handle_function_call(self, part: Any, language: Language) -> Optional[str]:
        """Handle function call from LLM response"""
        if not hasattr(part, 'function_call') or not part.function_call:
            return None
        
        tool_name = part.function_call.name
        tool_args = dict(part.function_call.args.items())
        
        result = self.tool_definitions.execute_tool(tool_name, tool_args)
        
        if not result["success"]:
            return f"❌ {result['message']}"
        
        # Handle different tool results
        if tool_name == "prepare_email":
            return "EMAIL_PREPARED_FOR_REVIEW"
        
        elif tool_name == "get_recent_posts":
            return result["data"]["formatted_response"]
        
        elif tool_name == "analyze_job_compatibility":
            st.session_state.last_compatibility_report = result["data"]["report"]
            st.session_state.last_job_title = result["data"]["job_title"]
            
            pdf_msg = (
                "\n\n📄 *Bu raporun PDF versiyonunu indirmek isterseniz söyleyebilirsiniz!*"
                if language == Language.TURKISH
                else "\n\n📄 *You can ask for a PDF version of this report if you'd like to download it!*"
            )
            
            return result["data"]["report"] + pdf_msg
        
        elif tool_name == "generate_compatibility_pdf":
            return "PDF_GENERATED"
        
        return None
    
    def generate_response(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response with tool calling capability and Turkish support"""
        if not self.configured:
            return "Gemini API not configured"
        
        # Detect language
        messages = (conversation_history or []) + [{"role": "user", "content": query}]
        language = LanguageDetector.detect_from_messages(messages)
        
        # Get conversation context
        recent_context = self._get_recent_context(conversation_history or [])
        
        # Classify query
        query_type = self.query_classifier.classify(query)
        
        # Get relevant chunks
        top_k = self._determine_top_k(query_type)
        relevant_chunks = self.search_similar_chunks(query, top_k=top_k)
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Build prompt
        prompt = self._build_prompt(query, context, language, recent_context)
        
        try:
            # Generate response
            response = self.client.models.generate_content(
                model=AppConstants.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=AppConstants.DEFAULT_TEMPERATURE,
                    max_output_tokens=AppConstants.MAX_OUTPUT_TOKENS,
                    tools=self.tool_definitions.get_all_tools()
                )
            )
            
            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if function_result := self._handle_function_call(part, language):
                        return function_result
            
            # Return text response
            return response.text if response.text else "No response generated"
            
        except Exception as e:
            error_msg = (
                f"Yanıt oluşturulurken hata: {e}"
                if language == Language.TURKISH
                else f"Error generating response: {e}. The API response might have been empty or invalid."
            )
            return error_msg


class ChatInterface:
    """Manage chat interface and interactions"""
    
    def __init__(self, rag_system: GeminiEmbeddingRAG):
        self.rag_system = rag_system
    
    def handle_email_actions(self) -> None:
        """Handle email-related actions"""
        if "email_action" not in st.session_state or not st.session_state.email_action:
            return
        
        action = st.session_state.email_action
        language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
        ui_text = get_ui_text(language.value)
        
        if action == "send":
            self._send_email(ui_text)
        elif action == "cancel":
            self._cancel_email(ui_text)
        elif action == "edit":
            self._edit_email()
    
    def _send_email(self, ui_text: Dict[str, str]) -> None:
        """Send pending email"""
        email_data = st.session_state.pending_email
        
        with st.spinner(ui_text.get("sending_email", "Sending email...")):
            result = self.rag_system.email_tool.send_email(
                email_data['sender_name'],
                email_data['sender_email'],
                email_data['subject'],
                email_data['message']
            )
        
        # Clear pending email
        del st.session_state.pending_email
        del st.session_state.email_action
        
        # Add result message
        message_content = (
            ui_text["email_sent"] if result["success"]
            else ui_text["email_failed"] + result['message']
        )
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": message_content
        })
        st.rerun()
    
    def _cancel_email(self, ui_text: Dict[str, str]) -> None:
        """Cancel pending email"""
        del st.session_state.pending_email
        del st.session_state.email_action
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": ui_text["email_cancelled"]
        })
        st.rerun()
    
    def _edit_email(self) -> None:
        """Switch to email edit mode"""
        st.session_state.editing_email = True
        del st.session_state.email_action
        st.rerun()
    
    def display_messages(self) -> None:
        """Display chat messages with special handling for emails"""
        language = LanguageDetector.detect_from_messages(st.session_state.messages)
        ui_text = get_ui_text(language.value)
        
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                # Check if this is an email preparation message
                is_email_message = (
                    i == len(st.session_state.messages) - 1 and
                    message.get("content") in [
                        ui_text["email_prepared"], 
                        "I've prepared your email to Selman. Please review the details below before sending."
                    ] and
                    "pending_email" in st.session_state
                )
                
                if is_email_message:
                    st.write(message["content"])
                    
                    # Show appropriate email card
                    if st.session_state.get("editing_email", False):
                        render_email_editor_card(
                            st.session_state.pending_email, 
                            language.value
                        )
                    else:
                        render_email_verification_card(
                            st.session_state.pending_email, 
                            language.value
                        )
                else:
                    st.write(message["content"])
    
    def process_user_input(self, prompt: str) -> None:
        """Process user input and generate response"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Detect language for UI
        language = LanguageDetector.detect_from_messages(st.session_state.messages)
        ui_text = get_ui_text(language.value)
        
        # Generate response
        with st.chat_message("assistant"):
            spinner_msg = (
                "İsteğiniz işleniyor..." 
                if language == Language.TURKISH 
                else "Processing your request..."
            )
            
            with st.spinner(spinner_msg):
                response = self.rag_system.generate_response(
                    prompt, 
                    st.session_state.messages
                )
            
            # Handle special responses
            if response == "EMAIL_PREPARED_FOR_REVIEW":
                message = ui_text["email_prepared"]
                st.write(message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": message
                })
                
                if "pending_email" in st.session_state:
                    render_email_verification_card(
                        st.session_state.pending_email, 
                        language.value
                    )
            
            elif response == "PDF_GENERATED":
                message = (
                    "✅ PDF raporu başarıyla oluşturuldu! Aşağıdaki butona tıklayarak indirebilirsiniz."
                    if language == Language.TURKISH
                    else "✅ PDF report generated successfully! You can download it using the button below."
                )
                st.write(message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": message
                })
            
            else:
                st.write(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })


def render_sidebar(rag_system: GeminiEmbeddingRAG) -> None:
    """Render sidebar with system information"""
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

import base64

def render_pdf_download() -> None:
    """Mobile-friendly PDF download with fallback options"""
    if "pdf_data" not in st.session_state or "pdf_filename" not in st.session_state:
        return

    # Language detection for UI text
    language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
    
    # Prepare data
    pdf_bytes = st.session_state.pdf_data
    file_name = st.session_state.pdf_filename
    
    # UI text based on language
    if language == Language.TURKISH:
        download_text = "📄 PDF Raporu İndir"
        clear_text = "🗑️ Temizle"
        mobile_note = "💡 Mobil cihazlarda indirme çalışmıyorsa, dosyayı açıp 'Farklı Kaydet' seçeneğini kullanın"
    else:
        download_text = "📄 Download PDF Report" 
        clear_text = "🗑️ Clear"
        mobile_note = "💡 If download doesn't work on mobile, open the file and use 'Save As' option"

    # Custom CSS for responsive design
    st.markdown("""
        <style>
        .pdf-download-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .pdf-button-row {
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .mobile-note {
            font-size: 12px;
            color: #666;
            text-align: center;
            font-style: italic;
            margin-top: 10px;
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .pdf-button-row {
                flex-direction: column;
                gap: 10px;
            }
            
            .mobile-note {
                font-size: 11px;
                line-height: 1.4;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Container for better styling
    st.markdown('<div class="pdf-download-container">', unsafe_allow_html=True)
    
    # Use Streamlit's built-in download button (more reliable on mobile)
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label=download_text,
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
            help="Click to download the PDF report" if language == Language.ENGLISH else "PDF raporu indirmek için tıklayın"
        )
    
    with col2:
        if st.button(
            clear_text, 
            key="clear_download", 
            use_container_width=True,
            type="secondary",
            help="Clear the download" if language == Language.ENGLISH else "İndirmeyi temizle"
        ):
            st.session_state.pop("pdf_data", None)
            st.session_state.pop("pdf_filename", None)
            st.rerun()
    
    # Mobile help note
    st.markdown(f'<div class="mobile-note">{mobile_note}</div>', unsafe_allow_html=True)
    
    # Alternative: Show PDF in browser for mobile users
    if st.button(
        "🔍 Tarayıcıda Görüntüle / View in Browser", 
        use_container_width=True,
        type="secondary",
        help="Open PDF in browser for mobile viewing" if language == Language.ENGLISH else "Mobil görüntüleme için PDF'i tarayıcıda aç"
    ):
        # Create a data URL for viewing
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="600px" type="application/pdf"></iframe>'
        
        # Show in expander for better UX
        with st.expander("📱 PDF Görüntüleyici / PDF Viewer", expanded=True):
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.info(
                "Mobil cihazlarda: Dosyayı görüntüledikten sonra paylaş butonunu kullanarak indirebilirsiniz." 
                if language == Language.TURKISH 
                else "On mobile: After viewing the file, you can use the share button to download it."
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def initialize_session_state() -> None:
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Hello! I'm here to answer questions about Selman. What would you like to know? "
                "I can also help you get in touch with him directly if needed! 📧\n\n"
                "I can also analyze job compatibility if you have a job description you'd like me "
                "to review against Selman's profile! 💼\n\n"
                "Merhaba! Selman hakkında sorularınızı yanıtlayabilirim. "
                "Onunla doğrudan iletişime geçmenize de yardımcı olabilirim! 📧\n\n"
                "Ayrıca bir iş ilanınız varsa, Selman'ın profiliyle uyumluluğunu analiz edebilirim! 💼"
            )
        }]


def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Selman DEDEAKAYOĞULLARI Portfolio RAG Chatbot",
        page_icon="🔍",
        layout="centered"
    )
    
    # Header
    st.title("Welcome!")
    st.caption("I'm Selman's AI portfolio assistant, what would you like to know about him?")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        with st.spinner("Initializing Chatbot"):
            st.session_state.rag_system = GeminiEmbeddingRAG()
    
    rag_system = st.session_state.rag_system
    
    # Check configuration
    if not rag_system.configured:
        st.error("Please configure GEMINI_API_KEY to continue")
        st.stop()
    
    # Check email configuration
    if not rag_system.email_tool.email_user or not rag_system.email_tool.email_password:
        st.warning(
            "⚠️ Email functionality is not configured. "
            "Please set EMAIL_USER and EMAIL_PASSWORD environment variables."
        )
    
    # Initialize chat interface
    chat_interface = ChatInterface(rag_system)
    
    # Handle email actions
    chat_interface.handle_email_actions()
    
    # Display messages
    chat_interface.display_messages()
    
    # Chat input
    if prompt := st.chat_input(
        "Ask about Selman's background, request to contact him, "
        "or paste a job description for compatibility analysis..."
    ):
        chat_interface.process_user_input(prompt)
    
    # PDF download button
    render_pdf_download()
    
    # Sidebar
    render_sidebar(rag_system)


if __name__ == "__main__":
    main()