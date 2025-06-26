import streamlit as st
import os
import numpy as np
import pickle
import hashlib
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
    
    # Cache settings
    CACHE_DIR = ".cache"
    EMBEDDINGS_CACHE_FILE = "cv_embeddings.pkl"
    CHUNKS_CACHE_FILE = "cv_chunks.pkl"
    CACHE_INFO_FILE = "cache_info.json"


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


class EmbeddingCache:
    """Handle embedding caching operations"""
    
    def __init__(self, cache_dir: str = AppConstants.CACHE_DIR):
        self.cache_dir = cache_dir
        self.embeddings_path = os.path.join(cache_dir, AppConstants.EMBEDDINGS_CACHE_FILE)
        self.chunks_path = os.path.join(cache_dir, AppConstants.CHUNKS_CACHE_FILE)
        self.cache_info_path = os.path.join(cache_dir, AppConstants.CACHE_INFO_FILE)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of a file"""
        if not os.path.exists(file_path):
            return ""
        
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        if not os.path.exists(self.cache_info_path):
            return {}
        
        try:
            with open(self.cache_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_cache_info(self, cv_file_path: str, cv_hash: str, chunks_count: int) -> None:
        """Save cache information"""
        cache_info = {
            "cv_file_path": cv_file_path,
            "cv_file_hash": cv_hash,
            "chunks_count": chunks_count,
            "cached_at": str(np.datetime64('now')),
            "embedding_model": AppConstants.EMBEDDING_MODEL
        }
        
        try:
            with open(self.cache_info_path, 'w', encoding='utf-8') as f:
                json.dump(cache_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.warning(f"Could not save cache info: {e}")
    
    def is_cache_valid(self, cv_file_path: str) -> bool:
        """Check if cached embeddings are still valid"""
        # Check if all cache files exist
        if not all(os.path.exists(path) for path in [
            self.embeddings_path, 
            self.chunks_path, 
            self.cache_info_path
        ]):
            return False
        
        # Check cache info
        cache_info = self._get_cache_info()
        if not cache_info:
            return False
        
        # Check if CV file path matches
        if cache_info.get("cv_file_path") != cv_file_path:
            return False
        
        # Check if CV file hash matches (to detect changes)
        current_hash = self._get_file_hash(cv_file_path)
        if cache_info.get("cv_file_hash") != current_hash:
            return False
        
        # Check if embedding model matches
        if cache_info.get("embedding_model") != AppConstants.EMBEDDING_MODEL:
            return False
        
        return True
    
    def load_from_cache(self) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
        """Load chunks and embeddings from cache"""
        try:
            # Load chunks
            with open(self.chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            
            # Load embeddings
            with open(self.embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Validate data
            if not isinstance(chunks, list) or not isinstance(embeddings, np.ndarray):
                return None, None
            
            if len(chunks) != len(embeddings):
                return None, None
            
            return chunks, embeddings
            
        except Exception as e:
            st.warning(f"Could not load from cache: {e}")
            return None, None
    
    def save_to_cache(self, cv_file_path: str, chunks: List[str], embeddings: np.ndarray) -> bool:
        """Save chunks and embeddings to cache"""
        try:
            # Save chunks
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save embeddings
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save cache info
            cv_hash = self._get_file_hash(cv_file_path)
            self._save_cache_info(cv_file_path, cv_hash, len(chunks))
            
            return True
            
        except Exception as e:
            st.error(f"Could not save to cache: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear all cached files"""
        for file_path in [self.embeddings_path, self.chunks_path, self.cache_info_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                st.warning(f"Could not remove cache file {file_path}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "cache_exists": self.is_cache_valid("") if os.path.exists(self.cache_info_path) else False,
            "cache_info": self._get_cache_info(),
            "cache_size": 0
        }
        
        # Calculate cache size
        for file_path in [self.embeddings_path, self.chunks_path, self.cache_info_path]:
            if os.path.exists(file_path):
                stats["cache_size"] += os.path.getsize(file_path)
        
        return stats


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
    """Enhanced RAG with tool calling for email using JSON data and embedding caching"""
    
    def __init__(self, json_path: str = "selman-cv.json"):
        self.json_path = json_path
        self.cv_data: Dict[str, Any] = {}
        self.cv_chunks: List[str] = []
        self.cv_embeddings: Optional[np.ndarray] = None
        self.configured = False
        
        # Initialize cache
        self.cache = EmbeddingCache()
        
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
        """Load CV from JSON and create embeddings with caching"""
        try:
            if not os.path.exists(self.json_path):
                st.error(f"❌ CV file '{self.json_path}' not found.")
                return
            
            # Load CV data
            with open(self.json_path, 'r', encoding='utf-8') as file:
                self.cv_data = json.load(file)
            
            if not self.cv_data:
                st.error("❌ JSON file is empty or unreadable")
                return
            
            # Check if cache is valid
            if self.cache.is_cache_valid(self.json_path):

                cached_chunks, cached_embeddings = self.cache.load_from_cache()
                
                if cached_chunks is not None and cached_embeddings is not None:
                    self.cv_chunks = cached_chunks
                    self.cv_embeddings = cached_embeddings

                else:
                    st.warning("⚠️ Cache corrupted, regenerating embeddings...")
                    self._generate_fresh_embeddings()
            else:
                st.info("🔄 Cache not found or invalid, generating embeddings...")
                self._generate_fresh_embeddings()
            
            # Initialize job compatibility analyzer
            if self.cv_embeddings is not None and self.cv_embeddings.size > 0:
                self.tool_definitions.initialize_job_analyzer(
                    self.client, 
                    self.cv_data, 
                    self
                )
            else:
                st.error("❌ Failed to load or generate embeddings")
                
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
        except Exception as e:
            st.error(f"Error loading CV: {e}")
    
    def _generate_fresh_embeddings(self) -> None:
        """Generate fresh embeddings and cache them"""
        try:
            # Convert to chunks
            self.cv_chunks = self.json_to_chunks(self.cv_data)
            
            # Generate embeddings with progress tracking
            with st.spinner(f"Generating embeddings for {len(self.cv_chunks)} chunks..."):
                progress_bar = st.progress(0)
                
                # Generate embeddings in batches with progress updates
                embeddings = []
                batch_size = 5
                total_batches = (len(self.cv_chunks) + batch_size - 1) // batch_size
                
                for i, batch_start in enumerate(range(0, len(self.cv_chunks), batch_size)):
                    batch_end = min(batch_start + batch_size, len(self.cv_chunks))
                    batch_texts = self.cv_chunks[batch_start:batch_end]
                    
                    # Get embeddings for this batch
                    batch_embeddings = self.get_embeddings(batch_texts)
                    if batch_embeddings.size > 0:
                        if len(embeddings) == 0:
                            embeddings = batch_embeddings
                        else:
                            embeddings = np.vstack([embeddings, batch_embeddings])
                    
                    # Update progress
                    progress = (i + 1) / total_batches
                    progress_bar.progress(progress)
                
                progress_bar.empty()
                
                if embeddings is not None and len(embeddings) > 0:
                    self.cv_embeddings = embeddings.astype(np.float32)
                    
                    # Save to cache
                    with st.spinner("Saving embeddings to cache..."):
                        if self.cache.save_to_cache(self.json_path, self.cv_chunks, self.cv_embeddings):
                            st.success(f"✅ Generated and cached {len(self.cv_chunks)} chunks!")
                        else:
                            st.warning("⚠️ Embeddings generated but caching failed")
                else:
                    st.error("❌ Failed to generate embeddings")
                    self.cv_embeddings = np.array([])
                    
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            self.cv_embeddings = np.array([])
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self.cache.clear_cache()
        st.success("🗑️ Cache cleared successfully!")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_cache_stats()
    
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
            # Enhanced error handling with retry suggestion
            if language == Language.TURKISH:
                error_msg = (
                    f"Yanıt oluşturulurken hata: {e}. "
                    "API yanıtı boş veya geçersiz olabilir. "
                    "Lütfen birkaç saniye sonra tekrar deneyin."
                )
            else:
                error_msg = (
                    f"Error generating response: {e}. "
                    "The API response might have been empty or invalid. "
                    "Please wait a moment and try again."
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
    """Render sidebar with system information and cache controls"""
    with st.sidebar:
        st.markdown("### 🔍 So you are a curious one :)")
        st.markdown("- **Embeddings**: text-embedding-004")
        st.markdown("- **Generation**: gemini-2.5-flash-lite-preview-06-17")
        st.markdown("- **Vector dims**: 768")
        st.markdown("- **Search**: Cosine similarity")
        st.markdown("- **Data Source**: JSON")
        
        # Cache information
        if rag_system.configured:
            cache_stats = rag_system.get_cache_stats()
            st.markdown("### 💾 Cache Status")
            
            if cache_stats["cache_info"]:
                cache_info = cache_stats["cache_info"]
                st.markdown(f"- **Status**: ✅ Active")
                st.markdown(f"- **Chunks**: {cache_info.get('chunks_count', 'N/A')}")
                st.markdown(f"- **Size**: {cache_stats['cache_size'] / 1024:.1f} KB")
                st.markdown(f"- **Cached**: {cache_info.get('cached_at', 'N/A')[:16]}")
                
                # Cache actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh", help="Regenerate embeddings"):
                        rag_system.clear_cache()
                        rag_system.load_cv()
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Clear", help="Clear cache"):
                        rag_system.clear_cache()
                        st.success("Cache cleared!")
                        st.rerun()
            else:
                st.markdown("- **Status**: ❌ No cache")
        
        # Chunk viewer
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
        
        # System status
        if rag_system.configured and rag_system.cv_chunks:
            st.markdown(f"- **Chunks loaded**: {len(rag_system.cv_chunks)}")
            st.markdown(f"- **Embeddings**: {'✅' if rag_system.cv_embeddings is not None else '❌'}")
            st.markdown(f"- **Job Analyzer**: {'✅' if rag_system.tool_definitions.job_compatibility_analyzer else '❌'}")


import streamlit as st
import base64, uuid
import streamlit.components.v1 as components

def render_pdf_download() -> None:
    # PDF henüz yoksa çık
    if not {"pdf_data", "pdf_filename"} <= st.session_state.keys():
        return

    # ---------------- 1) Veriler --------------------------
    pdf_bytes = st.session_state.pdf_data
    file_name = st.session_state.pdf_filename
    b64_pdf   = base64.b64encode(pdf_bytes).decode()

    # ---------------- 2) Dil başlıkları -------------------
    lang = LanguageDetector.detect_from_messages(
        st.session_state.get("messages", [])
    )
    tr = lang == Language.TURKISH
    title         = "📄 PDF Raporu Hazır!"          if tr else "📄 PDF Report Ready!"
    view_text     = "👁️ PDF'yi Görüntüle"          if tr else "👁️ View PDF"
    download_text = "💾 PDF İndir"                 if tr else "💾 Download PDF"
    mobile_tip    = "📱 Mobilde PDF görüntüleme önerilir!" if tr \
                    else "📱 PDF viewing recommended on mobile!"
    email_text    = "📧 Email Gönder"              if tr else "📧 Email PDF"
    clear_text    = "🗑️ Temizle"                   if tr else "🗑️ Clear"

    # ---------------- 3) Başlık + indirme -----------------
    st.download_button(download_text, pdf_bytes, file_name,
                       mime="application/pdf", use_container_width=True)

    # ------------------------------------------------------
    # 5)  E-posta ve Temizle butonları - Responsive
    # ------------------------------------------------------
    col1, col2 = st.columns([1, 1], gap="small")
    with col1:
        if st.button(email_text, use_container_width=True):
            st.session_state.show_email_form = True
    with col2:
        if st.button(clear_text, use_container_width=True):
            for k in ("pdf_data", "pdf_filename", "show_email_form"):
                st.session_state.pop(k, None)
            st.rerun()

    # ------------------------------------------------------
    # 6)  E-posta formu
    # ------------------------------------------------------
    if st.session_state.get("show_email_form"):
        render_email_form_for_pdf(pdf_bytes, file_name, lang)


def render_email_form_for_pdf(pdf_bytes: bytes, filename: str, language: Language):
    """Clean email form without JavaScript"""
    
    if language == Language.TURKISH:
        form_title = "📧 PDF'i Email ile Alın"
        email_label = "Email Adresiniz:"
        email_placeholder = "ornek@email.com"
        send_text = "📧 PDF'i Gönder"
        cancel_text = "❌ İptal"
        success_msg = "✅ PDF başarıyla gönderildi! Email'inizi kontrol edin."
        error_msg = "❌ Email gönderilirken hata oluştu."
        invalid_email = "❌ Geçerli bir email adresi girin."
    else:
        form_title = "📧 Get PDF via Email"
        email_label = "Your Email:"
        email_placeholder = "example@email.com"
        send_text = "📧 Send PDF"
        cancel_text = "❌ Cancel"
        success_msg = "✅ PDF sent successfully! Check your email."
        error_msg = "❌ Failed to send email."
        invalid_email = "❌ Please enter a valid email address."
    
    # Email form
    st.markdown("---")
    st.markdown(f"### {form_title}")
    
    with st.form("pdf_email_form", clear_on_submit=True):
        user_email = st.text_input(
            email_label,
            placeholder=email_placeholder,
            help="We'll send the PDF report to this email address"
        )
        
        # Form submission buttons
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button(
                send_text, 
                use_container_width=True, 
                type="primary"
            )
        
        with col2:
            cancelled = st.form_submit_button(
                cancel_text, 
                use_container_width=True
            )
        
        # Handle form submission
        if submitted:
            if user_email and "@" in user_email and "." in user_email:
                with st.spinner("Sending PDF..." if language == Language.ENGLISH else "PDF gönderiliyor..."):
                    success = send_pdf_via_email(pdf_bytes, filename, user_email, language)
                
                if success:
                    st.success(success_msg)
                    st.session_state.show_email_form = False
                    st.rerun()
                else:
                    st.error(error_msg)
            else:
                st.error(invalid_email)
        
        if cancelled:
            st.session_state.show_email_form = False
            st.rerun()


def send_pdf_via_email(pdf_bytes: bytes, filename: str, recipient_email: str, language: Language) -> bool:
    """Send PDF via email - simplified and reliable"""
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        
        # Get email credentials
        sender_email = st.secrets.get("GMAIL_EMAIL") or os.getenv("GMAIL_EMAIL")
        sender_password = st.secrets.get("GMAIL_APP_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD")

        if not sender_email or not sender_password:
            st.error("Email configuration missing")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        # Email content based on language
        if language == Language.TURKISH:
            msg['Subject'] = f"İş Uyumluluk Raporu - {filename}"
            body = f"""
            Merhaba,

            Talep etmiş olduğunuz iş uyumluluk raporu ektedir.

            📄 Dosya Adı: {filename}

            Bu rapor, Selman Dedeakayoğulları'nın AI portföy asistanı tarafından otomatik olarak oluşturulmuştur.

            Herhangi bir sorunuz olması durumunda lütfen iletişime geçmekten çekinmeyin.

            Eğer bu e-postayı siz talep etmediyseniz, e-posta adresiniz yanlışlıkla girilmiş olabilir. Bu durumda lütfen bu mesajı dikkate almayınız.

            Saygılarımla,

            ---
            Selman Dedeakayoğulları  
            AI Portfolio Assistant
            """
        else:
            msg['Subject'] = f"Job Compatibility Report - {filename}"
            body = f"""
            Hello,

            Please find attached the job compatibility report you requested.

            📄 File Name: {filename}

            This report was automatically generated by Selman Dedeakayoğulları's AI portfolio assistant.

            If you have any questions, feel free to reach out.

            If you did not request this email, it's possible that your address was entered by mistake. In that case, please disregard this message.

            Best regards,

            ---
            Selman Dedeakayoğulları  
            AI Portfolio Assistant
            """
        
        # Attach text body
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # Attach PDF
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename="{filename}"'
        )
        msg.attach(part)
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
        
        return True
        
    except Exception as e:
        st.error(f"Email sending failed: {str(e)}")
        return False


def initialize_session_state() -> None:
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Hello! I'm here to answer questions about Selman. What would you like to know?\n\n "
                "Merhaba! Selman hakkında sorularınızı yanıtlayabilirim. Ne öğrenmek istersiniz?"
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
    if prompt := st.chat_input("Start chatting... "):
        chat_interface.process_user_input(prompt)
    
    # PDF download button
    render_pdf_download()
    
    # Sidebar
    render_sidebar(rag_system)


if __name__ == "__main__":
    main()