# Core imports
import streamlit as st
import os
import json
import time
import datetime 
import uuid
import base64
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client
from st_copy import copy_button
# Local imports
from tools.email_tool import EmailTool
from tools.social_media_tool import SocialMediaAggregator
from tools.tool_definitions import ToolDefinitions
from ui.email_components import get_ui_text, render_email_verification_card, render_email_editor_card

load_dotenv()
class BugReportManager:
    """Handle bug report submissions to Supabase"""
    
    def __init__(self):
        self.supabase_url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        self.supabase_key = st.secrets.get("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self.client: Optional[Client] = None
        self.configured = False
        
        if self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                self.configured = True
            except Exception as e:
                st.error(f"Failed to initialize Supabase: {e}")
                self.configured = False
    
    def _prepare_chat_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare chat history for storage (limit size and clean data)"""
        # Son 10 mesajı al
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        # Her mesajı temizle ve boyutunu sınırla
        cleaned_messages = []
        for msg in recent_messages:
            content = msg.get('content', '')
            # İçeriği 500 karakterle sınırla
            if len(content) > 500:
                content = content[:497] + "..."
            
            cleaned_messages.append({
                'role': msg.get('role', 'unknown'),
                'content': content,
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        return cleaned_messages
    
    def submit_bug_report(self, description: str, language: str = "en") -> Dict[str, Any]:
        """Submit a bug report to Supabase"""
        if not self.configured:
            return {"success": False, "message": "Bug reporting not configured" if language=="en" else "Hata raporlama sistemi hazır değil."}
        
        try:
            # Get session info
            session_id = st.session_state.get('session_id', str(uuid.uuid4()))
            if 'session_id' not in st.session_state:
                st.session_state.session_id = session_id
            
            # Sohbet geçmişini hazırla
            chat_history = self._prepare_chat_history(
                st.session_state.get('messages', [])
            )

            
            # Prepare bug report data
            bug_data = {
                "user_session_id": session_id,
                "description": description.strip(),
                "chat_history": chat_history,  # JSON olarak sakla
                "language": language,
                "page_url": "Portfolio RAG Chatbot",
                "status": "open",
            }
            
            # Insert into Supabase
            result = self.client.table('bug_reports').insert(bug_data).execute()
            
            if result.data:
                return {
                    "success": True, 
                    "message": "Bug report submitted successfully",
                    "report_id": result.data[0]['id']
                }
            else:
                return {"success": False, "message": "Failed to submit bug report"}
                
        except Exception as e:
            return {"success": False, "message": f"Error submitting bug report: {str(e)}"}


def get_bug_svg():
    """Return SVG icon for bug report button"""
    return """
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
        <path d="M19.5 11.5C19.78 11.17 20.03 10.8 20.24 10.4L22 11.4L21.5 12.3L20.04 11.46C20.13 11.97 20.17 12.5 20.17 13.03H22V14.03H20.17C20.17 14.56 20.13 15.09 20.04 15.6L21.5 16.44L21 17.34L19.24 16.34C19.03 16.74 18.78 17.09 18.5 17.42V20H17.5V17.84C17.09 18.03 16.65 18.17 16.2 18.27L16.6 19.72L15.65 20L15.25 18.55C14.84 18.63 14.42 18.67 14 18.67C13.58 18.67 13.16 18.63 12.75 18.55L12.35 20L11.4 19.72L11.8 18.27C11.35 18.17 10.91 18.03 10.5 17.84V20H9.5V17.42C9.22 17.09 8.97 16.74 8.76 16.34L7 17.34L6.5 16.44L7.96 15.6C7.87 15.09 7.83 14.56 7.83 14.03H6V13.03H7.83C7.83 12.5 7.87 11.97 7.96 11.46L6.5 10.6L7 9.7L8.76 10.7C8.97 10.3 9.22 9.95 9.5 9.62V7H10.5V9.2C10.91 9.01 11.35 8.87 11.8 8.77L11.4 7.32L12.35 7.04L12.75 8.49C13.16 8.41 13.58 8.37 14 8.37C14.42 8.37 14.84 8.41 15.25 8.49L15.65 7.04L16.6 7.32L16.2 8.77C16.65 8.87 17.09 9.01 17.5 9.2V7H18.5V9.62C18.78 9.95 19.03 10.3 19.24 10.7L21 9.7L21.5 10.6L20.04 11.46M14 16C15.66 16 17 14.66 17 13S15.66 10 14 10 11 11.34 11 13 12.34 16 14 16M14 12C14.55 12 15 12.45 15 13S14.55 14 14 14 13 13.55 13 13 13.45 12 14 12Z"/>
    </svg>
    """



class AppConstants:
    """Application-wide constants"""
    MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
    EMBEDDING_MODEL = "models/text-embedding-004"
    DEFAULT_TEMPERATURE = 0.1
    MAX_OUTPUT_TOKENS = 1200
    NUM_CONTEXT_MESSAGES = 10
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


# Translation system for system messages
def get_system_text(language_code: str) -> Dict[str, str]:
    """Get system texts based on language"""
    if language_code == "tr":
        return {
            # Connection and setup errors
            "connection_error": "❌ Chatbot'a bağlanırken sorun yaşıyoruz.",
            "setup_failed": "❌ Chatbot kurulumu başarısız: {error}",
            
            # File and data errors
            "cv_file_not_found": "❌ CV dosyası '{file_path}' bulunamadı.",
            "json_empty": "❌ JSON dosyası boş veya okunamaz",
            "json_parse_error": "JSON ayrıştırma hatası: {error}",
            "cv_load_error": "CV yükleme hatası: {error}",
            
            # Cache messages
            "cache_corrupted": "⚠️ Önbellek bozulmuş, embeddings yeniden oluşturuluyor...",
            "cache_not_found": "🔄 Önbellek bulunamadı veya geçersiz, embeddings oluşturuluyor...",
            "generating_embeddings": "{count} chunk için embeddings oluşturuluyor...",
            "saving_to_cache": "Embeddings önbelleğe kaydediliyor...",
            "cache_success": "✅ {count} chunk oluşturuldu ve önbelleğe kaydedildi!",
            "cache_save_failed": "⚠️ Embeddings oluşturuldu ancak önbellekleme başarısız",
            "embedding_generation_failed": "❌ Embeddings oluşturulamadı",
            "cache_cleared": "🗑️ Önbellek başarıyla temizlendi!",
            
            # Cache warnings
            "cache_info_save_failed": "Önbellek bilgisi kaydedilemedi: {error}",
            "cache_load_failed": "Önbellekten yüklenemedi: {error}",
            "cache_save_error": "Önbelleğe kaydedilemedi: {error}",
            "cache_file_remove_failed": "Önbellek dosyası {file_path} silinemedi: {error}",
            
            # Processing messages
            "processing_request": "İsteğiniz işleniyor...",
            "sending_email": "E-posta gönderiliyor...",
            
            # PDF messages
            "pdf_generated": "✅ PDF raporu başarıyla oluşturuldu! Aşağıdaki butona tıklayarak indirebilirsiniz.",
            "pdf_title": "📄 PDF Raporu Hazır!",
            "pdf_view": "👁️ PDF'yi Görüntüle",
            "pdf_download": "💾 PDF İndir",
            "pdf_mobile_tip": "📱 Mobilde PDF görüntüleme önerilir!",
            "pdf_email": "📧 Email Gönder",
            "pdf_clear": "🗑️ Temizle",
            
            # Email form
            "email_form_title": "📧 PDF'i Email ile Alın",
            "email_label": "Email Adresiniz:",
            "email_placeholder": "ornek@email.com",
            "email_send": "📧 PDF'i Gönder",
            "email_cancel": "❌ İptal",
            "email_success": "✅ PDF başarıyla gönderildi! Email'inizi kontrol edin.",
            "email_error": "❌ Email gönderilirken hata oluştu.",
            "email_invalid": "❌ Geçerli bir email adresi girin.",
            "email_sending": "PDF gönderiliyor...",
            
            # Email configuration
            "email_not_configured": "⚠️ Email işlevselliği yapılandırılmamış. Lütfen EMAIL_USER ve EMAIL_PASSWORD ortam değişkenlerini ayarlayın.",
            "email_config_missing": "Email yapılandırması eksik",
            
            # Main app messages
            "initializing_chatbot": "Chatbot başlatılıyor",
            "configure_api_key": "Devam etmek için GEMINI_API_KEY'i yapılandırın",
            
            # Sidebar
            "last_update":f"Son Güncelleme: "+str(datetime.datetime.now())[:-7],
            "sidebar_title": "🔍 Meraklı biriymişsin",
            "cache_status": "💾 Önbellek Durumu",
            "cache_active": "✅ Aktif",
            "cache_chunks": "Chunk'lar",
            "cache_size": "Boyut",
            "cache_cached": "Önbelleğe Alındı",
            "cache_refresh": "🔄 Yenile",
            "cache_refresh_help": "Embeddings'i yeniden oluştur",
            "cache_clear_btn": "🗑️ Temizle",
            "cache_clear_help": "Önbelleği temizle",
            "cache_no_cache": "❌ Önbellek yok",
            "view_chunks": "🔍 Oluşturulan Chunk'ları Görüntüle",
            "chunks_title": "📋 Oluşturulan Chunk'lar",
            "chunks_not_available": "Chunk'lar mevcut değil",
            "chunks_loaded": "Yüklenen chunk'lar",
            "cache_cleared_success": "Önbellek temizlendi!",
            
            # Error messages
            "api_retry": "🔄 API'den yanıt alamadım, tekrar deniyorum...",
            "api_retry_failed": "❌ API'den birkaç denemeye rağmen yanıt alınamadı. Lütfen daha sonra tekrar deneyin.",
            "embedding_error": "Embedding hatası: {error}",
            "error_generating_response": "Yanıt oluşturulurken hata: {error}. API yanıtı boş veya geçersiz olabilir. Lütfen birkaç saniye sonra tekrar deneyin.",
            "embeddings_not_available": "Embeddings mevcut değil",
            "query_process_failed": "Sorgu işlenemedi",
            "no_response_generated": "Yanıt oluşturulamadı, lütfen birazdan tekrar deneyin.",
            "api_not_configured": "Gemini API yapılandırılmamış"
        }
    else:  # English
        return {
            # Connection and setup errors
            "connection_error": "❌ We are having trouble connecting to Chatbot.",
            "setup_failed": "❌ Chatbot setup failed: {error}",
            
            # File and data errors
            "cv_file_not_found": "❌ CV file '{file_path}' not found.",
            "json_empty": "❌ JSON file is empty or unreadable",
            "json_parse_error": "Error parsing JSON: {error}",
            "cv_load_error": "Error loading CV: {error}",
            
            # Cache messages
            "cache_corrupted": "⚠️ Cache corrupted, regenerating embeddings...",
            "cache_not_found": "🔄 Cache not found or invalid, generating embeddings...",
            "generating_embeddings": "Generating embeddings for {count} chunks...",
            "saving_to_cache": "Saving embeddings to cache...",
            "cache_success": "✅ Generated and cached {count} chunks!",
            "cache_save_failed": "⚠️ Embeddings generated but caching failed",
            "embedding_generation_failed": "❌ Failed to generate embeddings",
            "cache_cleared": "🗑️ Cache cleared successfully!",
            
            # Cache warnings
            "cache_info_save_failed": "Could not save cache info: {error}",
            "cache_load_failed": "Could not load from cache: {error}",
            "cache_save_error": "Could not save to cache: {error}",
            "cache_file_remove_failed": "Could not remove cache file {file_path}: {error}",
            
            # Processing messages
            "processing_request": "Processing your request...",
            "sending_email": "Sending email...",
            
            # PDF messages
            "pdf_generated": "✅ PDF report generated successfully! You can download it using the button below.",
            "pdf_title": "📄 PDF Report Ready!",
            "pdf_view": "👁️ View PDF",
            "pdf_download": "💾 Download PDF",
            "pdf_mobile_tip": "📱 PDF viewing recommended on mobile!",
            "pdf_email": "📧 Email PDF",
            "pdf_clear": "🗑️ Clear",
            
            # Email form
            "email_form_title": "📧 Get PDF via Email",
            "email_label": "Your Email:",
            "email_placeholder": "example@email.com",
            "email_send": "📧 Send PDF",
            "email_cancel": "❌ Cancel",
            "email_success": "✅ PDF sent successfully! Check your email.",
            "email_error": "❌ Failed to send email.",
            "email_invalid": "❌ Please enter a valid email address.",
            "email_sending": "Sending PDF...",
            
            # Email configuration
            "email_not_configured": "⚠️ Email functionality is not configured. Please set EMAIL_USER and EMAIL_PASSWORD environment variables.",
            "email_config_missing": "Email configuration missing",
            
            # Main app messages
            "initializing_chatbot": "Initializing Chatbot",
            "configure_api_key": "Please configure GEMINI_API_KEY to continue",

            
            # Sidebar
            "last_update":f"Lastly Updated: "+str(datetime.datetime.now())[:-7],
            "sidebar_title": "🔍 Okay, okay... Mr.Curious.",
            "cache_status": "💾 Cache Status",
            "cache_active": "✅ Active",
            "cache_chunks": "Chunks",
            "cache_size": "Size",
            "cache_cached": "Cached",
            "cache_refresh": "🔄 Refresh",
            "cache_refresh_help": "Regenerate embeddings",
            "cache_clear_btn": "🗑️ Clear",
            "cache_clear_help": "Clear cache",
            "cache_no_cache": "❌ No cache",
            "view_chunks": "🔍 View Generated Chunks",
            "chunks_title": "📋 Generated Chunks",
            "chunks_not_available": "No chunks available",
            "chunks_loaded": "Chunks loaded",
            "cache_cleared_success": "Cache cleared!",
            
            # Error messages
            "api_retry": "🔄 Didn't get response from API, retrying...",
            "api_retry_failed": "❌ Failed to get response from API after multiple attempts. Please try again later.",
            "embedding_error": "Embedding error: {error}",
            "error_generating_response": "Error generating response: {error}. The API response might have been empty or invalid. Please wait a moment and try again.",
            "embeddings_not_available": "Embeddings not available",
            "query_process_failed": "Could not process query",
            "no_response_generated": "No response generated, please try again in a moment.",
            "api_not_configured": "Gemini API not configured"
        }


@dataclass
class QueryType:
    """Query type detection result"""
    is_social_query: bool = False
    is_job_query: bool = False
    is_project_query: bool = False
    is_experience_query: bool = False
    is_education_query: bool = False
    is_contact_query: bool = False
    is_awards_query: bool = False


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
    
    def _save_cache_info(self, cv_file_path: str, cv_hash: str, chunks_count: int, language: Language) -> None:
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
            system_text = get_cached_system_text(language.value)
            st.warning(system_text["cache_info_save_failed"].format(error=e))
    
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
    
    def load_from_cache(self, language: Language) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
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
            system_text = get_cached_system_text(language.value)
            st.warning(system_text["cache_load_failed"].format(error=e))
            return None, None
    
    def save_to_cache(self, cv_file_path: str, chunks: List[str], embeddings: np.ndarray, language: Language) -> bool:
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
            self._save_cache_info(cv_file_path, cv_hash, len(chunks), language)
            
            return True
            
        except Exception as e:
            system_text = get_cached_system_text(language.value)
            st.error(system_text["cache_save_error"].format(error=e))
            return False
    
    def clear_cache(self, language: Language) -> None:
        """Clear all cached files"""
        for file_path in [self.embeddings_path, self.chunks_path, self.cache_info_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                system_text = get_cached_system_text(language.value)
                st.warning(system_text["cache_file_remove_failed"].format(file_path=file_path, error=e))
    
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
    
    # Pre-compiled sets for faster lookup
    TURKISH_CHARS = frozenset('çğıöşü')
    TURKISH_KEYWORDS = frozenset({
        'hakkında', 'nedir', 'kimdir', 'nasıl', 'merhaba', 'teşekkür', 'iletişim', 'mesaj', 'gönder',
        'anlat', 'söyle', 'nerede', 'ne zaman', 'hangi', 'proje', 'projeler', 'deneyim', 'eğitim',
        'çalışma', 'iş', 'üniversite', 'okul', 'mezun', 'değil', 'yok', 'var', 'olan', 'yapan',
        'merhabalar', 'selam', 'günaydın', 'teşekkürler', 'sağol', 'kariyer', 'bilgi', 'selamlar',
        'anladım', 'bilmiyorum', 'istiyorum', 'isterim', 've', 'bir', 'bu', 'şu', 'o', 'ben', 'sen',
        'ile', 'için', 'ama', 'fakat', 'lakin', 'çünkü', 'ki', 'da', 'de', 'ta', 'te', 'neler', 'yap'
    })
    
    ENGLISH_KEYWORDS = frozenset({
        'hello', 'hi', 'what', 'who', 'when', 'where', 'why', 'how', 'about', 'thank', 'thanks',
        'tell', 'show', 'project', 'experience', 'work', 'education', 'university', 'job', 'i', 'you',
        'know', 'dont', "don't", 'want', 'need', 'can', 'could', 'would', 'should', 'the', 'and',
        'with', 'for', 'but', 'because', 'that', 'this', 'they', 'we', 'he', 'she', 'it', 'my', 'your'
    })
    
    TURKISH_GREETINGS = frozenset({'selam', 'merhaba', 'merhabalar', 'selamlar', 'günaydın', 'iyi günler', 'meraba'})
    ENGLISH_GREETINGS = frozenset({'hello', 'hi', 'hey', 'greetings', 'good morning', 'good day'})
    
    # Cache for recent detections
    _cache = {}
    _cache_size_limit = 100
    
    @classmethod
    def detect_from_text(cls, text: str) -> Language:
        """Detect language from a single text with caching"""
        if not text:
            return Language.ENGLISH
        
        # Check cache first
        text_hash = hash(text.lower().strip()[:50])  # Hash first 50 chars for cache key
        if text_hash in cls._cache:
            return cls._cache[text_hash]
        
        # Clean cache if too large
        if len(cls._cache) > cls._cache_size_limit:
            cls._cache.clear()
        
        text_lower = text.lower().strip()
        result = Language.ENGLISH  # Default
        
        # Quick checks for very short messages
        if len(text_lower) <= 3:
            if text_lower in {'hi', 'hey'}:
                result = Language.ENGLISH
            elif text_lower in {'selam', 'mrb'}:
                result = Language.TURKISH
        else:
            # Check greetings first (fastest)
            if text_lower in cls.TURKISH_GREETINGS:
                result = Language.TURKISH
            elif text_lower in cls.ENGLISH_GREETINGS:
                result = Language.ENGLISH
            # Turkish character detection (very strong indicator)
            elif any(char in text_lower for char in cls.TURKISH_CHARS):
                result = Language.TURKISH
            else:
                # Keyword scoring - optimized
                text_words = frozenset(text_lower.split())
                turkish_score = len(text_words & cls.TURKISH_KEYWORDS) * 2
                english_score = len(text_words & cls.ENGLISH_KEYWORDS)
                
                if turkish_score > english_score:
                    result = Language.TURKISH
        
        # Cache result
        cls._cache[text_hash] = result
        return result
    
    @classmethod
    def detect_from_messages(cls, messages: List[Dict[str, str]]) -> Language:
        """Detect language from conversation history - optimized"""
        if not messages:
            return Language.ENGLISH
        
        # Get last user message only for efficiency
        for msg in reversed(messages[-5:]):  # Check only last 5 messages
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
    AWARDS_KEYWORDS = {
        'award', 'awards', 'prize', 'competition', 'contest', 'achievement',
        'ödül', 'ödüller', 'yarışma', 'başarı', 'kazandı', 'aldı', 'ödüllü'
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
            is_contact_query=any(kw in query_lower for kw in cls.CONTACT_KEYWORDS),
            is_awards_query=any(kw in query_lower for kw in cls.AWARDS_KEYWORDS) 
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
        
        # Handle both degree and program fields
        degree_info = edu.get('degree') or edu.get('program', 'N/A')
        edu_text += f"Degree/Program/Derece: {degree_info}\n"
        
        year_info = edu.get('years') or edu.get('year', 'N/A')
        edu_text += f"Years/Duration/Süre: {year_info}\n"
        
        # Add GPA if available
        if gpa := edu.get('gpa'):
            edu_text += f"GPA/Başarı Notu: {gpa}\n"
        
        location_info = edu.get('location', 'N/A')
        edu_text += f"Location/Konum: {location_info}\n"
        
        # Add description if available (for exchange programs)
        if description := edu.get('description'):
            edu_text += f"Description/Açıklama: {description}\n"
        
        # Format memberships more clearly
        if memberships := edu.get('memberships'):
            edu_text += f"Memberships/Üyelikler:\n"
            for membership in memberships:
                edu_text += f"- {membership}\n"
        
        # Enhanced keywords
        keywords = [
            "education", "eğitim", "university", "üniversite", "degree", "derece", 
            "diploma", "bachelor", "lisans", "graduate", "mezun", "student", "öğrenci",
            edu.get('institution', '').lower().replace(' ', '_')
        ]
        if "exchange" in degree_info.lower() or "erasmus" in degree_info.lower():
            keywords.extend(["exchange", "erasmus", "study abroad", "yurtdışı eğitim"])
        
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
    @staticmethod
    def build_awards_chunk(award: Dict[str, Any]) -> str:
        """Build award chunk with enhanced keywords"""
        award_text = f"""Award / Ödül:
    Award Name/Ödül Adı: {award.get('name', 'N/A')}
    Organization/Organizasyon: {award.get('organization', 'N/A')}
    Description/Açıklama: {award.get('description', 'N/A')}
    Year/Yıl: {award.get('year', 'N/A')}
    Keywords: award, ödül, prize, yarışma, competition, achievement, başarı, kazandı, {award.get('name', '').lower()}, {award.get('organization', '').lower()}"""
        
        return award_text
@st.cache_data
def get_cached_system_text(language_code: str) -> Dict[str, str]:
    """Cached version of system text to avoid repeated calls"""
    return get_system_text(language_code)

def get_current_language() -> Language:
    """Get current language with efficient caching"""
    if 'current_language' not in st.session_state:
        detected = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
        st.session_state.current_language = detected
    return st.session_state.current_language

def update_language_if_needed(new_language: Language) -> bool:
    """Update language only if changed, return True if updated"""
    current = st.session_state.get('current_language', Language.ENGLISH)
    if current != new_language:
        st.session_state.current_language = new_language
        return True
    return False

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
                # Get language for error message
                language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
                system_text = get_cached_system_text(language.value)
                st.error(system_text["connection_error"])
        except Exception as e:
            language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
            system_text = get_cached_system_text(language.value)
            st.error(system_text["setup_failed"].format(error=e))
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
            language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
            system_text = get_system_text(language.value)
            st.error(system_text["embedding_error"].format(error=e))
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
        if awards := data.get('awards', []):
            for award in awards:
                chunks.append(self.chunk_builder.build_awards_chunk(award))
            
            # Awards summary chunk ekleyin
            summary = "All Awards and Achievements / Tüm Ödüller ve Başarılar:\n"
            for i, award in enumerate(awards, 1):
                summary += f"{i}. {award.get('name', 'N/A')} - {award.get('organization', 'N/A')}"
                if year := award.get('year'):
                    summary += f" ({year})"
                summary += "\n"
            summary += "\nKeywords: all awards, tüm ödüller, achievements, başarılar, competitions, yarışmalar"
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
                language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
                system_text = get_cached_system_text(language.value)
                st.error(system_text["cv_file_not_found"].format(file_path=self.json_path))
                return
            
            # Load CV data
            with open(self.json_path, 'r', encoding='utf-8') as file:
                self.cv_data = json.load(file)
            
            if not self.cv_data:
                language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
                system_text = get_cached_system_text(language.value)
                st.error(system_text["json_empty"])
                return
            
            # Detect language for messages
            language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
            
            # Check if cache is valid
            if self.cache.is_cache_valid(self.json_path):
                cached_chunks, cached_embeddings = self.cache.load_from_cache(language)
                
                if cached_chunks is not None and cached_embeddings is not None:
                    self.cv_chunks = cached_chunks
                    self.cv_embeddings = cached_embeddings
                else:
                    system_text = get_cached_system_text(language.value)
                    st.warning(system_text["cache_corrupted"])
                    self._generate_fresh_embeddings(language)
            else:
                system_text = get_cached_system_text(language.value)
                st.info(system_text["cache_not_found"])
                self._generate_fresh_embeddings(language)
            
            # Initialize job compatibility analyzer
            if self.cv_embeddings is not None and self.cv_embeddings.size > 0:
                self.tool_definitions.initialize_job_analyzer(
                    self.client, 
                    self.cv_data, 
                    self
                )
            else:
                system_text = get_cached_system_text(language.value)
                st.error(system_text["embedding_generation_failed"])
                
        except json.JSONDecodeError as e:
            language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
            system_text = get_cached_system_text(language.value)
            st.error(system_text["json_parse_error"].format(error=e))
        except Exception as e:
            language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
            system_text = get_cached_system_text(language.value)
            st.error(system_text["cv_load_error"].format(error=e))
    
    def _generate_fresh_embeddings(self, language: Language) -> None:
        """Generate fresh embeddings and cache them"""
        try:
            system_text = get_cached_system_text(language.value)
            
            # Convert to chunks
            self.cv_chunks = self.json_to_chunks(self.cv_data)
            
            # Generate embeddings with progress tracking
            with st.spinner(system_text["generating_embeddings"].format(count=len(self.cv_chunks))):
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
                    with st.spinner(system_text["saving_to_cache"]):
                        if self.cache.save_to_cache(self.json_path, self.cv_chunks, self.cv_embeddings, language):
                            st.success(system_text["cache_success"].format(count=len(self.cv_chunks)))
                        else:
                            st.warning(system_text["cache_save_failed"])
                else:
                    st.error(system_text["embedding_generation_failed"])
                    self.cv_embeddings = np.array([])
                    
        except Exception as e:
            system_text = get_cached_system_text(language.value)
            st.error(system_text["embedding_generation_failed"])
            self.cv_embeddings = np.array([])
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
        self.cache.clear_cache(language)
        system_text = get_cached_system_text(language.value)
        st.success(system_text["cache_cleared"])
    
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
             'ödül': ['award', 'ödül', 'prize', 'achievement', 'başarı'],
        'ödüller': ['award', 'ödül', 'prize', 'achievement', 'başarı'],
        'award': ['award', 'ödül', 'prize', 'achievement', 'başarı'],
        'awards': ['award', 'ödül', 'prize', 'achievement', 'başarı'],
        'yarışma': ['competition', 'contest', 'yarışma', 'award', 'ödül'],
        'kazandı': ['won', 'achieved', 'award', 'ödül', 'başarı'],
        'başarı': ['achievement', 'success', 'award', 'ödül', 'başarı'],
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
            language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
            system_text = get_cached_system_text(language.value)
            return [{"text": system_text["embeddings_not_available"], "similarity": 0.0, "index": -1}]
        
        # Get query embedding
        query_embedding = self.get_embeddings([query])
        if query_embedding.size == 0:
            language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
            system_text = get_cached_system_text(language.value)
            return [{"text": system_text["query_process_failed"], "similarity": 0.0, "index": -1}]
        
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
    
# Ana kodda _build_prompt fonksiyonunu güncelleyin:
    def _build_prompt(self, query: str, context: str, language: Language, recent_context: str) -> str:
        """Build appropriate prompt based on language"""
        if language == Language.TURKISH:
            return f"""Sen Selman Dedeakayoğulları'nın AI portföy asistanısın. Portföy web sitesine yerleştiriliyorsun. Ziyaretçiler sana sorular soracak. ASLA GEMINI KULLANDIĞINDAN YA DA GOOGLE TARAFINDAN GELİŞTİRİLDİĞİNDEN BAHSETME.
    Kurallar:
    - SADECE TÜRKÇE yanıtla
    - CV soruları için yalnızca sağlanan bağlamdan bilgi kullan
    - Profesyonel ve yardımsever ol
    - Netlik ve okunabilirlik için şık bir markdown biçimlendirmesini kullan
    - Kullanıcı referans isterse, bunları görüntüle ve talep üzerine iletişim bilgilerinin mevcut olduğuna dair bir not ekle
    - Projeler veya iş deneyimleri hakkında sorulduğunda, bağlamdan TÜM ilgili öğeleri listele
    - Proje soruları için, proje adlarını, kullanılan teknolojileri ve açıklamaları ekle. Özel olarak istenmediği sürece bağlantı verme.
    -"Agentic Portfolio Bot" hakkında konuşurken, onun sen olduğun için bununla ilgili bir şaka yap
    - Deneyim soruları için şirket adlarını, pozisyonları, süreleri ve açıklamaları ekle

    EMAIL KURALLARI - ÇOK ÖNEMLİ:
    - Birisi Selman ile iletişime geçmek istediğinde, prepare_email aracını KULLANMADAN ÖNCE şu bilgilerin TAMAMINI toplayın:
    1. Gönderenin tam adı (ad ve soyad gerekli.)
    2. Gönderenin e-posta adresi
    3. Mesaj içeriği (kullanıcının tam mesajını AYNEN ALMA, bunun yerine profesyonel bir şekilde yeniden formülle)
    - Bu bilgilerden HERHANGİ BİRİ eksikse, önce eksik bilgileri isteyin
    - Örnek: "E-posta gönderebilmem için adınızı, soyadınızı ve e-posta adresinizi öğrenebilir miyim?"
    - TÜM bilgiler toplandıktan SONRA prepare_email aracını kullanın
    - MESAJ FORMATLAMASI: Kullanıcının mesajını olduğu gibi kopyalama. Bunun yerine üçüncü şahıs bakış açısıyla profesyonel bir şekilde özetle ve yeniden formülle.
    - Örnek: <ad,soyad> "3 haziran toplantı yapalım" derse → "<ad,soyad> 3 Haziran tarihinde bir toplantı düzenlemek istiyor."
    - Örnek: <ad,soyad> "projelerinizle ilgili konuşalım" derse → "<ad,soyad> projeleriniz hakkında görüşme yapmak istiyor."

    İŞ UYUMLULUK ANALİZİ KURALLARI - ÇOK ÖNEMLİ:
    - Birisi bir iş tanımı SAĞLADIĞINDA (sadece analiz istemediğinde) analyze_job_compatibility aracını kullan
    - ÖNCE şunu kontrol et: Kullanıcı gerçek bir iş ilanı mı verdi yoksa sadece analiz mi istiyor?
    - Eğer sadece "iş uyumluluğu analizi yap" gibi bir istek varsa ve iş tanımı YOKSA, önce iş tanımını iste:
    "Size yardımcı olabilmem için lütfen analiz etmemi istediğiniz iş ilanının tamamını paylaşır mısınız?"
    - İş tanımı en az 50 karakter olmalı ve gerçek iş detayları içermeli
    - İş tanımı aldıktan SONRA rapor dilini sor: "Raporu Türkçe mi İngilizce mi istersiniz?"
    - Şirket adı belirtilmemişse MUTLAKA sor
    
    ÖNEMLİ NOTLAR:
    - Bir kullanıcı hem analiz hem de PDF isterse, önce analyze_job_compatibility aracını çağır, sonra generate_compatibility_pdf aracını otomatik olarak çağır.

    
    DİĞER ARAÇLAR:
    - Birisi Selman'ın son gönderileri, makaleleri, Medium içeriği veya sosyal medyası hakkında soru sorduğunda get_recent_posts aracını kullanın
    - Kullanıcı PDF istediğinde, indirdiğinde veya iş uyumluluk raporunu kaydetmek istediğinde generate_compatibility_pdf aracını kullanın

    Son Konuşma Bağlamı:
    {recent_context}

    CV Bağlamı:
    {context}

    Kullanıcı Sorusu: {query}
    Yanıt:"""
        else:
            return f"""You are Selman Dedeakayoğulları's AI portfolio assistant. You are embedded in his portfolio website. Visitors will ask questions to you. NEVER MENTION YOU USE GEMINI OR DEVELOPED BY GOOGLE.

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
    3. Message content (DO NOT copy user's exact message, instead reformulate it professionally)
    - If ANY of this information is missing, ask for the missing details first
    - Example: "I'd be happy to help you contact Selman. Could you please provide your full name and email address?"
    - ONLY use prepare_email tool after ALL information is collected
    - MESSAGE FORMATTING: Do not copy the user's message verbatim. Instead, reformulate and summarize it professionally from a third-person perspective.
    - Example: <name,surname> says "let's meet on June 3rd" → "<name,surname> would like to schedule a meeting on June 3rd."
    - Example: <name,surname> says "I want to discuss your projects" → "<name,surname> is interested in discussing your projects."
    
    JOB COMPATIBILITY ANALYSIS RULES - VERY IMPORTANT:
    - When someone PROVIDES a job description (not just asks for analysis), use analyze_job_compatibility tool
    - FIRST check: Did the user provide an actual job posting or just request an analysis?
    - If they only request "do job compatibility analysis" without a job description, ask for it first:
    "I'd be happy to help! Please share the complete job posting you'd like me to analyze."
    - Job description must be at least 50 characters and contain actual job details
    - After receiving job description, ask for report language: "Would you like the report in English or Turkish?"
    - If company name is not specified, ALWAYS ask for it
    
    IMPORTANT NOTES:
    - If a user requests both analysis and PDF, first call the analyze_job_compatibility tool, then automatically call the generate_compatibility_pdf tool.

    OTHER TOOLS:
    - Use get_recent_posts tool when someone asks about Selman's recent posts, articles, Medium content or social media
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
        recent_messages = conversation_history[-AppConstants.NUM_CONTEXT_MESSAGES:]
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
        """Handle function call from LLM response with support for multiple tools"""
        if not hasattr(part, 'function_call') or not part.function_call:
            return None
        
        tool_name = part.function_call.name
        tool_args = dict(part.function_call.args.items())
        
        result = self.tool_definitions.execute_tool(tool_name, tool_args)
        
        if not result["success"]:
            return f"❌ {result['message']}"
        
        # Handle tool results with potential chaining
        if tool_name == "analyze_job_compatibility":
            st.session_state.last_compatibility_report = result["data"]["report"]
            st.session_state.last_job_title = result["data"]["job_title"]
            
            # Check if we should automatically generate PDF
            if st.session_state.get("auto_generate_pdf", False):
                st.session_state.auto_generate_pdf = False  # Reset flag
                pdf_result = self.tool_definitions.execute_tool(
                    "generate_compatibility_pdf", {}
                )
                if pdf_result["success"]:
                    return "PDF_GENERATED"
            
            pdf_msg = (
                "\n\n📄 *Bu raporun PDF versiyonunu indirmek isterseniz söyleyebilirsiniz!*"
                if language == Language.TURKISH
                else "\n\n📄 *You can ask for a PDF version of this report if you'd like to download it!*"
            )
            return result["data"]["report"] + pdf_msg
        
        elif tool_name == "generate_compatibility_pdf":
            return "PDF_GENERATED"
        
        elif tool_name == "prepare_email":
            return "EMAIL_PREPARED_FOR_REVIEW"
        
        elif tool_name == "get_recent_posts":
    # Store posts data for rendering cards
            if result["data"].get("render_cards", False):
                st.session_state.social_media_posts = result["data"]["posts"]
                return "SOCIAL_MEDIA_POSTS_READY"  # 👈 BU ÖZEL RETURN
            else:
                return result["data"]["formatted_response"]
        
        
        return None
    
  
    def generate_response(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response with tool calling capability, Turkish support, and retry mechanism"""
        if not self.configured:
            language = LanguageDetector.detect_from_messages(conversation_history or [])
            system_text = get_cached_system_text(language.value)
            return system_text["api_not_configured"]
        
        # 1. PDF generation kontrolü ekleyin (YENİ KOD)
        pdf_keywords = {
            "tr": ["pdf", "indir", "kaydet", "rapor al"],
            "en": ["pdf", "download", "save", "get report"]
        }
        
        # Dil algılama
        messages = (conversation_history or []) + [{"role": "user", "content": query}]
        language = LanguageDetector.detect_from_messages(messages)
        job_analysis_keywords = {
    'en': ['job compatibility', 'job analysis', 'analyze job', 'compatibility report'],
    'tr': ['iş uyumu', 'iş analizi', 'uyumluluk raporu', 'iş uyumluluk']
}

# Check if user is asking for job analysis without providing a job
        query_lower = query.lower()
        is_job_request = any(keyword in query_lower for keyword in job_analysis_keywords.get(language.value, job_analysis_keywords['en']))

        # If it's a job request but the message is too short to contain a real job description
        if is_job_request and len(query) < 100:
            if language == Language.TURKISH:
                return "Memnuniyetle iş uyumluluk analizi yapabilirim! Lütfen analiz etmemi istediğiniz iş ilanının tamamını paylaşır mısınız? İlan metninde görev tanımı, gereksinimler ve nitelikler bulunmalıdır."
            else:
                return "I'd be happy to analyze job compatibility! Please share the complete job posting you'd like me to analyze. The posting should include job responsibilities, requirements, and qualifications."
        # PDF isteği kontrolü
        if any(kw in query.lower() for kw in pdf_keywords.get(language.value, pdf_keywords["en"])):
            st.session_state.auto_generate_pdf = True
        
        # 2. Orijinal kodun devamı (aşağıdaki kısmı değiştirmeyin)
        recent_context = self._get_recent_context(conversation_history or [])
        
        # Classify query
        query_type = self.query_classifier.classify(query)
        
        # Get relevant chunks
        top_k = self._determine_top_k(query_type)
        relevant_chunks = self.search_similar_chunks(query, top_k=top_k)
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Build prompt
        prompt = self._build_prompt(query, context, language, recent_context)
        
        # System text for error messages
        system_text = get_cached_system_text(language.value)
        
        # Retry mechanism
        max_retries = 2
        for attempt in range(max_retries):
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
                
                # Check if response is valid
                if response is None:
                    raise Exception("API returned None response")
                
                # Check for function calls - with safe access
                if (hasattr(response, 'candidates') and response.candidates 
                    and len(response.candidates) > 0 
                    and hasattr(response.candidates[0], 'content')
                    and response.candidates[0].content
                    and hasattr(response.candidates[0].content, 'parts')
                    and response.candidates[0].content.parts):
                    
                    for part in response.candidates[0].content.parts:
                        if function_result := self._handle_function_call(part, language):
                            return function_result
                
                # Return text response - with safe access
                if hasattr(response, 'text') and response.text:
                    return response.text
                else:
                    # If no text, might be an empty response
                    if attempt < max_retries - 1:  # Not the last attempt
                        raise Exception("Empty response from API")
                    else:
                        return system_text["no_response_generated"]
                    
            except Exception as e:
                if attempt < max_retries - 1:  # Not the last attempt
                    # Show retry message
                    retry_message = (
                        "🔄 API'den yanıt alamadım, tekrar deniyorum..." 
                        if language == Language.TURKISH 
                        else "🔄 Didn't get response from API, retrying..."
                    )
                    
                    # Display retry message in UI
                    if 'messages' in st.session_state:
                        # Add temporary message
                        temp_placeholder = st.empty()
                        with temp_placeholder.container():
                            st.info(retry_message)
                        
                        # Wait 1 second
                        time.sleep(1)
                        
                        # Clear temporary message
                        temp_placeholder.empty()
                    
                    continue  # Try again
                else:
                    # Last attempt failed, return error
                    return system_text["error_generating_response"].format(error=str(e))
        
        # Should never reach here, but just in case
        return system_text["no_response_generated"]


class ChatInterface:
    """Manage chat interface and interactions"""
    
    def __init__(self, rag_system: GeminiEmbeddingRAG):
        self.rag_system = rag_system
    
    def handle_email_actions(self) -> None:
        """Handle email-related actions"""
        if "email_action" not in st.session_state or not st.session_state.email_action:
            return
        
        action = st.session_state.email_action
        # Session state'deki mevcut dili kullan
        if 'current_language' in st.session_state:
            language = st.session_state.current_language
        else:
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
        """Display chat messages with special handling for emails and copy functionality"""
        # Session state'deki mevcut dili kullan
        if 'current_language' in st.session_state:
            language = st.session_state.current_language
        else:
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
                        # Use a more stable approach with columns
                    content_col, copy_col = st.columns([0.95, 0.05],vertical_alignment="center") # 95% for content, 5% for button
                    
                    with content_col:
                        st.markdown(message["content"])
                    
                    with copy_col:
                        # Use a simpler, more stable key for the button.
                        message_key = f"cp_{i}" 
                        
                        # The copy_button is now placed in its own pre-allocated column.
                        copy_button(
                            text=message["content"],
                            tooltip="Copy" if language == Language.ENGLISH else "Kopyala",
                            copied_label="✓",
                            key=message_key,
                        )
    def process_user_input(self, prompt: str) -> None:
        """Process user input and generate response"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Önceki dil durumunu kaydet
        previous_language = st.session_state.get('current_language', Language.ENGLISH)
        
        # Check if user is requesting a report in a specific language
        requested_language = None
        prompt_lower = prompt.lower()
        
        # Check for Turkish report request
        turkish_indicators = ['turkish', 'türkçe', 'turkce', 'report in turkish', 'raporu türkçe']
        english_indicators = ['english', 'ingilizce', 'report in english', 'raporu ingilizce']
        
        for indicator in turkish_indicators:
            if indicator in prompt_lower:
                requested_language = Language.TURKISH
                # Store the requested language preference
                st.session_state.preferred_language = Language.TURKISH
                st.session_state.current_language = Language.TURKISH
                break
        
        if not requested_language:
            for indicator in english_indicators:
                if indicator in prompt_lower:
                    requested_language = Language.ENGLISH
                    # Store the requested language preference
                    st.session_state.preferred_language = Language.ENGLISH
                    st.session_state.current_language = Language.ENGLISH
                    break
        
        # Eğer dil indicator'ı yoksa, mevcut mesajdan dil algıla ve güncelle
        if not requested_language:
            detected_language = LanguageDetector.detect_from_messages(st.session_state.messages)
            st.session_state.current_language = detected_language
            language = detected_language
        else:
            language = requested_language
        
        # Dil değişimini kontrol et
        current_language = st.session_state.get('current_language', Language.ENGLISH)
        language_changed = (previous_language != current_language)
        
        system_text = get_cached_system_text(language.value)
        
        # Generate response
        with st.chat_message("assistant"):
            spinner_msg = system_text["processing_request"]
            
            with st.spinner(spinner_msg):
                response = self.rag_system.generate_response(
                    prompt, 
                    st.session_state.messages
                )
            
            # Handle special responses
            if response == "EMAIL_PREPARED_FOR_REVIEW":
                ui_text = get_ui_text(language.value)
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
                message = system_text["pdf_generated"]
                st.write(message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": message
                })
                
            elif response == "SOCIAL_MEDIA_POSTS_READY":  # 👈 YENİ CASE
                # Render social media cards
                if "social_media_posts" in st.session_state:
                    posts = st.session_state.social_media_posts
                    
                    # Render beautiful cards
                    self.rag_system.social_media_aggregator.render_posts_cards(
                        posts, 
                        language.value
                    )
                    
                    # Also add text response to chat history for context
                    formatted_text = self.rag_system.social_media_aggregator.format_posts_for_chat(
                        posts, 
                        language.value
                    )
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_text
                    })
                    
                    # Clean up
                    del st.session_state.social_media_posts
                else:
                    st.write("📭 No posts found")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "📭 No posts found"
                    })    
            
            else:
                st.write(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
        
        # Dil değiştiyse sayfayı yenile
        if language_changed:
            st.rerun()


def render_sidebar(rag_system: GeminiEmbeddingRAG) -> None:
    """Render sidebar with system information and cache controls"""
    # Session state'deki mevcut dili kullan
    if 'current_language' in st.session_state:
        language = st.session_state.current_language
    else:
        language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
    system_text = get_cached_system_text(language.value)
    
    with st.sidebar:
        st.info(system_text["last_update"])
        st.markdown(f"### {system_text['sidebar_title']}")
        st.markdown("- **Embeddings**: text-embedding-004")
        st.markdown("- **Generation**: gemini-2.5-flash-lite-preview-06-17")
        st.markdown("- **Vector dims**: 768")
        st.markdown("- **Search**: Cosine similarity")
        st.markdown("- **Data Source**: JSON")
        
        # Cache information
        if rag_system.configured:
            cache_stats = rag_system.get_cache_stats()
            st.markdown(f"### {system_text['cache_status']}")
            
            if cache_stats["cache_info"]:
                cache_info = cache_stats["cache_info"]
                st.markdown(f"- **Status**: {system_text['cache_active']}")
                st.markdown(f"- **{system_text['cache_chunks']}**: {cache_info.get('chunks_count', 'N/A')}")
                st.markdown(f"- **{system_text['cache_size']}**: {cache_stats['cache_size'] / 1024:.1f} KB")
                st.markdown(f"- **{system_text['cache_cached']}**: {cache_info.get('cached_at', 'N/A')[:16]}")
                
                # Cache actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(system_text["cache_refresh"], help=system_text["cache_refresh_help"],use_container_width=True):
                        rag_system.clear_cache()
                        rag_system.load_cv()
                        st.rerun()
                
                with col2:
                    if st.button(system_text["cache_clear_btn"], help=system_text["cache_clear_help"],use_container_width=True):
                        rag_system.clear_cache()
                        st.success(system_text["cache_cleared_success"])
                        st.rerun()
            else:
                st.markdown(f"- **Status**: {system_text['cache_no_cache']}")
        
        # Chunk viewer
        if st.button(system_text["view_chunks"],use_container_width=True):
            st.session_state.show_chunks = not st.session_state.get("show_chunks", False)
        
        if st.session_state.get("show_chunks", False):
            st.markdown(f"### {system_text['chunks_title']}")
            if rag_system.configured and rag_system.cv_chunks:
                for i, chunk in enumerate(rag_system.cv_chunks):
                    with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                        st.text(chunk)
            else:
                st.warning(system_text["chunks_not_available"])
        
        # System status
        if rag_system.configured and rag_system.cv_chunks:
            st.markdown(f"- **{system_text['chunks_loaded']}**: {len(rag_system.cv_chunks)}")
            st.markdown(f"- **Embeddings**: {'✅' if rag_system.cv_embeddings is not None else '❌'}")
            st.markdown(f"- **Job Analyzer**: {'✅' if rag_system.tool_definitions.job_compatibility_analyzer else '❌'}")



def render_pdf_download() -> None:
    """Optimized PDF download rendering"""
    # Early return if no PDF data
    if not all(key in st.session_state for key in ["pdf_data", "pdf_filename"]):
        return

    lang = get_current_language()
    system_text = get_cached_system_text(lang.value)
    
    pdf_bytes = st.session_state.pdf_data
    file_name = st.session_state.pdf_filename

    # Download button
    st.download_button(
        system_text["pdf_download"], 
        pdf_bytes, 
        file_name,
        mime="application/pdf", 
        use_container_width=True
    )

    # Action buttons
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button(system_text["pdf_email"], use_container_width=True):
            st.session_state.show_email_form = True
    with col2:
        if st.button(system_text["pdf_clear"], use_container_width=True):
            # Efficient cleanup
            for key in ("pdf_data", "pdf_filename", "show_email_form"):
                st.session_state.pop(key, None)
            st.rerun()

    # Email form
    if st.session_state.get("show_email_form"):
        render_email_form_for_pdf(pdf_bytes, file_name, lang)


def render_email_form_for_pdf(pdf_bytes: bytes, filename: str, language: Language):
    """Clean email form without JavaScript"""
    
    system_text = get_cached_system_text(language.value)
    
    # Email form
    st.markdown("---")
    st.markdown(f"### {system_text['email_form_title']}")
    
    with st.form("pdf_email_form", clear_on_submit=True):
        user_email = st.text_input(
            system_text['email_label'],
            placeholder=system_text['email_placeholder'],
            help="We'll send the PDF report to this email address" if language == Language.ENGLISH else "PDF raporunu bu e-posta adresine göndereceğiz"
        )
        
        # Form submission buttons
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button(
                system_text['email_send'], 
                use_container_width=True, 
                type="primary"
            )
        
        with col2:
            cancelled = st.form_submit_button(
                system_text['email_cancel'], 
                use_container_width=True
            )
        
        # Handle form submission
        if submitted:
            if user_email and "@" in user_email and "." in user_email:
                with st.spinner(system_text['email_sending']):
                    success = send_pdf_via_email(pdf_bytes, filename, user_email, language)
                
                if success:
                    st.success(system_text['email_success'])
                    st.session_state.show_email_form = False
                    st.rerun()
                else:
                    st.error(system_text['email_error'])
            else:
                st.error(system_text['email_invalid'])
        
        if cancelled:
            st.session_state.show_email_form = False
            st.rerun()
def render_header_popovers():
    """Render functionality and bug report popovers in header"""
    # Önce mevcut tüm mesajlardan dil algıla
    current_messages = st.session_state.get("messages", [])
    detected_language = LanguageDetector.detect_from_messages(current_messages)
    
    # Session state'deki dili güncelle
    st.session_state.current_language = detected_language
    language = detected_language
    
    # Create two columns for popovers
    col1, col2 = st.columns(2, gap=None)
    
    with col1:
        # Functionalities popover
        func_label = "🚀 Özellikler" if language == Language.TURKISH else "🚀 Features"
        with st.popover(func_label, use_container_width=True):
            if language == Language.TURKISH:
                st.markdown("""
                ### 📋 Mevcut Özellikler
                
                **📧 İletişime Geç** - Selman'a göndermek üzere sizin için bir e-posta hazırlayabilirim.
                
                **💼 İş Uyumu Raporu** - Size Selman'ın güncel özgeçmişine göre hazırlanmış bir işe uyumluluk raporu hazırlayabilirim.\\
                Rapor için iş tanımını yapıştırmanız yeterli. PDF de isteyerek raporu indirebilirsiniz.
                
                **📱 Güncel Gönderiler** - Selman'ın son paylaşımlarını ve makalelerini gösterebilirim.
                
                **🎯 Daha fazla gelişmiş özellik yakında!**
                """)
            else:
                st.markdown("""
                ### 📋 Available Features
                
                **📧 Contact Selman** - I can prepare emails for you to send Selman.
                
                **💼 Job Compatibility Analysis** - I can make a job compatibility analysis based on Selman's recent CV.\\
                Paste job description for compatibility report. Ask for a PDF to download the report
                
                **📱 Social Updates** - I can get you Selman's recent posts and articles.
                
                **🎯 More advanced features coming soon!**
                """)
    
    with col2:
        # Bug report popover
        bug_label = "🐛 Hata Bildir" if language == Language.TURKISH else "🐛 Bug Report"
        with st.popover(bug_label, use_container_width=True):
            # Bug report manager'ı başlat
            if 'bug_manager' not in st.session_state:
                st.session_state.bug_manager = BugReportManager()
            
            bug_manager = st.session_state.bug_manager
            
            # Bug report success state kontrolü
            bug_success_key = f"bug_success_{language.value}"
            if st.session_state.get(bug_success_key, False):
                if language == Language.TURKISH:
                    st.success("✅ Hata raporunuz başarıyla gönderildi! Teşekkür ederiz.")
                    if st.button("Yeni Hata Bildir", key="new_bug_tr", use_container_width=True):
                        st.session_state[bug_success_key] = False
                        st.rerun()
                else:
                    st.success("✅ Bug report submitted successfully! Thank you.")
                    if st.button("Report New Bug", key="new_bug_en", use_container_width=True):
                        st.session_state[bug_success_key] = False
                        st.rerun()
                return
            
            if language == Language.TURKISH:
                st.markdown("#### 🐛 Hata Bildirimi")
                
                # Bilgilendirme mesajı
                st.info("ℹ️ **Gizlilik Notu:** Sorunu anlayabilmemiz için son 10 mesaj gönderilecektir.")
                
                # Form kullanarak widget state'ini yönet
                with st.form(key="bug_form_tr", clear_on_submit=True):
                    bug_description = st.text_area(
                        "Karşılaştığınız hatayı detaylı olarak açıklayın:",
                        placeholder="Lütfen hatayı mümkün olduğunca detaylı şekilde açıklayın...",
                        height=100,
                        help="Hatanın ne zaman ve nasıl oluştuğunu detaylı açıklayın"
                    )
                    
                    # Submit button
                    submitted = st.form_submit_button(
                        "Hata Raporunu Gönder", 
                        use_container_width=True,
                        type="primary"
                    )
                    
                    if submitted:
                        if bug_description.strip():
                            with st.spinner("Hata raporu gönderiliyor..."):
                                result = bug_manager.submit_bug_report(bug_description.strip(), language.value)
                            
                            if result["success"]:
                                st.session_state[bug_success_key] = True
                                st.rerun()
                            else:
                                st.error(f"❌ Hata raporu gönderilemedi: {result['message']}")
                        else:
                            st.warning("⚠️ Lütfen hata açıklaması girin.")
                        
            else:
                st.markdown("###### 🐛 Bug Report")
                
                # Bilgilendirme mesajı
                st.info("ℹ️ **Privacy Note:** This bug report includes your last 10 messages to help us find out the issue better.")
                
                # Form kullanarak widget state'ini yönet
                with st.form(key="bug_form_en", clear_on_submit=True):
                    bug_description = st.text_area(
                        "Tell us the bug you faced in detail:",
                        placeholder="Please tell us the bug as detailed as possible...",
                        height=100,
                        help="Please describe when and how the bug occurred in detail"
                    )
                    
                    # Submit button
                    submitted = st.form_submit_button(
                        "Submit Bug Report", 
                        use_container_width=True,
                        type="primary"
                    )
                    
                    if submitted:
                        if bug_description.strip():
                            with st.spinner("Submitting bug report..."):
                                result = bug_manager.submit_bug_report(bug_description.strip(), language.value)
                            
                            if result["success"]:
                                st.session_state[bug_success_key] = True
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to submit bug report: {result['message']}")
                        else:
                            st.warning("⚠️ Please enter a bug description.")

@st.cache_data
def get_welcome_message() -> str:
    """Cached welcome message"""
    return """
    #### 👋 Merhaba! | Hello there!
    
    **Ben Selman'ın AI Portföy Asistanıyım. Nasıl yardımcı olabilirim?**
    
    **I'm Selman's AI Portfolio Assistant. How can I help you?**
    """

def render_welcome_message():
    """Render optimized welcome message"""
    # Update language efficiently
    current_messages = st.session_state.get("messages", [])
    if current_messages:  # Only detect if there are messages
        detected_language = LanguageDetector.detect_from_messages(current_messages)
        update_language_if_needed(detected_language)
    
    st.markdown(get_welcome_message())
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
            system_text = get_cached_system_text(language.value)
            st.error(system_text["email_config_missing"])
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
        st.session_state.messages = []


def optimize_memory():
    """Optimize memory usage for embedded environment"""
    import gc
    
    # Clear any unused session state
    keys_to_clean = [key for key in st.session_state.keys() 
                     if key.startswith('temp_') or key.startswith('cache_')]
    for key in keys_to_clean:
        if key in st.session_state:
            del st.session_state[key]
    
    # Force garbage collection
    gc.collect()



# Add this call at the beginning of main() function:
def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Selman DEDEAKAYOĞULLARI Portfolio RAG Chatbot",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Memory optimization for embedded environment
    optimize_memory()

    
    # Initialize session state
    initialize_session_state()
    
    # Dil değişikliği kontrolü
    if 'language_changed_flag' in st.session_state:
        del st.session_state.language_changed_flag
        st.rerun()
    
    # Header popovers
    render_header_popovers()
    
    # Welcome message
    render_welcome_message()
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        # Detect language for initialization messages
        language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
        system_text = get_cached_system_text(language.value)
        
        with st.spinner(system_text["initializing_chatbot"]):
            st.session_state.rag_system = GeminiEmbeddingRAG()
    
    rag_system = st.session_state.rag_system
    
    # Check configuration
    if not rag_system.configured:
        language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
        system_text = get_cached_system_text(language.value)
        st.error(system_text["configure_api_key"])
        st.stop()
    
    # Check email configuration
    if not rag_system.email_tool.email_user or not rag_system.email_tool.email_password:
        language = LanguageDetector.detect_from_messages(st.session_state.get("messages", []))
        system_text = get_cached_system_text(language.value)
        st.warning(system_text["email_not_configured"])
    
    # Initialize chat interface
    chat_interface = ChatInterface(rag_system)
    
    # Handle email actions
    chat_interface.handle_email_actions()
    
    # Display messages
    chat_interface.display_messages()
    
    # Chat input
    if prompt := st.chat_input("💬"):
        chat_interface.process_user_input(prompt)
    
    # PDF download button
    render_pdf_download()
    
    # Sidebar
    render_sidebar(rag_system)


if __name__ == "__main__":
    main()