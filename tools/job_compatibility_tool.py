import re
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import streamlit as st
from google import genai
from google.genai import types
import numpy as np


class AnalysisConstants:
    """Constants for job compatibility analysis"""
    DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-06-17"
    DEFAULT_TEMPERATURE = 0.1
    ANALYSIS_TEMPERATURE = 0.2
    REPORT_TEMPERATURE = 0.3
    MAX_OUTPUT_TOKENS = 12000
    STOP_SEQUENCES = []
    MAX_EDUCATION_CHUNKS = 4
    EDUCATION_SEARCH_BOOST = 0.3
    # Search limits
    MAX_SKILLS_FOR_SEARCH = 5
    MAX_RESPONSIBILITIES_FOR_SEARCH = 3
    MAX_CHUNKS_PER_SEARCH = 4
    MAX_TOTAL_CHUNKS = 10
    GENERAL_SEARCH_CHUNKS = 3
    
    # Report generation
    MAX_RETRIES = 3
    MIN_REPORT_LENGTH = 500


@dataclass
class JobRequirements:
    """Structured job requirements data"""
    position_title: str = ""
    required_skills: List[str] = None
    preferred_skills: List[str] = None
    experience_years: str = ""
    education_requirements: str = ""
    key_responsibilities: List[str] = None
    company_info: str = ""
    location: str = ""
    industry: str = ""
    soft_skills: List[str] = None
    
    def __post_init__(self):
        """Initialize list fields if None"""
        self.required_skills = self.required_skills or []
        self.preferred_skills = self.preferred_skills or []
        self.key_responsibilities = self.key_responsibilities or []
        self.soft_skills = self.soft_skills or []


class JobCompatibilityAnalyzer:
    """
    Analyze job compatibility between CV and job description using RAG chunks.
    
    This analyzer uses LLM to extract job requirements, searches relevant CV chunks
    using RAG, and generates comprehensive compatibility reports.
    """
    
    def __init__(self, client: genai.Client, cv_data: Dict[str, Any], rag_system: Optional[Any] = None):
        self.client = client
        self.cv_data = cv_data or {}
        self.rag_system = rag_system
        
        # Enhanced education keywords for better detection
        self.education_keywords = {
            'en': [
                'university', 'college', 'degree', 'bachelor', 'master', 'phd', 'doctorate',
                'education', 'academic', 'diploma', 'certification', 'graduate', 'undergraduate',
                'school', 'institute', 'faculty', 'major', 'minor', 'gpa', 'thesis', 'research',
                'erasmus', 'exchange', 'study', 'course', 'program', 'qualification'
            ],
            'tr': [
                'üniversite', 'üniversitesi', 'lisans', 'yüksek lisans', 'doktora', 'eğitim',
                'okul', 'akademi', 'fakülte', 'bölüm', 'diploma', 'sertifika', 'mezun',
                'öğrenci', 'ders', 'program', 'kurs', 'araştırma', 'tez', 'not ortalaması',
                'erasmus', 'değişim', 'öğrenim', 'tahsil', 'yeterlilik'
            ]
        }
        
    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean LLM response to extract valid JSON.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        cleaned = response_text.strip()
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = re.sub(r'^\s*```\s*', '', cleaned)
        
        # Remove any trailing/leading whitespace
        return cleaned.strip()
    
    def _safe_json_parse(self, json_str: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Safely parse JSON with fallback to default.
        
        Args:
            json_str: JSON string to parse
            default: Default value if parsing fails
            
        Returns:
            Parsed dictionary or default value
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            st.warning(f"JSON parsing error: {e}")
            return default or {}
    
    def extract_job_requirements(self, job_description: str) -> JobRequirements:
        """
        Extract key requirements from job description using LLM.
        
        Args:
            job_description: Raw job description text
            
        Returns:
            JobRequirements object with extracted data
        """
        if not job_description or not job_description.strip():
            st.error("Job description is empty")
            return JobRequirements()
        
        prompt = f"""Analyze this job description and extract key information in JSON format:

Job Description:
{job_description}

Please extract and return a JSON with these fields:
- position_title: Job title (string)
- required_skills: List of technical skills mentioned (be comprehensive)
- preferred_skills: List of nice-to-have skills
- experience_years: Required years of experience (number or "entry-level")
- education_requirements: Education requirements (string)
- key_responsibilities: Main job responsibilities (list of strings)
- company_info: Any company information mentioned (string)
- location: Job location if mentioned (string)
- industry: Industry/domain if identifiable (string)
- soft_skills: Any soft skills mentioned (list of strings)

Important: Return ONLY valid JSON without any markdown formatting or additional text."""

        try:
            response = self.client.models.generate_content(
                model=AnalysisConstants.DEFAULT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=AnalysisConstants.DEFAULT_TEMPERATURE,
                    max_output_tokens=2000
                )
            )
            
            # Clean and parse response
            cleaned_response = self._clean_json_response(response.text)
            requirements_dict = self._safe_json_parse(cleaned_response)
            
            # Convert to JobRequirements object
            return JobRequirements(**requirements_dict)
            
        except Exception as e:
            st.error(f"Error extracting job requirements: {e}")
            return JobRequirements()
    
    def _build_search_queries(self, job_requirements: JobRequirements) -> List[str]:
        """
        Build comprehensive search queries from job requirements with enhanced education focus.
        """
        queries = []
        
        # Position title query
        if job_requirements.position_title:
            queries.append(job_requirements.position_title)
        
        # Skills query (limited to top skills)
        if job_requirements.required_skills:
            skills_query = ' '.join(
                job_requirements.required_skills[:AnalysisConstants.MAX_SKILLS_FOR_SEARCH]
            )
            if skills_query.strip():
                queries.append(skills_query)
        
        # Enhanced education queries - KRITIK DÜZELTME
        if job_requirements.education_requirements:
            # Direct education requirement search
            queries.append(job_requirements.education_requirements)
            
            # Extract specific education terms
            education_terms = []
            education_text = job_requirements.education_requirements.lower()
            
            # Look for degree types
            degree_patterns = [
                r'\b(bachelor|lisans|bs|ba|b\.s\.|b\.a\.)\b',
                r'\b(master|yüksek lisans|ms|ma|m\.s\.|m\.a\.)\b',
                r'\b(phd|doktora|doctorate|ph\.d\.)\b',
                r'\b(university|üniversite|college|akademi)\b',
                r'\b(engineering|mühendislik|computer science|bilgisayar)\b',
                r'\b(degree|derece|diploma|sertifika)\b'
            ]
            
            for pattern in degree_patterns:
                matches = re.findall(pattern, education_text)
                education_terms.extend(matches)
            
            # Add education-specific query
            if education_terms:
                queries.append(' '.join(set(education_terms)))
        
        # Generic education search - YENI EKLEME
        education_search_queries = [
            'education university degree',
            'eğitim üniversite lisans',
            'academic background qualification',
            'akademik geçmiş eğitim'
        ]
        queries.extend(education_search_queries)
        
        # Responsibilities query
        if job_requirements.key_responsibilities:
            resp_query = ' '.join(
                job_requirements.key_responsibilities[:AnalysisConstants.MAX_RESPONSIBILITIES_FOR_SEARCH]
            )
            if resp_query.strip():
                queries.append(resp_query)
        
        # Industry context
        if job_requirements.industry:
            queries.append(job_requirements.industry)
        
        # Filter out empty queries
        return [q for q in queries if q and q.strip()]
    
    def _collect_unique_chunks(self, queries: List[str]) -> List[str]:
        """
        Collect unique chunks from multiple searches.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of unique chunk texts
        """
        seen_chunks: Set[str] = set()
        unique_chunks: List[str] = []
        
        for query in queries:
            try:
                chunks = self.rag_system.search_similar_chunks(
                    query, 
                    top_k=AnalysisConstants.MAX_CHUNKS_PER_SEARCH
                )
                
                for chunk in chunks:
                    chunk_text = chunk.get('text', '').strip()
                    if chunk_text and chunk_text not in seen_chunks:
                        unique_chunks.append(chunk_text)
                        seen_chunks.add(chunk_text)
                        
                        # Stop if we have enough chunks
                        if len(unique_chunks) >= AnalysisConstants.MAX_TOTAL_CHUNKS:
                            return unique_chunks
                            
            except Exception as e:
                st.warning(f"Error searching for query '{query}': {e}")
                continue
        
        return unique_chunks
    
    def _is_education_chunk(self, chunk_text: str) -> bool:
        """
        YENI FONKSIYON: Chunk'ın eğitim içeriği olup olmadığını kontrol et.
        """
        chunk_lower = chunk_text.lower()
        
        # Güçlü eğitim indikatörleri
        strong_indicators = [
            'education', 'eğitim', 'university', 'üniversite', 'college', 'degree', 'derece',
            'bachelor', 'lisans', 'master', 'yüksek lisans', 'phd', 'doktora', 'graduation',
            'mezuniyet', 'graduate', 'mezun', 'diploma', 'academic', 'akademik', 'gpa',
            'thesis', 'tez', 'course', 'ders', 'program', 'faculty', 'fakülte', 'erasmus'
        ]
        
        # En az 2 güçlü indikatör arayın
        indicator_count = sum(1 for indicator in strong_indicators if indicator in chunk_lower)
        
        # Veya spesifik eğitim formatları
        education_patterns = [
            r'\b\d{4}\s*-\s*\d{4}\b',  # Yıl aralığı (2018-2022)
            r'\b(bs|ba|ms|ma|phd|b\.s\.|b\.a\.|m\.s\.|m\.a\.)\b',  # Derece kısaltmaları
            r'\b(gpa|not ortalaması|cgpa)\b',  # GPA belirteçleri
            r'\b(semester|dönem|year|yıl|term)\b'  # Akademik dönem belirteçleri
        ]
        
        pattern_matches = sum(1 for pattern in education_patterns 
                            if re.search(pattern, chunk_lower))
        
        return indicator_count >= 2 or pattern_matches >= 1

    def _get_education_specific_chunks(self, job_requirements: JobRequirements) -> List[str]:
        """
        YENI FONKSIYON: Eğitim-spesifik chunk'ları hedefli şekilde arama.
        """
        education_chunks = []
        
        if not self.rag_system:
            return education_chunks
        
        # Eğitim-spesifik arama terimleri
        education_search_terms = [
            "education university degree",
            "eğitim üniversite lisans", 
            "academic background",
            "akademik geçmiş",
            "diploma certificate",
            "diploma sertifika",
            "graduation graduate",
            "mezuniyet mezun",
            "bachelor master",
            "lisans yüksek lisans"
        ]
        
        # Job requirements'tan eğitim terimleri çıkar
        if job_requirements.education_requirements:
            education_search_terms.append(job_requirements.education_requirements)
        
        # Her arama terimi için chunk'ları al
        seen_chunks = set()
        for term in education_search_terms:
            try:
                chunks = self.rag_system.search_similar_chunks(term, top_k=3)
                for chunk in chunks:
                    chunk_text = chunk.get('text', '').strip()
                    if chunk_text and chunk_text not in seen_chunks:
                        # Eğitim içeriği kontrolü
                        if self._is_education_chunk(chunk_text):
                            education_chunks.append(chunk_text)
                            seen_chunks.add(chunk_text)
                            
                        if len(education_chunks) >= 4:  # Maksimum 4 eğitim chunk'ı
                            break
                            
                if len(education_chunks) >= 4:
                    break
                    
            except Exception as e:
                st.warning(f"Error searching education chunks for '{term}': {e}")
                continue
        
        return education_chunks

    def get_relevant_cv_context(self, job_requirements: JobRequirements) -> str:
        """
        Enhanced CV context retrieval with specific education chunk prioritization.
        """
        if not self.rag_system or not hasattr(self.rag_system, 'search_similar_chunks'):
            return self._format_cv_data_as_text()
        
        try:
            # Build search queries
            search_queries = self._build_search_queries(job_requirements)
            
            # Collect unique chunks from searches
            relevant_chunks = self._collect_unique_chunks(search_queries)
            
            # KRITIK EKLEME: Eğitim-spesifik chunk arama
            education_chunks = self._get_education_specific_chunks(job_requirements)
            
            # Merge education chunks with existing ones
            seen_texts = {chunk.strip() for chunk in relevant_chunks}
            for edu_chunk in education_chunks:
                if edu_chunk.strip() not in seen_texts:
                    relevant_chunks.append(edu_chunk)
                    seen_texts.add(edu_chunk.strip())
                    if len(relevant_chunks) >= AnalysisConstants.MAX_TOTAL_CHUNKS:
                        break
            
            # Add general experience chunks
            try:
                exp_chunks = self.rag_system.search_similar_chunks(
                    "work experience projects achievements skills", 
                    top_k=AnalysisConstants.GENERAL_SEARCH_CHUNKS
                )
                
                for chunk in exp_chunks:
                    chunk_text = chunk.get('text', '').strip()
                    if chunk_text and chunk_text not in seen_texts:
                        relevant_chunks.append(chunk_text)
                        seen_texts.add(chunk_text)
                        if len(relevant_chunks) >= AnalysisConstants.MAX_TOTAL_CHUNKS:
                            break
                            
            except Exception as e:
                st.warning(f"Error in general experience search: {e}")
            
            # Format chunks with separators
            if relevant_chunks:
                return '\n\n---\n\n'.join(relevant_chunks[:AnalysisConstants.MAX_TOTAL_CHUNKS])
            else:
                return self._format_cv_data_as_text()
                
        except Exception as e:
            st.error(f"Error getting CV context: {e}")
            return self._format_cv_data_as_text()
    
    def _format_section(self, title: str, content: Any) -> List[str]:
        """
        Format a CV section for text output.
        
        Args:
            title: Section title
            content: Section content (various types)
            
        Returns:
            List of formatted lines
        """
        lines = [f"\n{title}:"]
        
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"  {key}: {value}")
                    
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Format nested dictionaries (e.g., experience entries)
                    for key, value in item.items():
                        if key == "description" and value:
                            lines.append(f"    {key}: {value}")
                        elif value:
                            lines.append(f"  {key}: {value}")
                else:
                    lines.append(f"  - {item}")
                    
        else:
            lines.append(f"  {content}")
            
        return lines
    
    def _format_cv_data_as_text(self) -> str:
        """
        Fallback method to format CV data as structured text.
        
        Returns:
            Formatted CV text
        """
        try:
            text_parts = []
            
            # Basic information
            if name := self.cv_data.get('name'):
                text_parts.append(f"Name: {name}")
            if title := self.cv_data.get('title'):
                text_parts.append(f"Title: {title}")
            if profile := self.cv_data.get('profile'):
                text_parts.append(f"Profile: {profile}")
            
            # Skills section
            if skills := self.cv_data.get('skills'):
                text_parts.extend(self._format_section("Skills", skills))
            
            # Experience section
            if experience := self.cv_data.get('experience'):
                text_parts.extend(self._format_section("Experience", experience))
            
            # Projects section
            if projects := self.cv_data.get('projects'):
                text_parts.extend(self._format_section("Projects", projects))
            
            # Education section
            if education := self.cv_data.get('education'):
                text_parts.extend(self._format_section("Education", education))
            
            # Certifications section
            if certifications := self.cv_data.get('certifications'):
                text_parts.extend(self._format_section("Certifications", certifications))
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            return f"Error formatting CV data: {e}"
    
    def analyze_compatibility_with_llm(
        self, 
        job_requirements: JobRequirements, 
        cv_context: str
    ) -> Dict[str, Any]:
        """
        Enhanced LLM analysis with better education focus.
        """
        # Convert JobRequirements to dict for JSON serialization
        requirements_dict = {
            k: v for k, v in job_requirements.__dict__.items() 
            if v  # Only include non-empty values
        }
        
        # Enhanced analysis prompt with education emphasis
        analysis_prompt = f"""You are an expert HR analyst with deep experience in talent matching. 
Analyze the compatibility between this job requirement and candidate profile with extreme attention to detail.

CRITICAL: Pay special attention to EDUCATION analysis. The candidate's education information is included in the CV context below.

JOB REQUIREMENTS:
{json.dumps(requirements_dict, indent=2)}

CANDIDATE PROFILE (from CV):
{cv_context}

Perform a comprehensive analysis and return a JSON response with this EXACT structure:
{{
    "overall_compatibility_score": <number 0-100>,
    "skill_analysis": {{
        "required_skills_match": <percentage 0-100>,
        "matched_required_skills": [<list of exactly matched required skills>],
        "missing_required_skills": [<list of required skills the candidate lacks>],
        "preferred_skills_match": <percentage 0-100>,
        "matched_preferred_skills": [<list of matched preferred skills>],
        "additional_relevant_skills": [<candidate skills relevant to the role but not explicitly mentioned>]
    }},
    "experience_analysis": {{
        "meets_experience_requirement": <true/false>,
        "relevant_experience_years": <estimated years as number>,
        "relevant_experiences": [<list of specific relevant work experiences>],
        "experience_quality_score": <0-100 based on relevance and impact>
    }},
    "education_analysis": {{
        "meets_education_requirement": <true/false>,
        "education_relevance_score": <0-100>,
        "relevant_education": [<list of relevant degrees/certifications with details>],
        "education_details": [<extract specific education details like degree type, institution, year>],
        "education_level_match": <how well education level matches requirement>
    }},
    "project_analysis": {{
        "relevant_projects": [<list of projects relevant to this role>],
        "project_relevance_score": <0-100>
    }},
    "strengths": [<top 5 candidate strengths for this specific role>],
    "weaknesses": [<top 3-5 areas where candidate needs development>],
    "recommendations": [<3-5 actionable recommendations for candidate and/or employer>]
}}

EDUCATION ANALYSIS GUIDELINES:
- Look for degree types (Bachelor, Master, PhD, etc.)
- Check for relevant fields of study
- Consider institution reputation if mentioned
- Evaluate education timeline and recency
- Look for continuous learning indicators
- Consider certifications and additional qualifications

Analysis Guidelines:
- Consider transferable skills and related technologies
- Look for patterns in experience that indicate capability
- Evaluate project complexity and relevance
- Consider industry experience and domain knowledge
- Be fair but thorough in identifying gaps
- PRIORITIZE education matching if education requirements are specified

Return ONLY valid JSON without any markdown formatting or additional text."""

        try:
            response = self.client.models.generate_content(
                model=AnalysisConstants.DEFAULT_MODEL,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    temperature=AnalysisConstants.ANALYSIS_TEMPERATURE,
                    max_output_tokens=AnalysisConstants.MAX_OUTPUT_TOKENS,
                    stop_sequences=AnalysisConstants.STOP_SEQUENCES
                )
            )
            
            # Clean and parse response
            cleaned_response = self._clean_json_response(response.text)
            analysis_result = self._safe_json_parse(cleaned_response)
            
            # Validate required fields
            required_fields = ["overall_compatibility_score", "skill_analysis", "experience_analysis", "education_analysis"]
            if all(field in analysis_result for field in required_fields):
                return analysis_result
            else:
                st.warning("Incomplete analysis response, using fallback")
                return self._create_fallback_analysis()
            
        except Exception as e:
            st.error(f"Error in LLM compatibility analysis: {e}")
            return self._create_fallback_analysis(error=str(e))
    
    def _create_fallback_analysis(self, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a fallback analysis structure when LLM fails.
        
        Args:
            error: Optional error message
            
        Returns:
            Basic analysis structure
        """
        base_analysis = {
            "overall_compatibility_score": 0,
            "skill_analysis": {
                "required_skills_match": 0,
                "matched_required_skills": [],
                "missing_required_skills": [],
                "preferred_skills_match": 0,
                "matched_preferred_skills": [],
                "additional_relevant_skills": []
            },
            "experience_analysis": {
                "meets_experience_requirement": False,
                "relevant_experience_years": 0,
                "relevant_experiences": [],
                "experience_quality_score": 0
            },
            "education_analysis": {
                "meets_education_requirement": False,
                "education_relevance_score": 0,
                "relevant_education": []
            },
            "project_analysis": {
                "relevant_projects": [],
                "project_relevance_score": 0
            },
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        if error:
            base_analysis["error"] = error
            
        return base_analysis
    
    def _validate_report_completeness(self, report_text: str, language: str) -> bool:
        """
        Validate if the generated report is complete based on language.
        
        Args:
            report_text: Generated report text
            language: Report language
            
        Returns:
            True if report appears complete
        """
        if not report_text or len(report_text.strip()) < AnalysisConstants.MIN_REPORT_LENGTH:
            return False
        
        # Check for required sections based on language
        if language == "tr":
            required_sections = [
                "özet", "yönetici özeti", "teknik beceriler", "beceriler", 
                "deneyim", "eğitim", "öneri", "tavsiye", "güçlü", "proje"
            ]
        else:
            required_sections = [
                "executive summary", "technical skills", "experience", 
                "education", "recommendation", "strengths", "project"
            ]
        
        # Check if at least 4 out of core sections are present
        sections_found = sum(1 for section in required_sections 
                            if section.lower() in report_text.lower())
        
        return sections_found >= 4

    def _generate_report_with_retry(
        self, 
        job_requirements: JobRequirements,
        compatibility_analysis: Dict[str, Any],
        language: str,
        max_retries: int = AnalysisConstants.MAX_RETRIES
    ) -> str:
        """
        Generate report with retry mechanism for incomplete responses.
        
        Args:
            job_requirements: Job requirements
            compatibility_analysis: Analysis results
            language: Report language
            max_retries: Maximum retry attempts
            
        Returns:
            Complete report text
        """
        # Language-specific messages
        messages = {
            "tr": {
                "retry_warning": "Deneme {}: Rapor eksik görünüyor, tekrar deneniyor...",
                "attempt_failed": "Deneme {} başarısız: {}",
                "generating_fallback": "Temel rapor oluşturuluyor..."
            },
            "en": {
                "retry_warning": "Attempt {}: Report appears incomplete, retrying...",
                "attempt_failed": "Attempt {} failed: {}",
                "generating_fallback": "Generating fallback report..."
            }
        }
        
        lang_msgs = messages.get(language, messages["en"])
        
        for attempt in range(max_retries):
            try:
                # Generate prompt
                report_prompt = self._generate_report_prompt(
                    job_requirements,
                    compatibility_analysis,
                    language
                )
                
                # Add completion instruction for retries
                if attempt > 0:
                    completion_instruction = {
                        "tr": (
                            "\n\nÖNEMLİ: EKSIKSIZ bir rapor oluşturun. "
                            "Yukarıda belirtilen tüm bölümleri kapsayana kadar durmayın. "
                            "Son Öneri bölümünü mutlaka sonuna ekleyin. TÜRKÇE yazın."
                        ),
                        "en": (
                            "\n\nIMPORTANT: Generate a COMPLETE report. "
                            "Do not stop until you have covered all sections mentioned above. "
                            "Make sure to include the Final Recommendation section at the end. Write in ENGLISH."
                        )
                    }
                    report_prompt += completion_instruction.get(language, completion_instruction["en"])
                
                # Generate response
                response = self.client.models.generate_content(
                    model=AnalysisConstants.DEFAULT_MODEL,
                    contents=report_prompt,
                    config=types.GenerateContentConfig(
                        temperature=AnalysisConstants.REPORT_TEMPERATURE + (attempt * 0.1),
                        max_output_tokens=AnalysisConstants.MAX_OUTPUT_TOKENS,
                    )
                )
                
                if response.text:
                    # Validate completeness
                    if self._validate_report_completeness(response.text, language):
                        return response.text
                    else:
                        st.warning(lang_msgs["retry_warning"].format(attempt + 1))
                        continue
                
            except Exception as e:
                st.warning(lang_msgs["attempt_failed"].format(attempt + 1, str(e)))
                if attempt == max_retries - 1:
                    raise e
                continue
        
        # If all retries fail, return a fallback report
        st.info(lang_msgs["generating_fallback"])
        return self._generate_fallback_report(job_requirements, compatibility_analysis, language)

    def _generate_report_prompt(
        self, 
        job_requirements: JobRequirements,
        compatibility_analysis: Dict[str, Any],
        language: str
    ) -> str:
        """
        Generate the prompt for final report generation with explicit language control.
        
        Args:
            job_requirements: Job requirements
            compatibility_analysis: Analysis results
            language: Report language
            
        Returns:
            Formatted prompt string
        """
        candidate_name = self.cv_data.get('name', 'Unknown Candidate')
        position_title = job_requirements.position_title or 'Unknown Position'
        
        # Language-specific prompts
        if language == "tr":
            return f"""Bu analiz sonuçlarına göre kapsamlı, profesyonel bir iş uyumluluk raporu oluştur:

POZİSYON: {position_title}
ADAY: {candidate_name}

UYUMLULUK ANALİZİ:
{json.dumps(compatibility_analysis, indent=2)}

TÜRKÇE olarak aşağıdaki bölümleri içeren detaylı bir rapor oluştur:

1. **Yönetici Özeti** 
- Genel uyumluluk skoru ve görsel gösterge (⭐ derecelendirmeleri)
- Bir paragraf özet öneri
- Ana noktalar (3-4 madde)

2. **Teknik Beceriler Değerlendirmesi**
- Gerekli becerilerin karşılanma yüzdesi
- Eşleşen beceriler ve yeterlilik göstergeleri
- Eksik beceriler ve öğrenme önerileri
- Değer katacak ek ilgili beceriler

3. **Profesyonel Deneyim Değerlendirmesi**
- İlgili deneyim yılları vs gereksinim
- En uygun roller ve başarılar
- Sektör/alan deneyimi uyumu
- Liderlik ve proje karmaşıklığı göstergeleri

4. **Eğitim ve Sertifikasyonlar**
- Formal eğitim uyumu
- İlgili sertifikalar ve eğitimler
- Sürekli öğrenme göstergeleri

5. **Proje Portföyü Analizi**
- En uygun projeler ve etkileri
- Gösterilen teknik karmaşıklık
- Gösterilen problem çözme yetenekleri

6. **Bu Rol İçin Ana Güçlü Yönler**
- Spesifik örneklerle en iyi 5 güçlü yön
- Benzersiz değer önerileri
- Kültürel/yumuşak beceri uyumları

7. **Gelişim Fırsatları**
- Öncelik düzeyleri ile beceri eksiklikleri
- Önerilen öğrenme yolları
- Beceri kazanımı için zaman çizelgesi

8. **Son Öneri**
- Net işe alım önerisi (Güçlü Şekilde Önerilir/Önerilir/Koşullu/Önerilmez)
- Risk değerlendirmesi
- İşe alınırsa uyum önerileri
- Uygunsa alternatif rol önerileri

Biçimlendirme Kuralları:
- Net başlıklar ve alt başlıklar kullan
- İlgili yerlerde yüzde skorları ve ölçütler ekle
- Netlik için madde işaretleri kullan
- Görsel çekicilik için emoji göstergeleri ekle (✅ ❌ ⚠️ 🌟 📊)
- Profesyonel ama ilgi çekici ton kullan
- Kalın metin kullanarak taranabilir yap
- Toplam uzunluk: 800-1200 kelime

Hem işe alan hem de aday için eyleme dönük içgörülere odaklan.

KRİTİK: 8 bölümü de kapsayan EKSİKSİZ raporu TÜRKÇE olarak yaz. Son Öneri bölümüne ulaşana kadar durma."""
        
        else:  # English
            return f"""Generate a comprehensive, professional job compatibility report based on this analysis:

JOB POSITION: {position_title}
CANDIDATE: {candidate_name}

COMPATIBILITY ANALYSIS:
{json.dumps(compatibility_analysis, indent=2)}

Generate a detailed report in ENGLISH with these sections:

1. **Executive Summary** 
- Overall compatibility score with visual indicator (e.g., ⭐ ratings)
- One-paragraph recommendation summary
- Key highlights (3-4 bullet points)

2. **Technical Skills Assessment**
- Required skills coverage with percentage
- Matched skills with proficiency indicators
- Missing skills with learning recommendations
- Additional relevant skills that add value

3. **Professional Experience Evaluation**
- Years of relevant experience vs. requirement
- Most relevant roles and achievements
- Industry/domain experience alignment
- Leadership and project complexity indicators

4. **Education & Certifications**
- Formal education alignment
- Relevant certifications and training
- Continuous learning indicators

5. **Project Portfolio Analysis**
- Most relevant projects with impact
- Technical complexity demonstrated
- Problem-solving capabilities shown

6. **Key Strengths for This Role**
- Top 5 strengths with specific examples
- Unique value propositions
- Cultural/soft skill alignments

7. **Development Opportunities**
- Skills gaps with priority levels
- Suggested learning paths
- Timeline for skill acquisition

8. **Final Recommendation**
- Clear hiring recommendation (Highly Recommended/Recommended/Conditional/Not Recommended)
- Risk assessment
- Onboarding suggestions if hired
- Alternative role suggestions if applicable

Formatting Guidelines:
- Use clear headings and subheadings
- Include percentage scores and metrics where relevant
- Use bullet points for clarity
- Add emoji indicators for visual appeal (✅ ❌ ⚠️ 🌟 📊)
- Keep professional but engaging tone
- Make it scannable with good use of bold text
- Total length: 800-1200 words

Focus on actionable insights for both the recruiter and the candidate.

CRITICAL: Write the COMPLETE report covering all 8 sections in ENGLISH. Do not stop until you reach the Final Recommendation section."""

    def _generate_fallback_report(
        self, 
        job_requirements: JobRequirements,
        compatibility_analysis: Dict[str, Any],
        language: str
    ) -> str:
        """
        Generate a basic fallback report when AI generation fails.
        
        Args:
            job_requirements: Job requirements
            compatibility_analysis: Analysis results
            language: Report language
            
        Returns:
            Basic structured report
        """
        candidate_name = self.cv_data.get('name', 'Unknown Candidate')
        position_title = job_requirements.position_title or 'Unknown Position'
        overall_score = compatibility_analysis.get('overall_compatibility_score', 0)
        
        # Get skill analysis data
        skill_analysis = compatibility_analysis.get('skill_analysis', {})
        experience_analysis = compatibility_analysis.get('experience_analysis', {})
        education_analysis = compatibility_analysis.get('education_analysis', {})
        
        if language == "tr":
            return f"""# 📋 {candidate_name} - {position_title} Uyum Raporu

## ⭐ Genel Değerlendirme
**Genel Uyum Skoru:** {overall_score}% {'🌟🌟🌟🌟🌟' if overall_score >= 80 else '🌟🌟🌟🌟' if overall_score >= 60 else '🌟🌟🌟' if overall_score >= 40 else '🌟🌟'}

Bu adayın söz konusu pozisyon için genel uyum düzeyi **{overall_score}%** olarak değerlendirilmiştir. Bu rapor, aday profilinin detaylı analizi sonucunda oluşturulmuştur.

### 🎯 Öne Çıkan Noktalar:
• **Teknik Beceri Uyumu:** {skill_analysis.get('required_skills_match', 0)}%
• **Deneyim Kalitesi:** {experience_analysis.get('experience_quality_score', 0)}%
• **Eğitim Uygunluğu:** {'✅ Uygun' if education_analysis.get('meets_education_requirement', False) else '⚠️ Kısmen Uygun'}

## 🔧 Teknik Beceriler Analizi
**Gerekli Becerilerin Karşılanma Oranı:** {skill_analysis.get('required_skills_match', 0)}%

### ✅ Eşleşen Gerekli Beceriler:
{chr(10).join([f"• **{skill}** - Doğrulanmış yetkinlik" for skill in skill_analysis.get('matched_required_skills', [])]) or '• Detaylar analiz edilmekte'}

### ❌ Eksik Beceriler:
{chr(10).join([f"• **{skill}** - Geliştirilmesi önerilen alan" for skill in skill_analysis.get('missing_required_skills', [])]) or '• Büyük beceri eksikliği tespit edilmedi'}

### 🌟 Ek Değerli Beceriler:
{chr(10).join([f"• **{skill}** - Pozisyona ek değer katacak" for skill in skill_analysis.get('additional_relevant_skills', [])]) or '• Ek beceriler değerlendirilmekte'}

## 💼 Profesyonel Deneyim Değerlendirmesi
**Deneyim Gereksinimini Karşılama:** {'✅ Evet' if experience_analysis.get('meets_experience_requirement', False) else '❌ Hayır'}

**Tahmini İlgili Deneyim:** {experience_analysis.get('relevant_experience_years', 0)} yıl

### 🏆 İlgili Deneyimler:
{chr(10).join([f"• {exp}" for exp in experience_analysis.get('relevant_experiences', [])]) or '• Deneyim detayları analiz edilmekte'}

**Deneyim Kalite Skoru:** {experience_analysis.get('experience_quality_score', 0)}/100

## 🎓 Eğitim ve Sertifikasyonlar
**Eğitim Gereksinimini Karşılama:** {'✅ Evet' if education_analysis.get('meets_education_requirement', False) else '❌ Hayır'}

**Eğitim İlgililik Skoru:** {education_analysis.get('education_relevance_score', 0)}/100

### 📚 İlgili Eğitim Geçmişi:
{chr(10).join([f"• {edu}" for edu in education_analysis.get('relevant_education', [])]) or '• Eğitim detayları değerlendirilmekte'}

## 🚀 Proje Portföyü Analizi
**Proje İlgililik Skoru:** {compatibility_analysis.get('project_analysis', {}).get('project_relevance_score', 0)}/100

### 💡 İlgili Projeler:
{chr(10).join([f"• {proj}" for proj in compatibility_analysis.get('project_analysis', {}).get('relevant_projects', [])]) or '• Proje portföyü değerlendirilmekte'}

## ⭐ Bu Pozisyon İçin Güçlü Yönler
{chr(10).join([f"🌟 **{strength}**" for strength in compatibility_analysis.get('strengths', [])]) or '🌟 Güçlü yönler detaylandırılmakte'}

## 📈 Gelişim Fırsatları
{chr(10).join([f"⚠️ **{weakness}**" for weakness in compatibility_analysis.get('weaknesses', [])]) or '⚠️ Gelişim alanları belirlenmekte'}

## 💡 Öneriler
{chr(10).join([f"💡 {rec}" for rec in compatibility_analysis.get('recommendations', [])]) or '💡 Detaylı öneriler hazırlanmakta'}

## 🎯 Final Önerisi
{
    '🌟 **Güçlü Şekilde Önerilir** - Yüksek uyum gösteren, pozisyona mükemmel uygun aday' if overall_score >= 80
    else '✅ **Önerilir** - İyi uyum gösteren, değerlendirilmeye değer aday' if overall_score >= 60
    else '⚠️ **Koşullu Öneri** - Belirli alanlarda gelişim gerektiren aday' if overall_score >= 40
    else '❌ **Önerilmez** - Bu pozisyon için uygun olmayan aday profili'
}

### 📊 Risk Değerlendirmesi:
• **Teknik Risk:** {'Düşük' if skill_analysis.get('required_skills_match', 0) >= 70 else 'Orta' if skill_analysis.get('required_skills_match', 0) >= 50 else 'Yüksek'}
• **Deneyim Riski:** {'Düşük' if experience_analysis.get('experience_quality_score', 0) >= 70 else 'Orta' if experience_analysis.get('experience_quality_score', 0) >= 50 else 'Yüksek'}

---
*Bu rapor AI analizi ile otomatik olarak oluşturulmuştur. Detaylı değerlendirme için insan kaynakları uzmanı ile görüşme önerilir.*"""
        
        else:
            return f"""# 📋 {candidate_name} - {position_title} Compatibility Report

## ⭐ Executive Summary
**Overall Compatibility Score:** {overall_score}% {'🌟🌟🌟🌟🌟' if overall_score >= 80 else '🌟🌟🌟🌟' if overall_score >= 60 else '🌟🌟🌟' if overall_score >= 40 else '🌟🌟'}

This candidate shows a **{overall_score}%** overall compatibility for the specified position. This report is based on comprehensive analysis of the candidate's profile against job requirements.

### 🎯 Key Highlights:
• **Technical Skills Match:** {skill_analysis.get('required_skills_match', 0)}%
• **Experience Quality:** {experience_analysis.get('experience_quality_score', 0)}%
• **Education Fit:** {'✅ Suitable' if education_analysis.get('meets_education_requirement', False) else '⚠️ Partially Suitable'}

## 🔧 Technical Skills Assessment
**Required Skills Coverage:** {skill_analysis.get('required_skills_match', 0)}%

### ✅ Matched Required Skills:
{chr(10).join([f"• **{skill}** - Verified competency" for skill in skill_analysis.get('matched_required_skills', [])]) or '• Skills details under analysis'}

### ❌ Missing Skills:
{chr(10).join([f"• **{skill}** - Recommended development area" for skill in skill_analysis.get('missing_required_skills', [])]) or '• No major skill gaps identified'}

### 🌟 Additional Relevant Skills:
{chr(10).join([f"• **{skill}** - Adds value to the role" for skill in skill_analysis.get('additional_relevant_skills', [])]) or '• Additional skills being evaluated'}

## 💼 Professional Experience Evaluation
**Meets Experience Requirement:** {'✅ Yes' if experience_analysis.get('meets_experience_requirement', False) else '❌ No'}

**Estimated Relevant Experience:** {experience_analysis.get('relevant_experience_years', 0)} years

### 🏆 Relevant Experiences:
{chr(10).join([f"• {exp}" for exp in experience_analysis.get('relevant_experiences', [])]) or '• Experience details under analysis'}

**Experience Quality Score:** {experience_analysis.get('experience_quality_score', 0)}/100

## 🎓 Education & Certifications
**Meets Education Requirement:** {'✅ Yes' if education_analysis.get('meets_education_requirement', False) else '❌ No'}

**Education Relevance Score:** {education_analysis.get('education_relevance_score', 0)}/100

### 📚 Relevant Educational Background:
{chr(10).join([f"• {edu}" for edu in education_analysis.get('relevant_education', [])]) or '• Educational details being evaluated'}

## 🚀 Project Portfolio Analysis
**Project Relevance Score:** {compatibility_analysis.get('project_analysis', {}).get('project_relevance_score', 0)}/100

### 💡 Relevant Projects:
{chr(10).join([f"• {proj}" for proj in compatibility_analysis.get('project_analysis', {}).get('relevant_projects', [])]) or '• Project portfolio under evaluation'}

## ⭐ Key Strengths for This Role
{chr(10).join([f"🌟 **{strength}**" for strength in compatibility_analysis.get('strengths', [])]) or '🌟 Strengths being detailed'}

## 📈 Development Opportunities
{chr(10).join([f"⚠️ **{weakness}**" for weakness in compatibility_analysis.get('weaknesses', [])]) or '⚠️ Development areas being identified'}

## 💡 Recommendations
{chr(10).join([f"💡 {rec}" for rec in compatibility_analysis.get('recommendations', [])]) or '💡 Detailed recommendations being prepared'}

## 🎯 Final Recommendation
{
    '🌟 **Highly Recommended** - Excellent match showing high compatibility for the position' if overall_score >= 80
    else '✅ **Recommended** - Good match worth considering for the role' if overall_score >= 60
    else '⚠️ **Conditional Recommendation** - Candidate requiring development in certain areas' if overall_score >= 40
    else '❌ **Not Recommended** - Candidate profile not suitable for this position'
}

### 📊 Risk Assessment:
• **Technical Risk:** {'Low' if skill_analysis.get('required_skills_match', 0) >= 70 else 'Medium' if skill_analysis.get('required_skills_match', 0) >= 50 else 'High'}
• **Experience Risk:** {'Low' if experience_analysis.get('experience_quality_score', 0) >= 70 else 'Medium' if experience_analysis.get('experience_quality_score', 0) >= 50 else 'High'}

---
*This report was automatically generated through AI analysis. For detailed evaluation, consultation with HR specialist is recommended.*"""

    def generate_compatibility_report(
        self, 
        job_description: str, 
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compatibility report using LLM analysis.
        
        Args:
            job_description: Raw job description text
            language: Report language ("en" or "tr")
            
        Returns:
            Dictionary containing report text and metadata
        """
        # Language-specific error messages
        error_messages = {
            "tr": {
                "empty_description": "❌ İş tanımı boş. Lütfen geçerli bir iş tanımı girin.",
                "extraction_failed": "❌ İş bilgileri çıkarılamadı. İş tanımının net gereksinimler içerdiğinden emin olun.",
                "cv_context_failed": "❌ CV bilgileri alınamadı. Lütfen CV verilerinizi kontrol edin.",
                "analysis_failed": "❌ Uyum analizi hatası: {}",
                "unexpected_error": "❌ Beklenmeyen hata: {}"
            },
            "en": {
                "empty_description": "❌ Job description is empty. Please provide a valid job description.",
                "extraction_failed": "❌ Could not extract job information. Please ensure the description contains clear requirements.",
                "cv_context_failed": "❌ Could not retrieve CV information. Please check your CV data.",
                "analysis_failed": "❌ Error in compatibility analysis: {}",
                "unexpected_error": "❌ Unexpected error: {}"
            }
        }
        
        # Language-specific progress messages
        progress_messages = {
            "tr": {
                "analyzing_job": "📋 İş gereksinimleri analiz ediliyor...",
                "matching_cv": "🔍 CV bilgileri eşleştiriliyor...",
                "analyzing_compatibility": "🤖 Uyumluluk analiz ediliyor...",
                "generating_report": "📝 Detaylı rapor oluşturuluyor..."
            },
            "en": {
                "analyzing_job": "📋 Analyzing job requirements...",
                "matching_cv": "🔍 Matching CV information...",
                "analyzing_compatibility": "🤖 Analyzing compatibility...",
                "generating_report": "📝 Generating detailed report..."
            }
        }
        
        errors = error_messages.get(language, error_messages["en"])
        progress = progress_messages.get(language, progress_messages["en"])
        
        # Validate inputs
        if not job_description or not job_description.strip():
            return {
                "error": errors["empty_description"],
                "error_type": "validation"
            }
        
        try:
            # Step 1: Extract job requirements
            with st.spinner(progress["analyzing_job"]):
                job_requirements = self.extract_job_requirements(job_description)
                if not job_requirements.position_title:
                    return {
                        "error": errors["extraction_failed"],
                        "error_type": "extraction"
                    }
            
            # Step 2: Get relevant CV context
            with st.spinner(progress["matching_cv"]):
                cv_context = self.get_relevant_cv_context(job_requirements)
                if not cv_context:
                    return {
                        "error": errors["cv_context_failed"],
                        "error_type": "cv_context"
                    }
            
            # Step 3: Perform compatibility analysis
            with st.spinner(progress["analyzing_compatibility"]):
                compatibility_analysis = self.analyze_compatibility_with_llm(
                    job_requirements, 
                    cv_context
                )
                if "error" in compatibility_analysis and compatibility_analysis.get("overall_compatibility_score", 0) == 0:
                    return {
                        "error": errors["analysis_failed"].format(compatibility_analysis['error']),
                        "error_type": "analysis"
                    }
            
            # Step 4: Generate final report with retry mechanism
            with st.spinner(progress["generating_report"]):
                report_text = self._generate_report_with_retry(
                    job_requirements,
                    compatibility_analysis,
                    language
                )
            
            # Prepare successful response
            return {
                "report_text": report_text,
                "job_title": job_requirements.position_title,
                "compatibility_score": compatibility_analysis.get('overall_compatibility_score', 0),
                "metadata": {
                    "candidate_name": self.cv_data.get('name', 'Unknown'),
                    "analysis_date": str(st.session_state.get('current_date', '')),
                    "language": language,
                    "skill_match": compatibility_analysis.get('skill_analysis', {}).get('required_skills_match', 0),
                    "experience_match": compatibility_analysis.get('experience_analysis', {}).get('experience_quality_score', 0),
                    "report_length": len(report_text),
                    "is_complete": self._validate_report_completeness(report_text, language)
                }
            }
            
        except Exception as e:
            return {
                "error": errors["unexpected_error"].format(str(e)),
                "error_type": "system",
                "details": str(e)
            }


# Optional: Utility functions for external use
def format_compatibility_score(score: float) -> str:
    """
    Format compatibility score with visual indicators.
    
    Args:
        score: Compatibility score (0-100)
        
    Returns:
        Formatted score string
    """
    if score >= 80:
        return f"🌟 {score}% - Excellent Match"
    elif score >= 60:
        return f"✅ {score}% - Good Match"
    elif score >= 40:
        return f"⚠️ {score}% - Moderate Match"
    else:
        return f"❌ {score}% - Low Match"


def create_skill_badge(skill: str, matched: bool = True) -> str:
    """
    Create a visual badge for a skill.
    
    Args:
        skill: Skill name
        matched: Whether the skill is matched
        
    Returns:
        Formatted skill badge
    """
    icon = "✅" if matched else "❌"
    return f"{icon} {skill}"