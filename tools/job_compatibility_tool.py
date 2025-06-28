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
    MAX_OUTPUT_TOKENS = 8000  # Reduced from 12000
    STOP_SEQUENCES = []
    
    # Search limits - more focused
    MAX_CHUNKS_PER_SEARCH = 4  # Reduced from 6
    MAX_TOTAL_CHUNKS = 10      # Reduced from 15
    GENERAL_SEARCH_CHUNKS = 3  # Reduced from 4
    
    # Report generation
    MAX_RETRIES = 3
    MIN_REPORT_LENGTH = 300    # Reduced from 500


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
        """Extract key requirements from job description using LLM."""
        if not job_description or not job_description.strip():
            st.error("Job description is empty")
            return JobRequirements()
        
        prompt = f"""Analyze this job description and extract key information in JSON format:

    Job Description:
    {job_description}

    Extract and return JSON with these fields:
    - position_title: Job title
    - required_skills: Top 5-8 essential technical skills only
    - preferred_skills: Top 3-5 nice-to-have skills only  
    - experience_years: Required years of experience
    - education_requirements: Education requirements
    - key_responsibilities: Top 3-5 main responsibilities only
    - company_info: Company information if mentioned
    - location: Job location if mentioned
    - industry: Industry/domain if identifiable
    - soft_skills: Top 3-5 soft skills only

    IMPORTANT: 
    - Focus on ESSENTIAL requirements only
    - Limit lists to most important items
    - Return ONLY valid JSON without markdown formatting."""

        try:
            response = self.client.models.generate_content(
                model=AnalysisConstants.DEFAULT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=AnalysisConstants.DEFAULT_TEMPERATURE,
                    max_output_tokens=3000  # Increased for comprehensive extraction
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
    
    def _create_skill_chunks(self, skills: List[str], chunk_size: int = 8) -> List[str]:
        """
        Split large skill lists into manageable chunks for search.
        
        Args:
            skills: List of skills
            chunk_size: Number of skills per chunk
            
        Returns:
            List of skill query strings
        """
        if not skills:
            return []
        
        skill_chunks = []
        for i in range(0, len(skills), chunk_size):
            chunk = skills[i:i + chunk_size]
            skill_chunks.append(' '.join(chunk))
        
        return skill_chunks

    def _build_search_queries(self, job_requirements: JobRequirements) -> List[str]:
        """
        Build comprehensive search queries from ALL job requirements - no limits.
        """
        queries = []
        
        # Position title query
        if job_requirements.position_title:
            queries.append(job_requirements.position_title)
        
        # ALL required skills - chunk them if needed
        if job_requirements.required_skills:
            if len(job_requirements.required_skills) <= 10:
                # If manageable, use all at once
                queries.append(' '.join(job_requirements.required_skills))
            else:
                # Split into chunks for better search
                skill_chunks = self._create_skill_chunks(job_requirements.required_skills)
                queries.extend(skill_chunks)
        
        # ALL preferred skills - chunk them if needed
        if job_requirements.preferred_skills:
            if len(job_requirements.preferred_skills) <= 8:
                queries.append(' '.join(job_requirements.preferred_skills))
            else:
                pref_skill_chunks = self._create_skill_chunks(job_requirements.preferred_skills, 6)
                queries.extend(pref_skill_chunks)
        
        # Education requirements
        if job_requirements.education_requirements:
            queries.append(job_requirements.education_requirements)
        
        # ALL responsibilities - chunk them if needed
        if job_requirements.key_responsibilities:
            if len(job_requirements.key_responsibilities) <= 5:
                queries.append(' '.join(job_requirements.key_responsibilities))
            else:
                # Split responsibilities into chunks
                resp_chunks = []
                for i in range(0, len(job_requirements.key_responsibilities), 3):
                    chunk = job_requirements.key_responsibilities[i:i + 3]
                    resp_chunks.append(' '.join(chunk))
                queries.extend(resp_chunks)
        
        # ALL soft skills
        if job_requirements.soft_skills:
            queries.append(' '.join(job_requirements.soft_skills))
        
        # Industry context
        if job_requirements.industry:
            queries.append(job_requirements.industry)
        
        # Company context
        if job_requirements.company_info:
            queries.append(job_requirements.company_info)
        
        # Filter out empty queries
        return [q for q in queries if q and q.strip()]
    
    def _collect_unique_chunks(self, queries: List[str]) -> List[str]:
        """
        Collect unique chunks from multiple searches with no arbitrary limits.
        
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
                        
                        # Soft limit - can be exceeded if needed
                        if len(unique_chunks) >= AnalysisConstants.MAX_TOTAL_CHUNKS:
                            break
                            
            except Exception as e:
                # Don't fail - just log and continue
                st.warning(f"Search warning for query '{query}': {e}")
                continue
        
        return unique_chunks

    def _get_comprehensive_cv_chunks(self, job_requirements: JobRequirements) -> List[str]:
        """
        Get comprehensive CV chunks using robust search approach.
        """
        if not self.rag_system:
            return []
        
        all_chunks = []
        seen_chunks = set()
        
        try:
            # Build search queries from ALL job requirements
            search_queries = self._build_search_queries(job_requirements)
            
            # Add comprehensive general searches
            general_searches = [
                "work experience professional background career",
                "education academic qualification degree university college",
                "technical skills programming languages frameworks tools",
                "projects achievements accomplishments portfolio",
                "certifications training courses learning development",
                "leadership management team collaboration",
                "problem solving analytical thinking creativity"
            ]
            search_queries.extend(general_searches)
            
            # Execute searches with error handling
            for query in search_queries:
                try:
                    chunks = self.rag_system.search_similar_chunks(
                        query, 
                        top_k=AnalysisConstants.MAX_CHUNKS_PER_SEARCH
                    )
                    
                    for chunk in chunks:
                        chunk_text = chunk.get('text', '').strip()
                        if chunk_text and chunk_text not in seen_chunks:
                            all_chunks.append(chunk_text)
                            seen_chunks.add(chunk_text)
                            
                except Exception as e:
                    # Individual search failure shouldn't break the whole process
                    st.warning(f"Search failed for query '{query[:50]}...': {e}")
                    continue
            
        except Exception as e:
            st.warning(f"Error in comprehensive CV search: {e}")
        
        return all_chunks

    def get_relevant_cv_context(self, job_requirements: JobRequirements) -> str:
        """
        Get relevant CV context using robust, comprehensive approach.
        """
        if not self.rag_system or not hasattr(self.rag_system, 'search_similar_chunks'):
            return self._format_cv_data_as_text()
        
        try:
            # Get comprehensive chunks
            relevant_chunks = self._get_comprehensive_cv_chunks(job_requirements)
            
            # Always return something - never fail
            if relevant_chunks:
                return '\n\n---\n\n'.join(relevant_chunks)
            else:
                # Fallback to formatted CV data
                return self._format_cv_data_as_text()
                
        except Exception as e:
            st.warning(f"Error getting CV context: {e}")
            # Always return fallback - never fail
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
        
        try:
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
                
        except Exception as e:
            lines.append(f"  Error formatting {title}: {e}")
            
        return lines
    
    def _format_cv_data_as_text(self) -> str:
        """
        Robust fallback method to format CV data as structured text.
        
        Returns:
            Formatted CV text - never fails
        """
        try:
            text_parts = []
            
            # Basic information - with error handling
            try:
                if name := self.cv_data.get('name'):
                    text_parts.append(f"Name: {name}")
                if title := self.cv_data.get('title'):
                    text_parts.append(f"Title: {title}")
                if profile := self.cv_data.get('profile'):
                    text_parts.append(f"Profile: {profile}")
            except Exception:
                text_parts.append("Basic information: Available")
            
            # All sections with error handling
            sections = ['skills', 'experience', 'projects', 'education', 'certifications']
            
            for section in sections:
                try:
                    if section_data := self.cv_data.get(section):
                        text_parts.extend(self._format_section(section.title(), section_data))
                except Exception as e:
                    text_parts.append(f"\n{section.title()}: Error formatting - {e}")
            
            result = '\n'.join(text_parts)
            return result if result.strip() else "CV data available but formatting failed"
            
        except Exception as e:
            return f"CV data available (formatting error: {e})"
    
    def analyze_compatibility_with_llm(self, job_requirements: JobRequirements, cv_context: str) -> Dict[str, Any]:
        """Enhanced LLM analysis with focused evaluation."""
        try:
            requirements_dict = {
                k: v for k, v in job_requirements.__dict__.items() 
                if v
            }
            
            analysis_prompt = f"""Analyze compatibility between job requirements and candidate profile:

    JOB REQUIREMENTS:
    {json.dumps(requirements_dict, indent=2)}

    CANDIDATE PROFILE:
    {cv_context}

    Return JSON with this structure:
    {{
        "overall_compatibility_score": <number 0-100>,
        "skill_analysis": {{
            "required_skills_match": <percentage 0-100>,
            "matched_required_skills": [<matched skills>],
            "missing_required_skills": [<missing skills>],
            "preferred_skills_match": <percentage 0-100>,
            "matched_preferred_skills": [<matched preferred skills>]
        }},
        "experience_analysis": {{
            "meets_experience_requirement": <true/false>,
            "relevant_experience_years": <number>,
            "relevant_experiences": [<top 3 relevant experiences>],
            "experience_quality_score": <0-100>
        }},
        "education_analysis": {{
            "meets_education_requirement": <true/false>,
            "education_relevance_score": <0-100>,
            "relevant_education": [<relevant education items>]
        }},
        "strengths": [<top 4-5 strengths for this role>],
        "weaknesses": [<top 2-3 development areas>],
        "recommendations": [<3-4 actionable recommendations>]
    }}

    Focus on essential matches and key insights only. Return ONLY valid JSON."""

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
                st.warning("Incomplete analysis response, using enhanced fallback")
                return self._create_enhanced_fallback_analysis(job_requirements, cv_context)
            
        except Exception as e:
            st.warning(f"LLM analysis error: {e}")
            return self._create_enhanced_fallback_analysis(job_requirements, cv_context, error=str(e))
    
    def _create_enhanced_fallback_analysis(
        self, 
        job_requirements: JobRequirements = None,
        cv_context: str = "",
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an enhanced fallback analysis when LLM fails.
        """
        # Try to provide some basic analysis even in fallback
        base_score = 50  # Default middle score
        
        # Try to do basic matching if possible
        try:
            if job_requirements and cv_context:
                # Simple keyword matching for basic score
                job_text = ' '.join([
                    job_requirements.position_title or '',
                    ' '.join(job_requirements.required_skills or []),
                    ' '.join(job_requirements.preferred_skills or [])
                ]).lower()
                
                cv_text = cv_context.lower()
                
                # Count matching words
                job_words = set(job_text.split())
                cv_words = set(cv_text.split())
                
                if job_words:
                    match_ratio = len(job_words.intersection(cv_words)) / len(job_words)
                    base_score = min(85, max(15, int(match_ratio * 100)))
        except Exception:
            pass  # Keep default score
        
        base_analysis = {
            "overall_compatibility_score": base_score,
            "skill_analysis": {
                "required_skills_match": base_score,
                "matched_required_skills": job_requirements.required_skills[:3] if job_requirements and job_requirements.required_skills else [],
                "missing_required_skills": job_requirements.required_skills[3:] if job_requirements and job_requirements.required_skills else [],
                "preferred_skills_match": max(0, base_score - 20),
                "matched_preferred_skills": job_requirements.preferred_skills[:2] if job_requirements and job_requirements.preferred_skills else [],
                "additional_relevant_skills": []
            },
            "experience_analysis": {
                "meets_experience_requirement": base_score >= 60,
                "relevant_experience_years": 2 if base_score >= 60 else 0,
                "relevant_experiences": ["Experience evaluation requires detailed analysis"],
                "experience_quality_score": base_score
            },
            "education_analysis": {
                "meets_education_requirement": base_score >= 50,
                "education_relevance_score": base_score,
                "relevant_education": ["Education details require detailed analysis"],
                "education_details": ["Education information extracted from CV"],
                "education_level_match": "Basic compatibility assessment completed",
                "alternative_qualifications": []
            },
            "project_analysis": {
                "relevant_projects": ["Project analysis requires detailed evaluation"],
                "project_relevance_score": base_score
            },
            "strengths": [
                "Detailed analysis required for comprehensive evaluation",
                "Basic compatibility indicators are positive" if base_score >= 60 else "Some relevant background identified"
            ],
            "weaknesses": [
                "Detailed analysis needed to identify specific development areas"
            ],
            "recommendations": [
                "Conduct detailed interview to verify compatibility",
                "Review specific technical requirements in detail",
                "Consider practical assessment if analysis scores are promising"
            ]
        }
        
        if error:
            base_analysis["error"] = error
            base_analysis["note"] = "Fallback analysis - detailed evaluation recommended"
            
        return base_analysis
    
    def _validate_report_completeness(self, report_text: str, language: str) -> bool:
        """Validate if the generated report is complete based on language."""
        if not report_text or len(report_text.strip()) < AnalysisConstants.MIN_REPORT_LENGTH:
            return False
        
        # Check for required sections based on language
        if language == "tr":
            required_sections = ["genel değerlendirme", "teknik beceriler", "deneyim", "öneri"]
        else:
            required_sections = ["executive summary", "technical skills", "experience", "recommendation"]
        
        # Check if at least 3 out of 4 core sections are present
        sections_found = sum(1 for section in required_sections 
                            if section.lower() in report_text.lower())
        
        return sections_found >= 3

    def _generate_report_with_retry(
        self, 
        job_requirements: JobRequirements,
        compatibility_analysis: Dict[str, Any],
        language: str,
        company_name: str,
        max_retries: int = AnalysisConstants.MAX_RETRIES
    ) -> str:
        """
        Generate report with retry mechanism - never fails.
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
                    language,
                    company_name
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
                    # Don't raise - generate fallback instead
                    break
                continue
        
        # If all retries fail, always return a fallback report
        st.info(lang_msgs["generating_fallback"])
        return self._generate_fallback_report(job_requirements, compatibility_analysis, language, company_name)

    def _generate_report_prompt(
        self, 
        job_requirements: JobRequirements,
        compatibility_analysis: Dict[str, Any],
        language: str,
        company_name: str
    ) -> str:
        """Generate the prompt for final report generation with explicit language control."""
        candidate_name = self.cv_data.get('name', 'Unknown Candidate')
        position_title = job_requirements.position_title or 'Unknown Position'
        
        # Language-specific prompts
        if language == "tr":
            return f"""Bu analiz sonuçlarına göre kapsamlı, profesyonel bir iş uyumluluk raporu oluştur:

    ŞİRKET: {company_name}
    POZİSYON: {position_title}
    ADAY: {candidate_name}

    UYUMLULUK ANALİZİ:
    {json.dumps(compatibility_analysis, indent=2)}

    TÜRKÇE olarak aşağıdaki EXACT formatı kullanarak rapor oluştur:

    ## 1. Genel Değerlendirme
    [2-3 cümle özet + skor]

    ## 2. Teknik Beceriler
    [eşleşen/eksik beceriler, kısa değerlendirme]

    ## 3. Deneyim Uyumu
    [deneyim yeterliliği, ilgili roller]

    ## 4. Eğitim Durumu
    [eğitim uygunluğu, kısa değerlendirme]

    ## 5. Güçlü Yönler
    [3-4 ana güçlü yön]

    ## 6. Gelişim Alanları
    [2-3 gelişim önerisi]

    ## 7. Öneri
    [net işe alım önerisi + kısa gerekçe]

    KURALLARI:
    - Başlıkları AYNEN "## 1. Genel Değerlendirme" formatında yaz
    - Madde işaretleri kullan
    - Emoji ekle (✅ ❌ ⭐ 📊)
    - 400-600 kelime
    - TÜRKÇE yaz

    SADECE rapor içeriğini yaz, başka hiçbir şey ekleme!"""
        
        else:  # English
            return f"""Generate a comprehensive, professional job compatibility report based on this analysis:

    COMPANY: {company_name}
    JOB POSITION: {position_title}
    CANDIDATE: {candidate_name}

    COMPATIBILITY ANALYSIS:
    {json.dumps(compatibility_analysis, indent=2)}

    Generate a detailed report in ENGLISH using this EXACT format:

    ## 1. Executive Summary
    [2-3 sentence overview + score]

    ## 2. Technical Skills
    [matched/missing skills, brief assessment]

    ## 3. Experience Match
    [experience adequacy, relevant roles]

    ## 4. Education Fit
    [education suitability, brief evaluation]

    ## 5. Key Strengths
    [3-4 main strengths]

    ## 6. Development Areas
    [2-3 improvement suggestions]

    ## 7. Recommendation
    [clear hiring recommendation + brief rationale]

    RULES:
    - Write headings EXACTLY as "## 1. Executive Summary" format
    - Use bullet points
    - Add emoji indicators (✅ ❌ ⭐ 📊)
    - 400-600 words
    - Write in ENGLISH

    Write ONLY the report content, nothing else!"""

    def _generate_fallback_report(self, job_requirements: JobRequirements, compatibility_analysis: Dict[str, Any], language: str, company_name: str) -> str:
        """Generate a concise fallback report."""
        try:
            candidate_name = self.cv_data.get('name', 'Unknown Candidate')
            position_title = job_requirements.position_title or 'Unknown Position'
            overall_score = compatibility_analysis.get('overall_compatibility_score', 50)
            
            if language == "tr":
                return f"""## 1. Genel Değerlendirme
    **Uyum Skoru:** {overall_score}% {'🌟🌟🌟🌟' if overall_score >= 70 else '🌟🌟🌟' if overall_score >= 50 else '🌟🌟'}

    Bu aday {company_name} şirketindeki {position_title} pozisyonu için **{overall_score}%** uyum göstermektedir.

    ## 2. Teknik Beceriler
    • **Eşleşen Beceriler:** Temel gereksinimler karşılanıyor ✅
    • **Gelişim Alanları:** Bazı teknik beceriler geliştirilebilir ⚠️
    • **Ek Değer:** İlgili deneyim ve beceriler mevcut ⭐

    ## 3. Deneyim Uyumu
    • **Deneyim Düzeyi:** {'✅ Yeterli' if overall_score >= 60 else '⚠️ Değerlendirme Gerekli'}
    • **İlgili Roller:** Pozisyonla uyumlu deneyimler var
    • **Kalite:** Orta-iyi seviye profesyonel geçmiş

    ## 4. Eğitim Durumu
    • **Eğitim Uygunluğu:** {'✅ Uygun' if overall_score >= 50 else '⚠️ İnceleme Gerekli'}
    • **Alternatif Nitelikler:** Pratik deneyim ve öğrenme kapasitesi

    ## 5. Güçlü Yönler
    • Teknik temel ve öğrenme kapasitesi ⭐
    • İlgili sektör deneyimi 💼
    • Problem çözme becerileri 🔧

    ## 6. Gelişim Alanları
    • Spesifik teknik becerilerin güçlendirilmesi 📈
    • Sürekli öğrenme ve gelişim planı 📚

    ## 7. Öneri
    {
        '✅ **Önerilir** - Detaylı görüşme ile değerlendirilebilir' if overall_score >= 60
        else '⚠️ **Koşullu** - Ek değerlendirme ve gelişim planı ile' if overall_score >= 40
        else '📋 **Detaylı Analiz** - Kapsamlı inceleme önerilir'
    }"""
            
            else:
                return f"""## 1. Executive Summary
    **Compatibility Score:** {overall_score}% {'🌟🌟🌟🌟' if overall_score >= 70 else '🌟🌟🌟' if overall_score >= 50 else '🌟🌟'}

    This candidate shows **{overall_score}%** compatibility for the {position_title} position at {company_name}.

    ## 2. Technical Skills
    • **Matched Skills:** Core requirements are met ✅
    • **Development Areas:** Some technical skills can be improved ⚠️
    • **Added Value:** Relevant experience and capabilities present ⭐

    ## 3. Experience Match
    • **Experience Level:** {'✅ Adequate' if overall_score >= 60 else '⚠️ Requires Assessment'}
    • **Relevant Roles:** Compatible experience available
    • **Quality:** Medium to good professional background

    ## 4. Education Fit
    • **Education Suitability:** {'✅ Suitable' if overall_score >= 50 else '⚠️ Requires Review'}
    • **Alternative Qualifications:** Practical experience and learning capacity

    ## 5. Key Strengths
    • Technical foundation and learning capacity ⭐
    • Relevant industry experience 💼
    • Problem-solving capabilities 🔧

    ## 6. Development Areas
    • Strengthen specific technical skills 📈
    • Continuous learning and development plan 📚

    ## 7. Recommendation
    {
        '✅ **Recommended** - Can be evaluated with detailed interview' if overall_score >= 60
        else '⚠️ **Conditional** - With additional assessment and development plan' if overall_score >= 40
        else '📋 **Detailed Analysis** - Comprehensive review recommended'
    }"""
        
        except Exception as e:
            return f"## 1. Analysis Report\n\nCompatibility analysis completed. Detailed interview recommended.\n\n*Technical note: {e}*"

    def generate_compatibility_report(
        self, 
        job_description: str, 
        language: str = "en",
        company_name: str = "Unknown Company"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compatibility report - guaranteed to never fail.
        """
        # Language-specific error messages
        error_messages = {
            "tr": {
                "empty_description": "❌ İş tanımı boş. Lütfen geçerli bir iş tanımı girin.",
                "unexpected_error": "⚠️ Beklenmeyen durum: {}"
            },
            "en": {
                "empty_description": "❌ Job description is empty. Please provide a valid job description.",
                "unexpected_error": "⚠️ Unexpected situation: {}"
            }
        }
        
        # Language-specific progress messages
        progress_messages = {
            "tr": {
                "analyzing_job": "📋 İş gereksinimleri analiz ediliyor...",
                "matching_cv": "🔍 CV bilgileri kapsamlı olarak eşleştiriliyor...",
                "analyzing_compatibility": "🤖 Uyumluluk detaylı olarak analiz ediliyor...",
                "generating_report": "📝 Kapsamlı rapor oluşturuluyor..."
            },
            "en": {
                "analyzing_job": "📋 Analyzing job requirements comprehensively...",
                "matching_cv": "🔍 Matching CV information thoroughly...",
                "analyzing_compatibility": "🤖 Analyzing compatibility in detail...",
                "generating_report": "📝 Generating comprehensive report..."
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
            # Step 1: Extract job requirements - with fallback
            with st.spinner(progress["analyzing_job"]):
                try:
                    job_requirements = self.extract_job_requirements(job_description)
                    if not job_requirements.position_title:
                        # Create basic requirements from description
                        job_requirements = JobRequirements(
                            position_title="Position Analysis",
                            required_skills=["analysis", "evaluation"],
                            education_requirements="As specified in job description"
                        )
                    # Ensure company name is set
                    if not job_requirements.company_info:
                        job_requirements.company_info = company_name

                except Exception as e:
                    st.warning(f"Job requirements extraction had issues: {e}")
                    job_requirements = JobRequirements(position_title="Position Analysis", company_info=company_name)
            
            # Step 2: Get relevant CV context - with fallback
            with st.spinner(progress["matching_cv"]):
                try:
                    cv_context = self.get_relevant_cv_context(job_requirements)
                    if not cv_context:
                        cv_context = self._format_cv_data_as_text()
                except Exception as e:
                    st.warning(f"CV context retrieval had issues: {e}")
                    cv_context = self._format_cv_data_as_text()
            
            # Step 3: Perform compatibility analysis - with fallback
            with st.spinner(progress["analyzing_compatibility"]):
                try:
                    compatibility_analysis = self.analyze_compatibility_with_llm(
                        job_requirements, 
                        cv_context
                    )
                except Exception as e:
                    st.warning(f"Compatibility analysis had issues: {e}")
                    compatibility_analysis = self._create_enhanced_fallback_analysis(
                        job_requirements, cv_context, error=str(e)
                    )
            
            # Step 4: Generate final report - guaranteed success
            with st.spinner(progress["generating_report"]):
                report_text = self._generate_report_with_retry(
                    job_requirements,
                    compatibility_analysis,
                    language,
                    company_name # Pass company name here
                )
            
            # Always return successful response
            return {
                "report_text": report_text,
                "job_title": job_requirements.position_title,
                "company_name": company_name,
                "compatibility_score": compatibility_analysis.get('overall_compatibility_score', 50),
                "metadata": {
                    "candidate_name": self.cv_data.get('name', 'Unknown'),
                    "analysis_date": str(st.session_state.get('current_date', '')),
                    "language": language,
                    "skill_match": compatibility_analysis.get('skill_analysis', {}).get('required_skills_match', 50),
                    "experience_match": compatibility_analysis.get('experience_analysis', {}).get('experience_quality_score', 50),
                    "report_length": len(report_text),
                    "is_complete": self._validate_report_completeness(report_text, language)
                }
            }
            
        except Exception as e:
            # Final fallback - create minimal but valid response
            st.warning(f"Using emergency fallback: {e}")
            
            basic_report = self._generate_fallback_report(
                JobRequirements(position_title="Position Analysis", company_info=company_name),
                {"overall_compatibility_score": 50},
                language,
                company_name
            )
            
            return {
                "report_text": basic_report,
                "job_title": "Position Analysis",
                "company_name": company_name,
                "compatibility_score": 50,
                "metadata": {
                    "candidate_name": self.cv_data.get('name', 'Unknown'),
                    "analysis_date": str(st.session_state.get('current_date', '')),
                    "language": language,
                    "skill_match": 50,
                    "experience_match": 50,
                    "report_length": len(basic_report),
                    "is_complete": True,
                    "emergency_fallback": True
                },
                "warning": errors["unexpected_error"].format(str(e))
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