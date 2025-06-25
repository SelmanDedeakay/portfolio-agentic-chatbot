# tools/job_compatibility_tool.py

import re
from typing import Dict, List, Any
import json
import streamlit as st
from google import genai
from google.genai import types
import numpy as np


class JobCompatibilityAnalyzer:
    """Analyze job compatibility between CV and job description using RAG chunks"""
    
    def __init__(self, client, cv_data: Dict, rag_system=None):
        self.client = client
        self.cv_data = cv_data
        self.rag_system = rag_system  # Reference to the main RAG system for chunk search
        
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """Extract key requirements from job description using LLM"""
        prompt = f"""Analyze this job description and extract key information in JSON format:

Job Description:
{job_description}

Please extract and return a JSON with these fields:
- position_title: Job title
- required_skills: List of technical skills mentioned (be comprehensive)
- preferred_skills: List of nice-to-have skills
- experience_years: Required years of experience (number or "entry-level")
- education_requirements: Education requirements
- key_responsibilities: Main job responsibilities (list)
- company_info: Any company information mentioned
- location: Job location if mentioned
- industry: Industry/domain if identifiable
- soft_skills: Any soft skills mentioned

Return only valid JSON without markdown formatting."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            # Clean response and extract JSON
            response_text = response.text.strip()
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*$', '', response_text)
            
            return json.loads(response_text)
            
        except Exception as e:
            st.error(f"Error extracting job requirements: {e}")
            return {}
    
    def get_relevant_cv_context(self, job_requirements: Dict) -> str:
        """Get relevant CV chunks based on job requirements using RAG search"""
        if not self.rag_system or not hasattr(self.rag_system, 'search_similar_chunks'):
            # Fallback to basic CV data if RAG system not available
            return self._format_cv_data_as_text()
        
        try:
            # Create search queries based on job requirements
            search_queries = []
            
            # Add position title for general relevance
            if job_requirements.get('position_title'):
                search_queries.append(job_requirements['position_title'])
            
            # Add required skills
            required_skills = job_requirements.get('required_skills', [])
            if required_skills:
                search_queries.append(' '.join(required_skills[:5]))  # Top 5 skills
            
            # Add key responsibilities
            responsibilities = job_requirements.get('key_responsibilities', [])
            if responsibilities:
                search_queries.append(' '.join(responsibilities[:3]))  # Top 3 responsibilities
            
            # Add industry/domain context
            if job_requirements.get('industry'):
                search_queries.append(job_requirements['industry'])
            
            # Perform searches and collect unique chunks
            all_relevant_chunks = []
            seen_chunks = set()
            
            for query in search_queries:
                if query and query.strip():
                    chunks = self.rag_system.search_similar_chunks(query, top_k=4)
                    for chunk in chunks:
                        chunk_text = chunk.get('text', '')
                        if chunk_text and chunk_text not in seen_chunks:
                            all_relevant_chunks.append(chunk_text)
                            seen_chunks.add(chunk_text)
            
            # Also search for general experience and projects
            exp_chunks = self.rag_system.search_similar_chunks("work experience projects", top_k=3)
            for chunk in exp_chunks:
                chunk_text = chunk.get('text', '')
                if chunk_text and chunk_text not in seen_chunks:
                    all_relevant_chunks.append(chunk_text)
                    seen_chunks.add(chunk_text)
            
            return '\n\n---\n\n'.join(all_relevant_chunks[:10])  # Limit to top 10 chunks
            
        except Exception as e:
            st.error(f"Error getting CV context: {e}")
            return self._format_cv_data_as_text()
    
    def _format_cv_data_as_text(self) -> str:
        """Fallback method to format CV data as text"""
        try:
            text_parts = []
            
            # Basic info
            text_parts.append(f"Name: {self.cv_data.get('name', 'N/A')}")
            text_parts.append(f"Title: {self.cv_data.get('title', 'N/A')}")
            text_parts.append(f"Profile: {self.cv_data.get('profile', 'N/A')}")
            
            # Skills
            skills = self.cv_data.get('skills', {})
            if skills:
                text_parts.append("Skills:")
                for category, skill_list in skills.items():
                    if isinstance(skill_list, list):
                        text_parts.append(f"  {category}: {', '.join(skill_list)}")
            
            # Experience
            experience = self.cv_data.get('experience', [])
            if experience:
                text_parts.append("Experience:")
                for exp in experience:
                    if isinstance(exp, dict):
                        text_parts.append(f"  {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
                        text_parts.append(f"    Duration: {exp.get('duration', 'N/A')}")
                        text_parts.append(f"    Description: {exp.get('description', 'N/A')}")
            
            # Projects
            projects = self.cv_data.get('projects', [])
            if projects:
                text_parts.append("Projects:")
                for proj in projects:
                    if isinstance(proj, dict):
                        text_parts.append(f"  {proj.get('name', 'N/A')}")
                        text_parts.append(f"    Technology: {proj.get('technology', 'N/A')}")
                        text_parts.append(f"    Description: {proj.get('description', 'N/A')}")
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            return f"Error formatting CV data: {e}"
    
    def analyze_compatibility_with_llm(self, job_requirements: Dict, cv_context: str) -> Dict[str, Any]:
        """Use LLM to analyze compatibility between job requirements and CV context"""
        
        analysis_prompt = f"""You are an expert HR analyst. Analyze the compatibility between this job requirement and candidate profile.

JOB REQUIREMENTS:
{json.dumps(job_requirements, indent=2)}

CANDIDATE PROFILE (from CV):
{cv_context}

Please analyze and return a JSON response with the following structure:
{{
    "overall_compatibility_score": <number 0-100>,
    "skill_analysis": {{
        "required_skills_match": <number 0-100>,
        "matched_required_skills": [<list of matched skills>],
        "missing_required_skills": [<list of missing skills>],
        "preferred_skills_match": <number 0-100>,
        "matched_preferred_skills": [<list of matched skills>],
        "additional_relevant_skills": [<skills candidate has that are relevant but not explicitly mentioned>]
    }},
    "experience_analysis": {{
        "meets_experience_requirement": <true/false>,
        "relevant_experience_years": <estimated years>,
        "relevant_experiences": [<list of relevant work experiences>],
        "experience_quality_score": <number 0-100>
    }},
    "education_analysis": {{
        "meets_education_requirement": <true/false>,
        "education_relevance_score": <number 0-100>,
        "relevant_education": [<list of relevant education>]
    }},
    "project_analysis": {{
        "relevant_projects": [<list of relevant projects>],
        "project_relevance_score": <number 0-100>
    }},
    "strengths": [<list of candidate's key strengths for this role>],
    "weaknesses": [<list of areas where candidate may need development>],
    "recommendations": [<list of recommendations for both candidate and employer>]
}}

Be thorough in your analysis. Consider not just exact keyword matches, but also transferable skills, related technologies, and relevant experience patterns. For example, if the job requires "React" and the candidate has "JavaScript" and "Frontend Development" experience, that's a strong indicator of compatibility.

Return only valid JSON without markdown formatting."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8000,
                    stop_sequences=["---END---"]
                )
            )
            
            # Clean and parse response
            response_text = response.text.strip()
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*$', '', response_text)
            
            return json.loads(response_text)
            
        except Exception as e:
            st.error(f"Error in LLM compatibility analysis: {e}")
            return {
                "overall_compatibility_score": 0,
                "error": str(e)
            }
    
    def generate_compatibility_report(self, job_description: str, language: str = "en") -> Dict[str, str]:
        """Generate comprehensive compatibility report using LLM analysis"""
        
        try:
            # Extract job requirements
            job_requirements = self.extract_job_requirements(job_description)
            if not job_requirements:
                return {"error": "❌ Could not analyze job description. Please ensure it contains clear requirements."}
            
            # Get relevant CV context using RAG
            cv_context = self.get_relevant_cv_context(job_requirements)
            if not cv_context:
                return {"error": "❌ Could not retrieve relevant CV information."}
            
            # Perform LLM-based compatibility analysis
            compatibility_analysis = self.analyze_compatibility_with_llm(job_requirements, cv_context)
            if "error" in compatibility_analysis:
                return {"error": f"❌ Error in compatibility analysis: {compatibility_analysis['error']}"}
            
            # Generate final report
            report_prompt = f"""Generate a comprehensive, professional job compatibility report based on this analysis:

JOB POSITION: {job_requirements.get('position_title', 'Not specified')}

COMPATIBILITY ANALYSIS:
{json.dumps(compatibility_analysis, indent=2)}

CANDIDATE: {self.cv_data.get('name', 'Unknown')}

Generate a detailed report in {"Turkish" if language == "tr" else "English"} that includes:

1. **Executive Summary** - Overall compatibility score and recommendation
2. **Skills Assessment** - Detailed breakdown of technical skills match
3. **Experience Evaluation** - How candidate's experience aligns with requirements
4. **Education & Qualifications** - Relevance of educational background
5. **Project Portfolio Relevance** - How candidate's projects demonstrate required capabilities
6. **Key Strengths** - What makes this candidate stand out for this role
7. **Development Areas** - Skills or experience gaps and how to address them
8. **Final Recommendation** - Clear hiring recommendation with reasoning

Use professional language, provide specific examples from the candidate's background, and make it actionable for both recruiters and the candidate. Include percentage scores where relevant and use markdown formatting for better readability.

The report should be comprehensive but concise, focusing on the most relevant aspects for the hiring decision."""

            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=report_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8000,  # Daha yüksek limit
    stop_sequences=["---END---"]  # Kontrollü sonlandırma
                )
            )
            
            return {
                "report_text": response.text,
                "job_title": job_requirements.get('position_title', 'Unknown Position'),
                "compatibility_score": compatibility_analysis.get('overall_compatibility_score', 0)
            }
            
        except Exception as e:
            return {"error": f"❌ Error generating report: {str(e)}"}