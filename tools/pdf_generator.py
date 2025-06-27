import io
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import requests
from urllib.parse import urlparse

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, 
    HRFlowable, KeepTogether, Image, Flowable, FrameBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Line
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage, ImageDraw
import tempfile
import os
import urllib.request


class PDFConstants:
    """Enhanced constants for PDF generation with Turkish font support"""
    # Page settings
    PAGE_SIZE = A4
    LEFT_MARGIN = 2.0 * cm
    RIGHT_MARGIN = 2.0 * cm
    TOP_MARGIN = 2.0 * cm
    BOTTOM_MARGIN = 2.5 * cm
    
    # Font settings - will be set by FontManager
    DEFAULT_FONT = 'DejaVuSans'
    BOLD_FONT = 'DejaVuSans-Bold'
    ITALIC_FONT = 'DejaVuSans-Oblique'
    
    # Score thresholds
    HIGH_SCORE_THRESHOLD = 80
    MEDIUM_SCORE_THRESHOLD = 60
    
    # Enhanced visual elements
    SCORE_BOX_HEIGHT = 35
    PROGRESS_BAR_HEIGHT = 8
    PROFILE_PHOTO_SIZE = 120
    HEADER_HEIGHT = 80
    
    # Section management
    MIN_SECTION_HEIGHT = 120
    
    # Enhanced emoji replacements with Turkish-friendly symbols
    EMOJI_REPLACEMENTS = {
        '✅': '✓', '❌': '✗', '⭐': '★', '🔍': '🔎', '💡': '●',
        '📊': '■', '🎯': '●', '⚡': '⚡', '🚀': '▲', '📌': '●',
        '💪': '●', '🔧': '●', '📈': '↗', '👍': '✓', '👎': '✗',
        '✨': '★', '🏆': '★', '🎓': '●', '💼': '■', '🌟': '★',
    }


class FontManager:
    """Manages font registration with Turkish character support"""
    
    _fonts_registered = False
    _available_fonts = {}
    
    @classmethod
    def setup_fonts(cls):
        """Setup fonts with Turkish character support"""
        if cls._fonts_registered:
            return cls._available_fonts
        
        try:
            # Try to register DejaVu fonts (best Turkish support)
            cls._register_dejavu_fonts()
            cls._available_fonts = {
                'default': 'DejaVuSans',
                'bold': 'DejaVuSans-Bold',
                'italic': 'DejaVuSans-Oblique'
            }
            print("DejaVu fonts registered successfully")
            
        except Exception as e:
            print(f"DejaVu fonts not available: {e}")
            try:
                # Fallback to Liberation fonts (good Turkish support)
                cls._register_liberation_fonts()
                cls._available_fonts = {
                    'default': 'LiberationSans',
                    'bold': 'LiberationSans-Bold',
                    'italic': 'LiberationSans-Italic'
                }
                print("Liberation fonts registered successfully")
                
            except Exception as e2:
                print(f"Liberation fonts not available: {e2}")
                # Final fallback to Times (has some Turkish support)
                cls._available_fonts = {
                    'default': 'Times-Roman',
                    'bold': 'Times-Bold',
                    'italic': 'Times-Italic'
                }
                print("Using Times fonts as fallback")
        
        # Update PDFConstants
        PDFConstants.DEFAULT_FONT = cls._available_fonts['default']
        PDFConstants.BOLD_FONT = cls._available_fonts['bold']
        PDFConstants.ITALIC_FONT = cls._available_fonts['italic']
        
        cls._fonts_registered = True
        return cls._available_fonts
    
    @classmethod
    def _register_dejavu_fonts(cls):
        """Register DejaVu fonts from system or download"""
        font_paths = cls._find_dejavu_fonts()
        
        
        # Register fonts
        pdfmetrics.registerFont(TTFont('DejaVuSans', font_paths['regular']))
        if font_paths['bold']:
            pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', font_paths['bold']))
        if font_paths['oblique']:
            pdfmetrics.registerFont(TTFont('DejaVuSans-Oblique', font_paths['oblique']))
        
        # Register font family
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily('DejaVuSans',
                          normal='DejaVuSans',
                          bold='DejaVuSans-Bold' if font_paths['bold'] else 'DejaVuSans',
                          italic='DejaVuSans-Oblique' if font_paths['oblique'] else 'DejaVuSans')
    
    @classmethod
    def _find_dejavu_fonts(cls) -> Dict[str, Optional[str]]:
        """Find DejaVu fonts in local fonts directory"""
        font_files = {
            'regular': 'DejaVuSans.ttf',
            'bold': 'DejaVuSans-Bold.ttf', 
            'oblique': 'DejaVuSans-Oblique.ttf'
        }
        
        found_fonts = {'regular': None, 'bold': None, 'oblique': None}
        
        fonts_dir = "fonts/"
        if os.path.exists(fonts_dir):
            for font_type, filename in font_files.items():
                font_path = os.path.join(fonts_dir, filename)
                if os.path.exists(font_path):
                    found_fonts[font_type] = font_path
        
        return found_fonts
    

    @classmethod
    def _register_liberation_fonts(cls):
        """Register Liberation fonts as fallback"""
        # This is a placeholder - Liberation fonts would need to be downloaded/found similarly
        # For now, we'll skip this and go to the final fallback
        raise Exception("Liberation fonts not implemented yet")
    
    @classmethod
    def get_fonts(cls):
        """Get available fonts"""
        if not cls._fonts_registered:
            return cls.setup_fonts()
        return cls._available_fonts


class EnhancedColorScheme:
    """Enhanced color scheme with gradients and professional colors"""
    # Primary colors
    primary = colors.HexColor('#1a365d')           # Deep blue
    primary_light = colors.HexColor('#2d3748')     # Lighter primary
    secondary = colors.HexColor('#3182ce')         # Bright blue
    secondary_light = colors.HexColor('#63b3ed')   # Light blue
    
    # Status colors
    success = colors.HexColor('#38a169')           # Green
    success_light = colors.HexColor('#c6f6d5')     # Light green
    warning = colors.HexColor('#d69e2e')           # Orange
    warning_light = colors.HexColor('#faf089')     # Light orange
    danger = colors.HexColor('#e53e3e')            # Red
    danger_light = colors.HexColor('#fed7d7')      # Light red
    
    # Neutral colors
    text_primary = colors.HexColor('#2d3748')      # Dark gray
    text_secondary = colors.HexColor('#4a5568')    # Medium gray
    text_muted = colors.HexColor('#718096')        # Light gray
    
    # Background colors
    bg_primary = colors.white
    bg_secondary = colors.HexColor('#f7fafc')      # Very light gray
    bg_accent = colors.HexColor('#edf2f7')         # Light gray
    
    # Borders
    border_light = colors.HexColor('#e2e8f0')      # Light border
    border_medium = colors.HexColor('#cbd5e0')     # Medium border
    
    @classmethod
    def get_score_colors(cls, score: float) -> Tuple[colors.Color, colors.Color]:
        """Get enhanced colors based on score value"""
        if score >= PDFConstants.HIGH_SCORE_THRESHOLD:
            return cls.success, cls.success_light
        elif score >= PDFConstants.MEDIUM_SCORE_THRESHOLD:
            return cls.warning, cls.warning_light
        else:
            return cls.danger, cls.danger_light


class Language(Enum):
    """Supported languages for PDF generation"""
    ENGLISH = "en"
    TURKISH = "tr"


@dataclass
class DocumentMetadata:
    """Enhanced metadata for PDF document"""
    candidate_name: str
    job_title: str
    language: Language
    generation_date: str = None
    profile_photo_url: str = "https://avatars.githubusercontent.com/u/77899403?s=400&u=701512e5e0fc5d6deae5b4269c0382b8f9ee95be&v=4"
    
    def __post_init__(self):
        if not self.generation_date:
            self.generation_date = datetime.now().strftime("%d/%m/%Y")


class ImageHandler:
    """Handle profile photo processing"""
    
    @staticmethod
    def download_and_process_image(url: str, size: int = 60) -> Optional[str]:
        """Download and create circular profile image"""
        try:
            # Download image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Open with PIL
            img = PILImage.open(io.BytesIO(response.content))
            
            # Resize to square
            img = img.resize((size * 2, size * 2), PILImage.Resampling.LANCZOS)
            
            # Create circular mask
            mask = PILImage.new('L', (size * 2, size * 2), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size * 2, size * 2), fill=255)
            
            # Apply mask
            img = img.convert("RGBA")
            img.putalpha(mask)
            
            # Save to temporary file with proper handling
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file_path = temp_file.name
            temp_file.close()  # Close the file handle first
            
            img.save(temp_file_path, 'PNG')
            
            return temp_file_path
            
        except Exception as e:
            print(f"Error processing profile image: {e}")
            return None


class EnhancedScoreBox(Flowable):
    """Enhanced score display with gradient and better visuals"""
    
    def __init__(self, score_text: str, width: float, 
                 height: float = PDFConstants.SCORE_BOX_HEIGHT,
                 color_scheme: EnhancedColorScheme = None,
                 fonts: dict = None):
        Flowable.__init__(self)
        self.score_text = score_text
        self.width = width
        self.height = height
        self.color_scheme = color_scheme or EnhancedColorScheme()
        self.fonts = fonts or FontManager.get_fonts()
        self.score_value = self._extract_score()
    
    def _extract_score(self) -> float:
        """Extract numerical score from text"""
        percentages = re.findall(r'(\d+\.?\d*)%', self.score_text)
        return float(percentages[0]) if percentages else 0.0
    
    def draw(self):
        """Draw enhanced score box with shadow and gradient effect"""
        main_color, bg_color = self.color_scheme.get_score_colors(self.score_value)
        
        # Draw shadow
        shadow_offset = 2
        self.canv.setFillColor(colors.HexColor('#00000020'))
        self.canv.roundRect(shadow_offset, -shadow_offset, self.width, self.height, 5, fill=1, stroke=0)
        
        # Main background with gradient effect
        self.canv.setFillColor(bg_color)
        self.canv.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=1)
        
        # Enhanced border
        self.canv.setStrokeColor(main_color)
        self.canv.setLineWidth(2)
        self.canv.roundRect(0, 0, self.width, self.height, 5, fill=0, stroke=1)
        
        # Progress bar with enhanced styling
        bar_margin = 20
        bar_width = self.width - (2 * bar_margin)
        bar_y = 8
        
        # Background bar with rounded ends
        self.canv.setFillColor(self.color_scheme.border_light)
        self.canv.roundRect(bar_margin, bar_y, bar_width, PDFConstants.PROGRESS_BAR_HEIGHT, 4, fill=1, stroke=0)
        
        # Filled bar with gradient effect
        progress_width = bar_width * (self.score_value / 100)
        if progress_width > 0:
            self.canv.setFillColor(main_color)
            self.canv.roundRect(bar_margin, bar_y, progress_width, PDFConstants.PROGRESS_BAR_HEIGHT, 4, fill=1, stroke=0)
            
            # Add highlight
            self.canv.setFillColor(colors.HexColor('#ffffff40'))
            self.canv.roundRect(bar_margin, bar_y + 2, progress_width, 2, 2, fill=1, stroke=0)
        
        # Score text with enhanced styling
        self.canv.setFont(self.fonts['bold'], 12)
        self.canv.setFillColor(self.color_scheme.text_primary)
        text_y = self.height - 16
        self.canv.drawCentredString(self.width / 2, text_y, self.score_text)
        
        # Score icon
        if self.score_value >= PDFConstants.HIGH_SCORE_THRESHOLD:
            icon = "★"
        elif self.score_value >= PDFConstants.MEDIUM_SCORE_THRESHOLD:
            icon = "●"
        else:
            icon = "▲"
        
        self.canv.setFont(self.fonts['default'], 10)
        self.canv.setFillColor(main_color)
        self.canv.drawString(10, text_y, icon)
        self.canv.drawRightString(self.width - 10, text_y, icon)


class SectionDivider(Flowable):
    """Custom section divider with enhanced styling"""
    
    def __init__(self, width: float, title: str = "", 
                 color_scheme: EnhancedColorScheme = None,
                 fonts: dict = None):
        Flowable.__init__(self)
        self.width = width
        self.height = 25
        self.title = title
        self.color_scheme = color_scheme or EnhancedColorScheme()
        self.fonts = fonts or FontManager.get_fonts()
    
    def draw(self):
        """Draw enhanced section divider"""
        # Main line
        self.canv.setStrokeColor(self.color_scheme.secondary)
        self.canv.setLineWidth(3)
        y = self.height / 2
        self.canv.line(0, y, self.width, y)
        
        # Decorative elements
        circle_radius = 4
        self.canv.setFillColor(self.color_scheme.secondary)
        self.canv.circle(circle_radius, y, circle_radius, fill=1, stroke=0)
        self.canv.circle(self.width - circle_radius, y, circle_radius, fill=1, stroke=0)
        
        # Title background if provided
        if self.title:
            text_width = len(self.title) * 6 + 20
            bg_x = (self.width - text_width) / 2
            
            self.canv.setFillColor(self.color_scheme.bg_primary)
            self.canv.roundRect(bg_x, y - 8, text_width, 16, 3, fill=1, stroke=1)
            
            self.canv.setFont(self.fonts['bold'], 10)
            self.canv.setFillColor(self.color_scheme.secondary)
            self.canv.drawCentredString(self.width / 2, y - 4, self.title)


class EnhancedStyleManager:
    """Enhanced style manager with better typography and Turkish support"""
    
    def __init__(self, color_scheme: EnhancedColorScheme = None):
        self.color_scheme = color_scheme or EnhancedColorScheme()
        self.styles = getSampleStyleSheet()
        self.fonts = FontManager.get_fonts()
        self._setup_enhanced_styles()
    
    def _setup_enhanced_styles(self):
        """Setup enhanced paragraph styles"""
        # Document title
        self.styles.add(ParagraphStyle(
            name='EnhancedTitle',
            parent=self.styles['Title'],
            fontSize=24,
            leading=30,
            spaceBefore=0,
            spaceAfter=25,
            textColor=self.color_scheme.primary,
            alignment=TA_CENTER,
            fontName=self.fonts['bold'],
            borderWidth=0,
            borderPadding=0
        ))
        
        # Section headings with enhanced styling
        self.styles.add(ParagraphStyle(
            name='EnhancedSectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            leading=20,
            spaceBefore=20,
            spaceAfter=12,
            textColor=self.color_scheme.primary,
            fontName=self.fonts['bold'],
            borderWidth=0,
            leftIndent=5,
        ))
        
        # Sub-headings
        self.styles.add(ParagraphStyle(
            name='EnhancedSubHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            leading=16,
            spaceBefore=12,
            spaceAfter=8,
            textColor=self.color_scheme.text_primary,
            fontName=self.fonts['bold'],
            leftIndent=15,
            borderWidth=0
        ))
        
        # Body text with better readability
        self.styles.add(ParagraphStyle(
            name='EnhancedBody',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=16,
            spaceBefore=4,
            spaceAfter=8,
            leftIndent=20,
            rightIndent=20,
            alignment=TA_JUSTIFY,
            fontName=self.fonts['default'],
            textColor=self.color_scheme.text_primary,
        ))
        
        # Enhanced list items
        self.styles.add(ParagraphStyle(
            name='EnhancedListItem',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=15,
            spaceAfter=6,
            leftIndent=35,
            bulletIndent=25,
            fontName=self.fonts['default'],
            textColor=self.color_scheme.text_primary,
        ))
        
        # Highlighted box style
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            spaceBefore=10,
            spaceAfter=10,
            leftIndent=25,
            rightIndent=25,
            backColor=self.color_scheme.bg_accent,
            borderColor=self.color_scheme.secondary_light,
            borderWidth=1,
            borderPadding=12,
            fontName=self.fonts['default'],
            textColor=self.color_scheme.text_primary,
            borderRadius=5
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='EnhancedFooter',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=self.color_scheme.text_muted,
            fontName=self.fonts['italic'],
            spaceBefore=5
        ))
    
    def get_style(self, style_name: str) -> ParagraphStyle:
        """Get a specific style"""
        return self.styles.get(style_name, self.styles['Normal'])


class EnhancedContentCleaner:
    """Enhanced content cleaning with better text processing"""
    
    @staticmethod
    def remove_intro_sentences(content: str, language: str = 'en') -> str:
        """Remove common LLM introduction sentences with better patterns"""
        intro_patterns = {
            'tr': [
                r'^.*?[Aa]naliz.*?[Ss]onuç.*?[:]\s*\n',
                r'^.*?[Rr]apor.*?[Hh]azır.*?[:]\s*\n',
                r'^.*?[Dd]eğerlendirme.*?[Ss]onuç.*?[:]\s*\n',
                r'^.*?[İi]şte.*?[Aa]naliz.*?[:]\s*\n',
                r'^.*?[Aa]şağıda.*?[Bb]ulabilirsiniz.*?\n',
            ],
            'en': [
                r'^.*?[Hh]ere.*?[Aa]nalysis.*?[:]\s*\n',
                r'^.*?[Cc]omprehensive.*?[Rr]eport.*?[:]\s*\n',
                r'^.*?[Aa]ssessment.*?[Rr]esults.*?[:]\s*\n',
                r'^.*?[Ff]ollowing.*?[Aa]nalysis.*?[:]\s*\n',
                r'^.*?[Bb]elow.*?[Ff]ind.*?[Rr]eport.*?\n',
            ]
        }
        
        patterns = intro_patterns.get(language, intro_patterns['en'])
        
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
        
        return content.lstrip('\n')
    
    @staticmethod
    def enhance_text_formatting(text: str) -> str:
        """Enhanced text formatting with better visual elements"""
        # Replace emojis with better symbols
        for emoji, replacement in PDFConstants.EMOJI_REPLACEMENTS.items():
            text = text.replace(emoji, replacement)
        
        # Enhance bullet points
        text = re.sub(r'^[•\-\*]\s', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*[-•\*]\s', '  ○ ', text, flags=re.MULTILINE)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()


class EnhancedContentParser:
    """Enhanced content parser with better section detection"""
    
    def __init__(self, style_manager: EnhancedStyleManager, 
                 color_scheme: EnhancedColorScheme = None):
        self.style_manager = style_manager
        self.color_scheme = color_scheme or EnhancedColorScheme()
        self.cleaner = EnhancedContentCleaner()
        
        self.section_keywords = {
            'en': [
                'overview', 'summary', 'analysis', 'skills', 'experience',
                'recommendations', 'conclusion', 'strengths', 'weaknesses',
                'assessment', 'evaluation', 'compatibility', 'qualifications',
                'requirements', 'match', 'score', 'rating'
            ],
            'tr': [
                'özet', 'analiz', 'beceriler', 'deneyim', 'öneriler', 
                'sonuç', 'güçlü yönler', 'zayıf yönler', 'değerlendirme',
                'uyumluluk', 'yeterlilik', 'gereksinimler', 'eşleşme',
                'puan', 'değerlendirme'
            ]
        }
    
    def clean_and_enhance_content(self, content: str, language: str) -> str:
        """Clean and enhance content with better processing"""
        # Remove intro sentences
        content = self.cleaner.remove_intro_sentences(content, language)
        
        # Enhance formatting
        content = self.cleaner.enhance_text_formatting(content)
        
        return content
    
    def detect_score_line(self, line: str) -> Tuple[bool, float]:
        """Enhanced score detection with better parsing"""
        score_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*%\b',
            r'\b(\d+(?:\.\d+)?)\s*/\s*100\b',
            r'\b(\d+(?:\.\d+)?)\s*/\s*10\b',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, line)
            if match:
                score = float(match.group(1))
                # Normalize score to percentage
                if '/10' in line:
                    score *= 10
                elif '/100' in line and score <= 1:
                    score *= 100
                return True, score
        
        return False, 0.0
    
    def is_main_heading(self, line: str, language: str = 'en') -> bool:
        """Enhanced heading detection"""
        line_clean = line.strip()
        
        # Markdown headings
        if line_clean.startswith(('##', '###')) and not line_clean.startswith('####'):
            return True
        
        # Bold headings
        if (line_clean.startswith('**') and line_clean.endswith('**') 
            and len(line_clean) > 4 and ':' not in line_clean):
            return True
        
        # Numbered headings
        if re.match(r'^\d+\.\s+\*\*.*\*\*$', line_clean):
            return True
        
        # Keyword-based detection with better logic
        keywords = self.section_keywords.get(language, self.section_keywords['en'])
        line_lower = line_clean.lower()
        
        if (len(line_clean.split()) <= 6 and 
            any(keyword in line_lower for keyword in keywords) and
            not line_clean.endswith(':')):
            return True
        
        return False
    
    def apply_rich_formatting(self, text: str) -> str:
        """Enhanced rich text formatting"""
        # Bold text
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Italic text
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Highlight scores and percentages
        text = re.sub(
            r'\b(\d+\.?\d*%)\b', 
            r'<font color="#3182ce" size="12"><b>\1</b></font>', 
            text
        )
        
        # Highlight keywords (English and Turkish)
        keywords = ['Strong', 'Excellent', 'Good', 'Poor', 'Weak', 'Outstanding',
                   'Güçlü', 'Mükemmel', 'İyi', 'Zayıf', 'Kötü', 'Olağanüstü']
        for keyword in keywords:
            text = re.sub(
                f'\\b({keyword})\\b', 
                f'<font color="#38a169"><b>\\1</b></font>', 
                text, flags=re.IGNORECASE
            )
        
        return text


class EnhancedPDFBuilder:
    """Enhanced PDF builder with better layout and visuals"""
    
    def __init__(self, style_manager: EnhancedStyleManager, 
                 content_parser: EnhancedContentParser,
                 color_scheme: EnhancedColorScheme = None):
        self.style_manager = style_manager
        self.content_parser = content_parser
        self.color_scheme = color_scheme or EnhancedColorScheme()
        self.fonts = FontManager.get_fonts()
        self.doc_width = None
        self.image_handler = ImageHandler()
    
    def create_enhanced_header(self, metadata: DocumentMetadata) -> List[Any]:
        """Create enhanced header with profile photo"""
        header_elements = []
        
        # Download and process profile photo
        profile_image_path = None
        if metadata.profile_photo_url:
            profile_image_path = self.image_handler.download_and_process_image(
                metadata.profile_photo_url, 
                PDFConstants.PROFILE_PHOTO_SIZE
            )
        
        if profile_image_path:
            # Header with photo
            if metadata.language == Language.TURKISH:
                header_data = [
                    ["", f"Aday: {metadata.candidate_name}", f"Tarih: {metadata.generation_date}"],
                    ["", f"Pozisyon: {metadata.job_title}", ""]
                ]
            else:
                header_data = [
                    ["", f"Candidate: {metadata.candidate_name}", f"Date: {metadata.generation_date}"],
                    ["", f"Position: {metadata.job_title}", ""]
                ]
            
            # Create header table with photo
            header_table = Table(header_data, colWidths=[2*cm, 8*cm, 5*cm])
            
            # Add profile photo
            try:
                img = Image(profile_image_path, width=2.5*cm, height=2.5*cm)  # 1.5*cm'den 2.5*cm'ye
                header_data[0][0] = img
                header_table = Table(header_data, colWidths=[3*cm, 7*cm, 5*cm])  # İlk sütunu 2*cm'den 3*cm'ye
            except:
                header_table = Table(header_data[1:], colWidths=[10*cm, 5*cm])
            
        else:
            # Header without photo
            if metadata.language == Language.TURKISH:
                header_data = [
                    [f"Aday: {metadata.candidate_name}", f"Tarih: {metadata.generation_date}"],
                    [f"Pozisyon: {metadata.job_title}", ""]
                ]
            else:
                header_data = [
                    [f"Candidate: {metadata.candidate_name}", f"Date: {metadata.generation_date}"],
                    [f"Position: {metadata.job_title}", ""]
                ]
            
            header_table = Table(header_data, colWidths=[10*cm, 5*cm])
        
        # Enhanced table styling
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.color_scheme.bg_secondary),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.color_scheme.text_primary),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (-1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, -1), self.fonts['bold']),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('LINEBELOW', (0, -1), (-1, -1), 2, self.color_scheme.secondary),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, self.color_scheme.border_light),
        ]))
        
        header_elements.append(header_table)
        
        return header_elements
    
    def parse_enhanced_content(self, content: str, language: str) -> List[Any]:
        """Parse content with enhanced formatting and layout"""
        # Clean and enhance content
        content = self.content_parser.clean_and_enhance_content(content, language)
        
        story = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        current_section = []
        current_paragraph = []
        in_list = False
        section_count = 0
        
        for i, line in enumerate(lines):
            # Main heading detection
            if self.content_parser.is_main_heading(line, language):
                # Flush current section
                self._flush_enhanced_section(story, current_section, current_paragraph)
                current_section = []
                current_paragraph = []
                in_list = False
                section_count += 1
                
                # Add section divider for sections after first
                if section_count > 1:
                    story.append(Spacer(1, 15))
                    story.append(SectionDivider(
                        self.doc_width, 
                        color_scheme=self.color_scheme,
                        fonts=self.fonts
                    ))
                    story.append(Spacer(1, 10))
                
                # Add enhanced heading
                self._add_enhanced_heading(current_section, line)
                continue
            
            # Score detection with enhanced visualization
            is_score, score_value = self.content_parser.detect_score_line(line)
            if is_score and len(line.split()) <= 12:
                self._flush_paragraph_to_section(current_section, current_paragraph)
                current_paragraph = []
                
                current_section.append(Spacer(1, 8))
                current_section.append(EnhancedScoreBox(
                    line, 
                    width=self.doc_width,
                    height=PDFConstants.SCORE_BOX_HEIGHT,
                    color_scheme=self.color_scheme,
                    fonts=self.fonts
                ))
                current_section.append(Spacer(1, 10))
                continue
            
            # Sub-heading detection
            if self._is_sub_heading(line):
                self._flush_paragraph_to_section(current_section, current_paragraph)
                current_paragraph = []
                
                current_section.append(Spacer(1, 8))
                current_section.append(Paragraph(
                    self.content_parser.apply_rich_formatting(line),
                    self.style_manager.get_style('EnhancedSubHeading')
                ))
                current_section.append(Spacer(1, 5))
                continue
            
            # List item detection
            if self._is_list_item(line):
                self._flush_paragraph_to_section(current_section, current_paragraph)
                current_paragraph = []
                
                if not in_list:
                    in_list = True
                    current_section.append(Spacer(1, 5))
                
                self._add_enhanced_list_item(current_section, line)
                continue
            
            # Regular paragraph content
            if in_list:
                in_list = False
                current_section.append(Spacer(1, 8))
            
            current_paragraph.append(line)
        
        # Flush remaining content
        self._flush_enhanced_section(story, current_section, current_paragraph)
        
        return story
    
    def _flush_enhanced_section(self, story: List[Any], section: List[Any], paragraph: List[str]):
        """Enhanced section flushing with better layout management"""
        if paragraph:
            self._flush_paragraph_to_section(section, paragraph)
        
        if section:
            # Smart section management
            estimated_height = sum(getattr(elem, 'height', 25) for elem in section)
            
            if estimated_height < 600:  # Keep smaller sections together
                story.append(KeepTogether(section))
            else:
                story.extend(section)
    
    def _flush_paragraph_to_section(self, section: List[Any], paragraph_lines: List[str]):
        """Enhanced paragraph formatting"""
        if paragraph_lines:
            text = ' '.join(paragraph_lines)
            formatted_text = self.content_parser.apply_rich_formatting(text)
            
            # Check if this should be a highlight box
            highlight_keywords = ['important', 'note', 'warning', 'key', 'önemli', 'not', 'uyarı', 'anahtar']
            if any(keyword in text.lower() for keyword in highlight_keywords):
                section.append(Paragraph(
                    formatted_text,
                    self.style_manager.get_style('HighlightBox')
                ))
            else:
                section.append(Paragraph(
                    formatted_text,
                    self.style_manager.get_style('EnhancedBody')
                ))
    
    def _add_enhanced_heading(self, section: List[Any], line: str):
        """Add enhanced section heading"""
        section.append(Spacer(1, 15))
        
        cleaned_heading = self._clean_heading(line)
        formatted_heading = self.content_parser.apply_rich_formatting(cleaned_heading)
        
        section.append(Paragraph(
            formatted_heading,
            self.style_manager.get_style('EnhancedSectionHeading')
        ))
        
        section.append(Spacer(1, 10))
    
    def _add_enhanced_list_item(self, section: List[Any], line: str):
        """Add enhanced list item with better bullets"""
        # Clean list marker
        text = re.sub(r'^[•\-\*◆◦▸►]\s*', '', line).strip()
        
        # Enhanced bullet with color
        bullet = '<font color="#3182ce" size="12">•</font>'
        
        formatted_item = f'{bullet} {self.content_parser.apply_rich_formatting(text)}'
        
        section.append(Paragraph(
            formatted_item,
            self.style_manager.get_style('EnhancedListItem')
        ))
    
    def _clean_heading(self, line: str) -> str:
        """Clean heading text with better processing"""
        cleaned = line.replace('##', '').replace('**', '').strip()
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        cleaned = cleaned.rstrip(':')
        return cleaned
    
    def _is_sub_heading(self, line: str) -> bool:
        """Enhanced sub-heading detection"""
        line_clean = line.strip()
        
        if '**' in line_clean and line_clean.endswith(':'):
            return True
        
        if (line_clean.endswith(':') and 3 <= len(line_clean.split()) <= 6 
            and not re.search(r'\d{1,2}:\d{2}', line_clean)):
            return True
        
        return False
    
    def _is_list_item(self, line: str) -> bool:
        """Enhanced list item detection"""
        return line.strip().startswith(('•', '-', '*', '◦', '○'))
    
    def add_enhanced_page_elements(self, canvas, doc):
        """Add enhanced page elements including watermark and page numbers"""
        canvas.saveState()
        
        # Subtle watermark
        canvas.setFillColor(colors.HexColor('#f8fafc'))
        canvas.setFont(self.fonts['bold'], 60)
        canvas.rotate(45)
        canvas.drawCentredString(300, -100, "AI ANALYSIS")
        canvas.rotate(-45)
        
        # Enhanced page number
        canvas.setFont(self.fonts['bold'], 10)
        canvas.setFillColor(self.color_scheme.text_muted)
        page_num = canvas.getPageNumber()
        
        # Page number with styling
        canvas.drawRightString(
            doc.pagesize[0] - 2*cm, 
            1.5*cm, 
            f"Page {page_num}"
        )
        
        # Footer line
        canvas.setStrokeColor(self.color_scheme.border_light)
        canvas.setLineWidth(1)
        canvas.line(2*cm, 2*cm, doc.pagesize[0] - 2*cm, 2*cm)
        
        canvas.restoreState()


class JobCompatibilityPDFGenerator:
    """Enhanced main PDF generator with professional design and Turkish support"""
    
    def __init__(self):
        # Setup fonts first
        self.fonts = FontManager.setup_fonts()
        print(f"Using fonts: {self.fonts}")
        
        self.color_scheme = EnhancedColorScheme()
        self.style_manager = EnhancedStyleManager(self.color_scheme)
        self.content_parser = EnhancedContentParser(self.style_manager, self.color_scheme)
        self.pdf_builder = EnhancedPDFBuilder(
            self.style_manager, 
            self.content_parser, 
            self.color_scheme
        )
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
    # This could be enhanced to track temp files if needed
        pass
    def generate_pdf(self, 
                     report_content: str,
                     job_title: str = "Unknown Position",
                     candidate_name: str = "Candidate",
                     language: str = "en") -> bytes:
        """Generate enhanced PDF report with professional design"""
        
        # Create enhanced metadata
        metadata = DocumentMetadata(
            candidate_name=candidate_name,
            job_title=job_title,
            language=Language(language)
        )
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create enhanced document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=PDFConstants.PAGE_SIZE,
            leftMargin=PDFConstants.LEFT_MARGIN,
            rightMargin=PDFConstants.RIGHT_MARGIN,
            topMargin=PDFConstants.TOP_MARGIN,
            bottomMargin=PDFConstants.BOTTOM_MARGIN,
            title=f"Job Compatibility Analysis - {candidate_name}",
            author="AI Portfolio Assistant - Selman Dedeakayoğulları"
        )
        
        # Calculate usable width
        page_width, _ = PDFConstants.PAGE_SIZE
        self.pdf_builder.doc_width = (
            page_width - PDFConstants.LEFT_MARGIN - PDFConstants.RIGHT_MARGIN
        )
        
        # Build enhanced document content
        story = self._build_enhanced_document(report_content, metadata)
        
        # Build PDF with enhanced page elements
        doc.build(
            story,
            onFirstPage=self.pdf_builder.add_enhanced_page_elements,
            onLaterPages=self.pdf_builder.add_enhanced_page_elements
        )
        self._cleanup_temp_files()
        # Return PDF bytes
        buffer.seek(0)
        return buffer.getvalue()
    
    def _build_enhanced_document(self, report_content: str, 
                                metadata: DocumentMetadata) -> List[Any]:
        """Build enhanced document with professional layout"""
        story = []
        
        # Enhanced title with styling (clean text, no emojis)
        title = self._get_enhanced_title(metadata.language)
        story.append(Paragraph(title, self.style_manager.get_style('EnhancedTitle')))
        story.append(Spacer(1, 20))
        
        # Enhanced header with profile photo
        story.extend(self.pdf_builder.create_enhanced_header(metadata))
        story.append(Spacer(1, 25))
        
        # Main content with enhanced parsing
        story.extend(self.pdf_builder.parse_enhanced_content(
            report_content, 
            metadata.language.value
        ))
        
        # Enhanced footer
        story.extend(self._build_enhanced_footer(metadata.language))
        
        return story
    
    def _get_enhanced_title(self, language: Language) -> str:
        """Get enhanced document title (clean, no emojis)"""
        if language == Language.ENGLISH:
            return "Job Compatibility Analysis Report"
        else:
            return "İş Uyumluluk Analiz Raporu"
    
    def _build_enhanced_footer(self, language: Language) -> List[Any]:
        """Build enhanced document footer"""
        footer_elements = []
        
        # Spacer before footer
        footer_elements.append(Spacer(1, 40))
        
        # Enhanced footer separator
        footer_elements.append(SectionDivider(
            self.pdf_builder.doc_width, 
            color_scheme=self.color_scheme,
            fonts=self.pdf_builder.fonts
        ))
        footer_elements.append(Spacer(1, 15))
        
        # Footer content
        if language == Language.ENGLISH:
            footer_text = (
                "<b>AI Portfolio Assistant</b><br/>"
                "Created by <i>Selman Dedeakayoğulları</i><br/>"
                "Professional AI-Powered Career Analysis"
            )
        else:
            footer_text = (
                "<b>AI Portföy Asistanı</b><br/>"
                "<i>Selman Dedeakayoğulları</i> tarafından oluşturuldu<br/>"
                "Profesyonel AI Destekli Kariyer Analizi"
            )
        
        footer_elements.append(Paragraph(
            footer_text,
            self.style_manager.get_style('EnhancedFooter')
        ))
        
        return footer_elements


# Enhanced utility function
def generate_enhanced_compatibility_pdf(report_content: str,
                                      job_title: str = "Unknown Position",
                                      candidate_name: str = "Candidate",
                                      language: str = "en") -> bytes:
    """
    Generate enhanced PDF report with professional design and Turkish character support.
    
    Args:
        report_content: The report text content
        job_title: Title of the job position
        candidate_name: Name of the candidate
        language: Language code ('en' or 'tr')
        
    Returns:
        Enhanced PDF file as bytes
    """
    try:
        generator = JobCompatibilityPDFGenerator()
        return generator.generate_pdf(
            report_content=report_content,
            job_title=job_title,
            candidate_name=candidate_name,
            language=language
        )
    except Exception as e:
        print(f"Error generating PDF: {e}")
        # Return a simple fallback PDF if enhanced version fails
        return _generate_simple_fallback_pdf(report_content, job_title, candidate_name, language)


def _generate_simple_fallback_pdf(report_content: str, job_title: str, candidate_name: str, language: str = "en") -> bytes:
    """Simple fallback PDF generator using basic ReportLab features with Turkish support"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Try to use a font that supports Turkish characters
    try:
        fonts = FontManager.setup_fonts()
        default_font = fonts['default']
    except:
        default_font = 'Times-Roman'  # Fallback to Times which has some Turkish support
    
    # Simple title
    if language == "tr":
        title_text = "İş Uyumluluk Analiz Raporu"
        candidate_text = f"<b>Aday:</b> {candidate_name}"
        position_text = f"<b>Pozisyon:</b> {job_title}"
    else:
        title_text = "Job Compatibility Analysis Report"
        candidate_text = f"<b>Candidate:</b> {candidate_name}"
        position_text = f"<b>Position:</b> {job_title}"
    
    story.append(Paragraph(title_text, styles['Title']))
    story.append(Spacer(1, 20))
    
    # Simple header
    story.append(Paragraph(candidate_text, styles['Normal']))
    story.append(Paragraph(position_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Simple content
    for line in report_content.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 6))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Backward compatibility
def generate_compatibility_pdf(report_content: str,
                              job_title: str = "Unknown Position", 
                              candidate_name: str = "Candidate",
                              language: str = "en") -> bytes:
    """Backward compatibility wrapper with Turkish support"""
    return generate_enhanced_compatibility_pdf(
        report_content, job_title, candidate_name, language
    )