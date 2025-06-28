import io
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import requests
from urllib.parse import urlparse
import traceback

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

        # If fonts are not found locally, raise an exception to fallback
        if not all(found_fonts.values()):
             raise FileNotFoundError("DejaVu fonts not found in the 'fonts/' directory.")

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
    company_name: str
    language: Language
    contact:str="selmandedeakayogullari@gmail.com"
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
        if not url:
            return None
        try:
            # Download image
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()

            # Open with PIL
            img = PILImage.open(response.raw)

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

        except requests.exceptions.RequestException as e:
            print(f"Error downloading profile image: {e}")
            return None
        except Exception as e:
            print(f"Error processing profile image: {e}")
            traceback.print_exc()
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

        # Section headings with CONTROLLED spacing
        self.styles.add(ParagraphStyle(
            name='EnhancedSectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            leading=20,
            spaceBefore=10,  # DÜZELTME: Küçültüldü (20'den 10'a)
            spaceAfter=8,    # DÜZELTME: Küçültüldü (12'den 8'e)
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
            spaceBefore=8,   # DÜZELTME: Küçültüldü (12'den 8'e)
            spaceAfter=6,    # DÜZELTME: Küçültüldü (8'den 6'ya)
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
            spaceBefore=3,   # DÜZELTME: Küçültüldü (4'ten 3'e)
            spaceAfter=6,    # DÜZELTME: Küçültüldü (8'den 6'ya)
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
            spaceAfter=4,    # DÜZELTME: Küçültüldü (6'dan 4'e)
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
            spaceBefore=8,   # DÜZELTME: Küçültüldü (10'dan 8'e)
            spaceAfter=8,    # DÜZELTME: Küçültüldü (10'dan 8'e)
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
    def remove_llm_generated_header(content: str) -> str:
        """Removes only problematic headers, keeps the good ones."""
        # Sadece gerçekten kötü başlıkları temizle
        bad_header_patterns = [
            # Sadece "Job Compatibility Report: ..." tarzı başlıkları temizle
            r'^\s*#+\s*Job Compatibility Report:.*?\n',
            r'^\s*Job Compatibility Report:.*?\n',
            # Candidate/Company info içeren başlıkları temizle
            r'^\s*#+\s*.*?Candidate:.*?\n.*?Company:.*?\n.*?Position:.*?\n',
            r'^\s*.*?Candidate:.*?\n.*?Company:.*?\n.*?Position:.*?\n',
        ]
        
        cleaned_content = content
        
        # Sadece kötü başlıkları temizle
        for pattern in bad_header_patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        return cleaned_content.strip()

    @staticmethod
    def clean_malformed_list_items(content: str) -> str:
        """Clean malformed or empty list items"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty bullets or bullets with just punctuation
            if re.match(r'^\s*[•\-\*◆◦▸►○]\s*[-\s]*$', line):
                continue
                
            # Skip bullets with minimal content (just punctuation)
            if re.match(r'^\s*[•\-\*◆◦▸►○]\s*[^\w]*$', line):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    @staticmethod
    def remove_intro_sentences(content: str, language: str = 'en') -> str:
        """Remove only intro sentences, not section headers."""
        intro_patterns = {
            'tr': [
                r'^.*?[Bb]u rapor.*?\n',
                r'^.*?[Aa]şağıdaki analiz.*?\n',
                r'^.*?[Aa]naliz sonuçları.*?\n',
            ],
            'en': [
                r'^.*?[Tt]his report.*?\n',
                r'^.*?[Bb]ased on the analysis.*?\n',
                r'^.*?[Ff]ollowing analysis.*?\n',
            ]
        }

        patterns = intro_patterns.get(language, intro_patterns['en'])
        
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        return content.strip()

    @staticmethod
    def enhance_text_formatting(text: str) -> str:
        """Enhanced text formatting with better visual elements"""
        # Replace emojis with better symbols
        for emoji, replacement in PDFConstants.EMOJI_REPLACEMENTS.items():
            text = text.replace(emoji, replacement)

        # Clean up malformed bullets first
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty or malformed bullet lines
            if re.match(r'^\s*[•\-\*◦○]\s*[-\s]*$', line):
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)

        # Standardize bullet points but preserve indentation for sub-lists
        # First handle sub-bullets (indented ones)
        text = re.sub(r'^\s{2,}[•\-\*]\s', '  ○ ', text, flags=re.MULTILINE)
        # Then handle main bullets
        text = re.sub(r'^\s*[•\-\*]\s', '• ', text, flags=re.MULTILINE)

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
                'recommendations', 'recommendation', 'conclusion', 'strengths', 'weaknesses',
                'assessment', 'evaluation', 'compatibility', 'qualifications',
                'requirements', 'match', 'score', 'rating', 'education', 'project',
                'development', 'areas', 'technical', 'executive'  # Added these
            ],
            'tr': [
                'özet', 'analiz', 'beceriler', 'deneyim', 'öneriler', 'öneri',
                'sonuç', 'güçlü yönler', 'zayıf yönler', 'değerlendirme',
                'uyumluluk', 'yeterlilik', 'gereksinimler', 'eşleşme',
                'puan', 'değerlendirme', 'eğitim', 'proje', 'gelişim', 'alanları',
                'teknik', 'yönetici'  # Added these
            ]
        }

    def clean_and_enhance_content(self, content: str, language: str) -> str:
        """Clean and enhance content - keep it simple."""
        # Sadece gerçekten kötü başlıkları temizle
        content = self.cleaner.remove_llm_generated_header(content)
        
        # Sadece intro cümlelerini temizle
        content = self.cleaner.remove_intro_sentences(content, language)
        
        # Malformed list items'ları temizle
        content = self.cleaner.clean_malformed_list_items(content)
        
        # Text formatting'i düzelt
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
        """Enhanced heading detection - specifically for our format."""
        line_clean = line.strip()

        # Ana format: "## 1. Executive Summary" veya "## 1. Genel Değerlendirme"
        if re.match(r'^#{2,3}\s+\d+\.\s+', line_clean):
            return True

        # Alternatif format: "1. **Executive Summary**"
        if re.match(r'^\d+\.\s+\*\*.*\*\*\s*$', line_clean):
            return True
        
        # Alternatif format: "**1. Executive Summary**"
        if re.match(r'^\*\*\d+\.\s+.*\*\*\s*$', line_clean):
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
        """Create enhanced header with profile photo and company name"""
        header_elements = []

        # Download and process profile photo
        profile_image_path = self.image_handler.download_and_process_image(
            metadata.profile_photo_url,
            PDFConstants.PROFILE_PHOTO_SIZE
        )

        # Prepare header data with company name
        if metadata.language == Language.TURKISH:
            header_text_data = [
                f"Aday: <b>{metadata.candidate_name}</b>",
                f"Şirket: <b>{metadata.company_name}</b>",
                f"Pozisyon: {metadata.job_title}",
                f"Tarih: {metadata.generation_date}",
                f"İletişim: {metadata.contact}"
            ]
        else:
            header_text_data = [
                f"Candidate: <b>{metadata.candidate_name}</b>",
                f"Company: <b>{metadata.company_name}</b>",
                f"Position: {metadata.job_title}",
                f"Date: {metadata.generation_date}",
                f"Contact: {metadata.contact}"
            ]

        # Create paragraphs for text part of the header
        header_text_style = ParagraphStyle(
            name='HeaderText',
            fontName=self.fonts['default'],
            fontSize=10,
            textColor=self.color_scheme.text_primary,
            leading=14
        )
        header_paragraphs = [Paragraph(text, header_text_style) for text in header_text_data]

        header_content = [header_paragraphs]
        col_widths = [self.doc_width]

        if profile_image_path:
            try:
                img = Image(profile_image_path, width=2.5*cm, height=2.5*cm)
                img.hAlign = 'CENTER'
                header_content = [[img, header_paragraphs]]
                col_widths = [3.5*cm, self.doc_width - 3.5*cm]
            except Exception as e:
                print(f"Could not load image for PDF header: {e}")
                # Fallback to text-only header
                header_content = [header_paragraphs]


        header_table = Table(header_content, colWidths=col_widths)

        # Enhanced table styling
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.color_scheme.bg_secondary),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, -1), (-1, -1), 2, self.color_scheme.secondary),
        ]))

        header_elements.append(header_table)

        return header_elements

    def parse_enhanced_content(self, content: str, language: str) -> List[Any]:
        """Parse content with enhanced formatting and layout, ensuring proper list and section handling."""
        # Clean and enhance content
        content = self.content_parser.clean_and_enhance_content(content, language)
        story = []
        lines = content.split('\n')

        section_count = 0
        current_paragraph_lines = []

        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Check if this is a main heading
            is_heading = self.content_parser.is_main_heading(stripped_line, language)
            is_list = self._is_list_item(stripped_line)
            
            # Flush paragraph if we encounter a heading or list item
            if is_heading or is_list:
                self._flush_paragraph_to_story(story, current_paragraph_lines)
                current_paragraph_lines = []

            if is_heading:
                # DÜZELTME: Sayfa geçişi mantığını değiştiriyoruz
                # Sadece çok uzun raporlar için sayfa geçişi yap
                if section_count > 0:
                    # 5. bölümden önce büyük boşluk bırakma!
                    # Sadece normal spacing ekle
                    story.append(Spacer(1, 15))  # Küçük boşluk
                
                section_count += 1
                self._add_enhanced_heading(story, stripped_line)
                
            elif is_list:
                # Additional validation to ensure the list item has meaningful content
                content_after_marker = re.sub(r'^[•\-\*◦○]\s*', '', stripped_line.strip())
                if content_after_marker and re.search(r'\w', content_after_marker):
                    self._add_enhanced_list_item(story, stripped_line)
            else:
                current_paragraph_lines.append(stripped_line)

        # Flush any remaining paragraph content
        self._flush_paragraph_to_story(story, current_paragraph_lines)

        return story

    def _flush_paragraph_to_story(self, story: List[Any], paragraph_lines: List[str]):
        """Flush a collected paragraph to the main story with proper styling."""
        if not paragraph_lines:
            return

        full_paragraph_text = ' '.join(paragraph_lines)
        if full_paragraph_text:
             self._add_enhanced_paragraph(story, full_paragraph_text)

    def _add_enhanced_paragraph(self, story: List[Any], text: str):
        """Adds a standard enhanced paragraph to the story."""
        formatted_text = self.content_parser.apply_rich_formatting(text)
        # Check if this should be a highlight box
        highlight_keywords = ['important', 'note', 'warning', 'key', 'önemli', 'not', 'uyarı', 'anahtar']
        if any(keyword in text.lower() for keyword in highlight_keywords):
            story.append(Paragraph(
                formatted_text,
                self.style_manager.get_style('HighlightBox')
            ))
        else:
            story.append(Paragraph(
                formatted_text,
                self.style_manager.get_style('EnhancedBody')
            ))

    def _add_enhanced_heading(self, story: List[Any], line: str):
        """Add enhanced section heading with consistent spacing"""
        # DÜZELTME: Başlık öncesi boşluğu sabitliyoruz
        # story.append(Spacer(1, 15)) # Bu satırı kaldırıyoruz çünkü üstte ekliyoruz

        cleaned_heading = self._clean_heading(line)
        formatted_heading = self.content_parser.apply_rich_formatting(cleaned_heading)

        story.append(Paragraph(
            formatted_heading,
            self.style_manager.get_style('EnhancedSectionHeading')
        ))

        # Başlık altına küçük bir çizgi ve boşluk
        story.append(HRFlowable(width="100%", thickness=1, color=self.color_scheme.border_light, spaceAfter=8))


    def _add_enhanced_list_item(self, story: List[Any], line: str):
        """Add enhanced list item with better bullets"""
        # Determine if this is a sub-bullet based on indentation
        is_sub_bullet = line.startswith('  ') and any(line.lstrip().startswith(marker) for marker in ['○', '◦', '-', '*'])
        
        # Clean list marker
        text = re.sub(r'^\s*[•\-\*◆◦▸►○]\s*', '', line).strip()

        # Enhanced bullet with color
        if is_sub_bullet:
            bullet = '<font color="#63b3ed" size="12">○</font>'
            style = self.style_manager.get_style('EnhancedListItem')
            style.leftIndent = 50  # Indent sub-bullets further
        else:
            bullet = '<font color="#3182ce" size="12">•</font>'
            style = self.style_manager.get_style('EnhancedListItem')
            style.leftIndent = 35  # Reset indent for main bullets

        formatted_item = f'{bullet} {self.content_parser.apply_rich_formatting(text)}'

        story.append(Paragraph(formatted_item, style))

    def _clean_heading(self, line: str) -> str:
        """Clean heading text with better processing for numbered sections"""
        cleaned = line.strip()
        
        # Remove markdown headers (## or ###)
        cleaned = re.sub(r'^#{2,3}\s*', '', cleaned)
        
        # Remove bold markers but preserve the text
        cleaned = cleaned.replace('**', '').strip()
        
        # For numbered sections, keep the number and title
        # Don't remove the "1. " part for numbered sections
        numbered_match = re.match(r'^(\d+\.\s+)(.+)', cleaned)
        if numbered_match:
            number_part = numbered_match.group(1)
            title_part = numbered_match.group(2).rstrip(':')
            cleaned = f"{number_part}{title_part}"
        else:
            # For non-numbered headings, remove trailing colons
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
        """Enhanced list item detection with validation"""
        stripped = line.strip()
        
        # Must start with a bullet marker
        if not stripped.startswith(('•', '-', '*', '◦', '○')):
            return False
        
        # Must have actual content after the marker (not just punctuation/whitespace)
        content_after_marker = re.sub(r'^[•\-\*◦○]\s*', '', stripped)
        
        # Check if there's meaningful content (at least one word character)
        if not re.search(r'\w', content_after_marker):
            return False
            
        return True

    def add_enhanced_page_elements(self, canvas, doc):
        """Add enhanced page elements including watermark and page numbers"""
        canvas.saveState()

        # Subtle watermark
        canvas.setFillColor(colors.HexColor('#f0f4f8'))
        canvas.setFont(self.fonts['bold'], 60)
        canvas.rotate(45)
        canvas.drawCentredString(300, -100, "AI ANALYSIS")
        canvas.rotate(-45)

        # Enhanced page number
        canvas.setFont(self.fonts['bold'], 10)
        canvas.setFillColor(self.color_scheme.text_muted)
        page_num_text = f"Page {canvas.getPageNumber()}"

        # Draw on the right
        canvas.drawRightString(doc.pagesize[0] - 2*cm, 1.5*cm, page_num_text)

        # Draw a line for the footer
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
                     language: str = "en",
                     company_name: str = "Unknown Company") -> bytes:
        """Generate enhanced PDF report with professional design"""

        # Create enhanced metadata
        metadata = DocumentMetadata(
            candidate_name=candidate_name,
            job_title=job_title,
            company_name=company_name,
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

        # Enhanced header with profile photo and company name
        story.extend(self.pdf_builder.create_enhanced_header(metadata))
        story.append(Spacer(1, 15)) # Reduced spacer after header

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
                                      language: str = "en",
                                      company_name: str = "Unknown Company") -> bytes:
    """
    Generate enhanced PDF report with professional design and Turkish character support.
    
    Args:
        report_content: The report text content
        job_title: Title of the job position
        candidate_name: Name of the candidate
        language: Language code ('en' or 'tr')
        company_name: Name of the company
        
    Returns:
        Enhanced PDF file as bytes
    """
    try:
        generator = JobCompatibilityPDFGenerator()
        return generator.generate_pdf(
            report_content=report_content,
            job_title=job_title,
            candidate_name=candidate_name,
            language=language,
            company_name=company_name
        )
    except Exception as e:
        print(f"Error generating PDF: {e}")
        traceback.print_exc()
        # Return a simple fallback PDF if enhanced version fails
        return _generate_simple_fallback_pdf(report_content, job_title, candidate_name, company_name, language)


def _generate_simple_fallback_pdf(report_content: str, job_title: str, candidate_name: str, company_name: str, language: str = "en") -> bytes:
    """Simple fallback PDF generator using basic ReportLab features with Turkish support"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Try to use a font that supports Turkish characters
    try:
        fonts = FontManager.get_fonts()
        styles['Normal'].fontName = fonts['default']
        styles['Title'].fontName = fonts['bold']
    except:
        print("Fallback font not found, using Times-Roman.")
        styles['Normal'].fontName = 'Times-Roman'
        styles['Title'].fontName = 'Times-Bold'


    # Simple title
    if language == "tr":
        title_text = "İş Uyumluluk Analiz Raporu"
        candidate_text = f"<b>Aday:</b> {candidate_name}"
        company_text = f"<b>Şirket:</b> {company_name}"
        position_text = f"<b>Pozisyon:</b> {job_title}"
    else:
        title_text = "Job Compatibility Analysis Report"
        candidate_text = f"<b>Candidate:</b> {candidate_name}"
        company_text = f"<b>Company:</b> {company_name}"
        position_text = f"<b>Position:</b> {job_title}"

    story.append(Paragraph(title_text, styles['Title']))
    story.append(Spacer(1, 20))

    # Simple header
    story.append(Paragraph(candidate_text, styles['Normal']))
    story.append(Paragraph(company_text, styles['Normal']))
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
                              language: str = "en",
                              company_name: str = "Unknown Company") -> bytes:
    """Backward compatibility wrapper with Turkish support"""
    return generate_enhanced_compatibility_pdf(
        report_content, job_title, candidate_name, language, company_name
    )