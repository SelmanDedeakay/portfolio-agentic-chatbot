import io
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, 
    HRFlowable, KeepTogether, Image, Flowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF


class PDFConstants:
    """Constants for PDF generation"""
    # Page settings
    PAGE_SIZE = A4
    LEFT_MARGIN = 2 * cm
    RIGHT_MARGIN = 2 * cm
    TOP_MARGIN = 1.5 * cm
    BOTTOM_MARGIN = 2 * cm
    
    # Font settings
    DEFAULT_FONT = 'DejaVuSans'
    BOLD_FONT = 'DejaVuSans-Bold'
    
    # Score thresholds
    HIGH_SCORE_THRESHOLD = 80
    MEDIUM_SCORE_THRESHOLD = 60
    
    # Visual elements
    SCORE_BOX_HEIGHT = 30
    PROGRESS_BAR_HEIGHT = 6
    DECORATOR_HEIGHT = 10
    
    # Watermark
    WATERMARK_TEXT = "AI ANALYSIS"
    WATERMARK_FONT_SIZE = 60
    WATERMARK_ROTATION = 45


class Language(Enum):
    """Supported languages for PDF generation"""
    ENGLISH = "en"
    TURKISH = "tr"


@dataclass
class ColorScheme:
    """Color scheme for PDF styling"""
    primary: colors.Color = colors.HexColor('#2C3E50')
    secondary: colors.Color = colors.HexColor('#3498DB')
    accent: colors.Color = colors.HexColor('#E74C3C')
    success: colors.Color = colors.HexColor('#27AE60')
    warning: colors.Color = colors.HexColor('#F39C12')
    text: colors.Color = colors.HexColor('#34495E')
    light_bg: colors.Color = colors.HexColor('#ECF0F1')
    white: colors.Color = colors.white
    muted: colors.Color = colors.HexColor('#7F8C8D')
    
    def get_score_colors(self, score: float) -> Tuple[colors.Color, colors.Color]:
        """Get colors based on score value"""
        if score >= PDFConstants.HIGH_SCORE_THRESHOLD:
            return self.success, colors.HexColor('#E8F8F5')
        elif score >= PDFConstants.MEDIUM_SCORE_THRESHOLD:
            return self.warning, colors.HexColor('#FEF5E7')
        else:
            return self.accent, colors.HexColor('#FADBD8')


@dataclass
class DocumentMetadata:
    """Metadata for PDF document"""
    candidate_name: str
    job_title: str
    language: Language
    generation_date: str = None
    
    def __post_init__(self):
        if not self.generation_date:
            self.generation_date = datetime.now().strftime("%d/%m/%Y")


class GradientBackground(Flowable):
    """Custom flowable for gradient backgrounds"""
    
    def __init__(self, width: float, height: float, 
                 color1: Tuple[float, float, float], 
                 color2: Tuple[float, float, float], 
                 steps: int = 20):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color1 = color1
        self.color2 = color2
        self.steps = steps
    
    def draw(self):
        """Draw gradient background"""
        step_height = self.height / self.steps
        
        for i in range(self.steps):
            ratio = i / float(self.steps)
            
            # Interpolate colors
            r = self._interpolate(self.color1[0], self.color2[0], ratio)
            g = self._interpolate(self.color1[1], self.color2[1], ratio)
            b = self._interpolate(self.color1[2], self.color2[2], ratio)
            
            self.canv.setFillColorRGB(r, g, b)
            y = step_height * i
            self.canv.rect(0, y, self.width, step_height, fill=1, stroke=0)
    
    @staticmethod
    def _interpolate(start: float, end: float, ratio: float) -> float:
        """Linear interpolation between two values"""
        return start * (1 - ratio) + end * ratio


class ScoreBox(Flowable):
    """Enhanced score display box with visual indicators"""
    
    def __init__(self, score_text: str, width: float, 
                 height: float = PDFConstants.SCORE_BOX_HEIGHT,
                 color_scheme: ColorScheme = None):
        Flowable.__init__(self)
        self.score_text = score_text
        self.width = width
        self.height = height
        self.color_scheme = color_scheme or ColorScheme()
        self.score_value = self._extract_score()
    
    def _extract_score(self) -> float:
        """Extract numerical score from text"""
        percentages = re.findall(r'(\d+\.?\d*)%', self.score_text)
        return float(percentages[0]) if percentages else 0.0
    
    def draw(self):
        """Draw the score box with progress bar"""
        # Get colors based on score
        main_color, bg_color = self.color_scheme.get_score_colors(self.score_value)
        
        # Draw rounded rectangle background
        self._draw_background(bg_color, main_color)
        
        # Draw progress bar
        self._draw_progress_bar(main_color)
        
        # Draw text
        self._draw_text()
    
    def _draw_background(self, bg_color: colors.Color, border_color: colors.Color):
        """Draw the background with border"""
        self.canv.setFillColor(bg_color)
        self.canv.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=1)
        
        self.canv.setStrokeColor(border_color)
        self.canv.setLineWidth(1.5)
        self.canv.roundRect(0, 0, self.width, self.height, 5, fill=0, stroke=1)
    
    def _draw_progress_bar(self, fill_color: colors.Color):
        """Draw the progress bar"""
        bar_margin = 20
        bar_width = self.width - (2 * bar_margin)
        bar_y = 10
        
        # Background bar
        self.canv.setFillColor(colors.HexColor('#ECEFF1'))
        self.canv.roundRect(
            bar_margin, bar_y, bar_width, 
            PDFConstants.PROGRESS_BAR_HEIGHT, 3, fill=1, stroke=0
        )
        
        # Filled bar
        progress_width = min(bar_width, bar_width * (self.score_value / 100))
        if progress_width > 0:
            self.canv.setFillColor(fill_color)
            self.canv.roundRect(
                bar_margin, bar_y, progress_width, 
                PDFConstants.PROGRESS_BAR_HEIGHT, 3, fill=1, stroke=0
            )
    
    def _draw_text(self):
        """Draw the score text"""
        self.canv.setFont(PDFConstants.BOLD_FONT, 12)
        self.canv.setFillColor(self.color_scheme.text)
        self.canv.drawCentredString(self.width / 2, self.height - 18, self.score_text)


class HeadingDecorator(Flowable):
    """Decorative element for section headings"""
    
    def __init__(self, width: float = 100, height: float = PDFConstants.DECORATOR_HEIGHT,
                 color_scheme: ColorScheme = None):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color_scheme = color_scheme or ColorScheme()
    
    def draw(self):
        """Draw decorative rectangles"""
        rectangles = [
            (0, 3, 30, 4, self.color_scheme.secondary),
            (35, 3, 15, 4, self.color_scheme.accent),
            (55, 3, 10, 4, self.color_scheme.warning)
        ]
        
        for x, y, w, h, color in rectangles:
            self.canv.setFillColor(color)
            self.canv.rect(x, y, w, h, fill=1, stroke=0)


class StyleManager:
    """Manage PDF styles with centralized configuration"""
    
    def __init__(self, color_scheme: ColorScheme = None):
        self.color_scheme = color_scheme or ColorScheme()
        self.styles = getSampleStyleSheet()
        self._register_fonts()
        self._setup_styles()
    
    def _register_fonts(self):
        """Register custom fonts"""
        try:
            pdfmetrics.registerFont(TTFont(PDFConstants.DEFAULT_FONT, '../fonts/DejaVuSans.ttf'))
            pdfmetrics.registerFont(TTFont(PDFConstants.BOLD_FONT, '../fonts/DejaVuSans-Bold.ttf'))
        except Exception:
            # Fallback to default fonts if custom fonts not available
            pass
    
    def _setup_styles(self):
        """Setup all paragraph styles"""
        style_configs = [
            ('CustomTitle', {
                'parent': self.styles['Title'],
                'fontSize': 28,
                'leading': 36,
                'spaceBefore': 20,
                'spaceAfter': 30,
                'textColor': self.color_scheme.primary,
                'alignment': TA_CENTER,
                'fontName': PDFConstants.BOLD_FONT
            }),
            ('SectionHeading', {
                'parent': self.styles['Heading2'],
                'fontSize': 18,
                'leading': 24,
                'spaceBefore': 25,
                'spaceAfter': 15,
                'textColor': self.color_scheme.secondary,
                'fontName': PDFConstants.BOLD_FONT,
                'borderPadding': (0, 0, 0, 5),
                'borderColor': self.color_scheme.secondary
            }),
            ('SubHeading', {
                'parent': self.styles['Heading3'],
                'fontSize': 14,
                'leading': 18,
                'spaceBefore': 15,
                'spaceAfter': 10,
                'textColor': self.color_scheme.text,
                'fontName': PDFConstants.BOLD_FONT,
                'leftIndent': 15
            }),
            ('CustomBody', {
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'leading': 18,
                'spaceBefore': 6,
                'spaceAfter': 12,
                'leftIndent': 25,
                'rightIndent': 25,
                'alignment': TA_JUSTIFY,
                'fontName': PDFConstants.DEFAULT_FONT,
                'textColor': self.color_scheme.text,
                'wordWrap': 'CJK'
            }),
            ('ListItem', {
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'leading': 16,
                'spaceAfter': 8,
                'leftIndent': 40,
                'bulletIndent': 25,
                'fontName': PDFConstants.DEFAULT_FONT,
                'textColor': self.color_scheme.text,
                'bulletFontName': 'Symbol',
                'bulletFontSize': 8,
                'bulletColor': self.color_scheme.secondary
            }),
            ('Footer', {
                'parent': self.styles['Normal'],
                'fontSize': 9,
                'leading': 12,
                'alignment': TA_CENTER,
                'textColor': self.color_scheme.muted,
                'fontName': PDFConstants.DEFAULT_FONT
            }),
            ('HighlightBox', {
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'leading': 18,
                'spaceBefore': 15,
                'spaceAfter': 15,
                'leftIndent': 30,
                'rightIndent': 30,
                'backColor': self.color_scheme.light_bg,
                'borderColor': self.color_scheme.secondary,
                'borderWidth': 1,
                'borderPadding': 15,
                'borderRadius': 5,
                'fontName': PDFConstants.DEFAULT_FONT,
                'textColor': self.color_scheme.text
            })
        ]
        
        for name, config in style_configs:
            self.styles.add(ParagraphStyle(name=name, **config))
    
    def get_style(self, style_name: str) -> ParagraphStyle:
        """Get a specific style"""
        return self.styles.get(style_name, self.styles['Normal'])


class ContentParser:
    """Parse and format report content"""
    
    def __init__(self, style_manager: StyleManager, color_scheme: ColorScheme = None):
        self.style_manager = style_manager
        self.color_scheme = color_scheme or ColorScheme()
        
        # Section keywords for detection
        self.section_keywords = {
            'en': [
                'overview', 'summary', 'analysis', 'skills', 'experience',
                'recommendations', 'conclusion', 'strengths', 'weaknesses'
            ],
            'tr': [
                'özet', 'analiz', 'beceriler', 'deneyim', 'öneriler', 
                'sonuç', 'güçlü yönler', 'zayıf yönler'
            ]
        }
    
    def contains_score(self, line: str) -> bool:
        """Check if line contains a percentage score"""
        return bool(re.search(r'\b\d+(\.\d+)?%\b', line))
    
    def is_main_heading(self, line: str, language: str = 'en') -> bool:
        """Detect main section headings"""
        line_clean = line.strip()
        
        # Markdown heading
        if line_clean.startswith('##'):
            return True
        
        # Bold heading
        if line_clean.startswith('**') and line_clean.endswith('**'):
            return True
        
        # Numbered heading
        if re.match(r'^\d+\.\s+\*\*.*\*\*$', line_clean):
            return True
        
        # Keyword-based detection
        keywords = self.section_keywords.get(language, self.section_keywords['en'])
        line_lower = line_clean.lower()
        
        return any(
            keyword in line_lower for keyword in keywords
        ) and len(line_clean.split()) <= 4
    
    def is_sub_heading(self, line: str) -> bool:
        """Detect sub-headings"""
        line_clean = line.strip()
        
        # Ends with colon and short
        if line_clean.endswith(':') and len(line_clean.split()) <= 6:
            # Not a time format
            if not re.search(r'\d{1,2}:\d{2}', line_clean):
                return True
        
        return False
    
    def clean_heading(self, line: str) -> str:
        """Clean heading text"""
        # Remove markdown and formatting
        cleaned = line.replace('##', '').replace('**', '').strip()
        
        # Remove numbering
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        
        # Capitalize first letter
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def apply_rich_formatting(self, text: str) -> str:
        """Apply rich text formatting"""
        # Bold text
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Italic text
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Code formatting
        text = re.sub(
            r'`(.*?)`', 
            r'<font face="Courier" color="#2C3E50" backColor="#F8F9FA">\1</font>', 
            text
        )
        
        # Highlight percentages
        text = re.sub(
            r'\b(\d+\.?\d*%)\b', 
            r'<font color="#E74C3C"><b>\1</b></font>', 
            text
        )
        
        # Highlight years
        text = re.sub(
            r'\b(20\d{2})\b', 
            r'<font color="#3498DB">\1</font>', 
            text
        )
        
        return text
    
    def is_list_item(self, line: str) -> bool:
        """Check if line is a list item"""
        return line.strip().startswith(('•', '-', '*', '►', '▸'))
    
    def is_key_value_pair(self, line: str) -> bool:
        """Check if line is a key-value pair"""
        if ':' not in line:
            return False
        
        parts = line.split(':', 1)
        if len(parts) != 2:
            return False
        
        key_part = parts[0].strip()
        return len(key_part.split()) <= 3


class PDFBuilder:
    """Build PDF document with structured content"""
    
    def __init__(self, style_manager: StyleManager, 
                 content_parser: ContentParser,
                 color_scheme: ColorScheme = None):
        self.style_manager = style_manager
        self.content_parser = content_parser
        self.color_scheme = color_scheme or ColorScheme()
        self.doc_width = None
    
    def create_header_section(self, metadata: DocumentMetadata) -> Table:
        """Create document header section"""
        # Prepare header data based on language
        if metadata.language == Language.TURKISH:
            header_data = [
                ["Aday:", metadata.candidate_name],
                ["Pozisyon:", metadata.job_title],
                ["Tarih:", metadata.generation_date]
            ]
        else:
            header_data = [
                ["Candidate:", metadata.candidate_name],
                ["Position:", metadata.job_title],
                ["Date:", metadata.generation_date]
            ]
        
        # Create header table
        header_table = Table(header_data, colWidths=[4*cm, 11*cm])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.color_scheme.primary),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.color_scheme.white),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), PDFConstants.DEFAULT_FONT),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # Wrap in outer table for styling
        wrapper = Table([[header_table]], colWidths=[16*cm], rowHeights=[3*cm])
        wrapper.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.color_scheme.primary),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        
        return wrapper
    
    def parse_content(self, content: str, language: str) -> List[Any]:
        """Parse report content into flowables"""
        story = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        current_paragraph = []
        in_list = False
        
        for line in lines:
            # Score box
            if self.content_parser.contains_score(line):
                self._flush_paragraph(story, current_paragraph)
                current_paragraph = []
                
                story.append(Spacer(1, 10))
                story.append(ScoreBox(
                    line, 
                    width=self.doc_width, 
                    height=PDFConstants.SCORE_BOX_HEIGHT,
                    color_scheme=self.color_scheme
                ))
                story.append(Spacer(1, 15))
                continue
            
            # Main heading
            if self.content_parser.is_main_heading(line, language):
                self._flush_paragraph(story, current_paragraph)
                current_paragraph = []
                
                if in_list:
                    in_list = False
                    story.append(Spacer(1, 10))
                
                self._add_main_heading(story, line)
                continue
            
            # Sub-heading
            if self.content_parser.is_sub_heading(line):
                self._flush_paragraph(story, current_paragraph)
                current_paragraph = []
                
                story.append(Spacer(1, 10))
                story.append(Paragraph(
                    self.content_parser.apply_rich_formatting(line),
                    self.style_manager.get_style('SubHeading')
                ))
                story.append(Spacer(1, 5))
                continue
            
            # List item
            if self.content_parser.is_list_item(line):
                self._flush_paragraph(story, current_paragraph)
                current_paragraph = []
                
                if not in_list:
                    in_list = True
                    story.append(Spacer(1, 5))
                
                self._add_list_item(story, line)
                continue
            
            # Key-value pair
            if self.content_parser.is_key_value_pair(line):
                self._flush_paragraph(story, current_paragraph)
                current_paragraph = []
                
                self._add_key_value_pair(story, line)
                continue
            
            # Normal paragraph content
            if in_list:
                in_list = False
                story.append(Spacer(1, 10))
            
            current_paragraph.append(line)
        
        # Flush remaining paragraph
        self._flush_paragraph(story, current_paragraph)
        
        return story
    
    def _flush_paragraph(self, story: List[Any], paragraph_lines: List[str]):
        """Add accumulated paragraph lines to story"""
        if paragraph_lines:
            text = ' '.join(paragraph_lines)
            formatted_text = self.content_parser.apply_rich_formatting(text)
            story.append(Paragraph(
                formatted_text,
                self.style_manager.get_style('CustomBody')
            ))
    
    def _add_main_heading(self, story: List[Any], line: str):
        """Add main heading with decoration"""
        story.append(Spacer(1, 20))
        story.append(HeadingDecorator(color_scheme=self.color_scheme))
        
        cleaned_heading = self.content_parser.clean_heading(line)
        formatted_heading = self.content_parser.apply_rich_formatting(cleaned_heading)
        
        story.append(Paragraph(
            formatted_heading,
            self.style_manager.get_style('SectionHeading')
        ))
        
        story.append(HRFlowable(
            width="100%",
            thickness=2,
            color=self.color_scheme.secondary,
            spaceBefore=5,
            spaceAfter=15
        ))
    
    def _add_list_item(self, story: List[Any], line: str):
        """Add formatted list item"""
        # Clean list marker
        text = line.lstrip('•-*►▸ ').strip()
        bullet = '▸'
        
        formatted_item = (
            f'<font color="{self.color_scheme.secondary}">{bullet}</font> '
            f'{self.content_parser.apply_rich_formatting(text)}'
        )
        
        story.append(Paragraph(
            formatted_item,
            self.style_manager.get_style('ListItem')
        ))
    
    def _add_key_value_pair(self, story: List[Any], line: str):
        """Add key-value pair"""
        key, value = line.split(':', 1)
        formatted = f'<b>{key.strip()}:</b> {value.strip()}'
        
        story.append(Paragraph(
            formatted,
            self.style_manager.get_style('CustomBody')
        ))
    
    def add_page_number(self, canvas, doc):
        """Add page number and watermark"""
        canvas.saveState()
        
        # Page number
        canvas.setFont(PDFConstants.DEFAULT_FONT, 9)
        canvas.setFillColor(self.color_scheme.muted)
        page_num = canvas.getPageNumber()
        canvas.drawRightString(
            doc.pagesize[0] - 2*cm, 
            2*cm, 
            f"Page {page_num}"
        )
        
        # Watermark
        self._add_watermark(canvas, doc)
        
        canvas.restoreState()
    
    def _add_watermark(self, canvas, doc):
        """Add diagonal watermark"""
        canvas.saveState()
        
        canvas.setFillColor(colors.HexColor('#F0F0F0'))
        canvas.setFont(PDFConstants.DEFAULT_FONT, PDFConstants.WATERMARK_FONT_SIZE)
        
        # Move to center and rotate
        canvas.translate(doc.pagesize[0] / 2, doc.pagesize[1] / 2)
        canvas.rotate(PDFConstants.WATERMARK_ROTATION)
        canvas.drawCentredString(0, 0, PDFConstants.WATERMARK_TEXT)
        
        canvas.restoreState()


class JobCompatibilityPDFGenerator:
    """Main PDF generator for job compatibility reports"""
    
    def __init__(self):
        self.color_scheme = ColorScheme()
        self.style_manager = StyleManager(self.color_scheme)
        self.content_parser = ContentParser(self.style_manager, self.color_scheme)
        self.pdf_builder = PDFBuilder(
            self.style_manager, 
            self.content_parser, 
            self.color_scheme
        )
    
    def generate_pdf(self, 
                     report_content: str,
                     job_title: str = "Unknown Position",
                     candidate_name: str = "Candidate",
                     language: str = "en") -> bytes:
        """
        Generate PDF report from content.
        
        Args:
            report_content: The report text content
            job_title: Title of the job position
            candidate_name: Name of the candidate
            language: Language code ('en' or 'tr')
            
        Returns:
            PDF file as bytes
        """
        # Create document metadata
        metadata = DocumentMetadata(
            candidate_name=candidate_name,
            job_title=job_title,
            language=Language(language)
        )
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=PDFConstants.PAGE_SIZE,
            leftMargin=PDFConstants.LEFT_MARGIN,
            rightMargin=PDFConstants.RIGHT_MARGIN,
            topMargin=PDFConstants.TOP_MARGIN,
            bottomMargin=PDFConstants.BOTTOM_MARGIN
        )
        
        # Calculate usable width
        page_width, _ = PDFConstants.PAGE_SIZE
        self.pdf_builder.doc_width = (
            page_width - PDFConstants.LEFT_MARGIN - PDFConstants.RIGHT_MARGIN
        )
        
        # Build document content
        story = self._build_document_content(report_content, metadata)
        
        # Build PDF
        doc.build(
            story,
            onFirstPage=self.pdf_builder.add_page_number,
            onLaterPages=self.pdf_builder.add_page_number
        )
        
        # Return PDF bytes
        buffer.seek(0)
        return buffer.getvalue()
    
    def _build_document_content(self, 
                                report_content: str, 
                                metadata: DocumentMetadata) -> List[Any]:
        """Build complete document content"""
        story = []
        
        # Title
        title = self._get_document_title(metadata.language)
        story.append(Paragraph(title, self.style_manager.get_style('CustomTitle')))
        story.append(Spacer(1, 20))
        
        # Header section
        story.append(self.pdf_builder.create_header_section(metadata))
        story.append(Spacer(1, 30))
        
        # Summary box (if applicable)
        if self._should_add_summary(report_content):
            summary_text = self._get_summary_text(metadata.language)
            story.append(Paragraph(
                summary_text,
                self.style_manager.get_style('HighlightBox')
            ))
            story.append(Spacer(1, 20))
        
        # Main content
        story.extend(self.pdf_builder.parse_content(
            report_content, 
            metadata.language.value
        ))
        
        # Footer
        story.extend(self._build_footer(metadata.language))
        
        return story
    
    def _get_document_title(self, language: Language) -> str:
        """Get document title based on language"""
        return (
            "Job Compatibility Analysis Report" 
            if language == Language.ENGLISH 
            else "İş Uyumluluk Analizi Raporu"
        )
    
    def _should_add_summary(self, content: str) -> bool:
        """Check if summary box should be added"""
        content_lower = content.lower()
        return "summary" in content_lower or "özet" in content_lower
    
    def _get_summary_text(self, language: Language) -> str:
        """Get summary text based on language"""
        return (
            "This comprehensive analysis evaluates the candidate's compatibility "
            "with the specified position based on skills, experience, and requirements."
            if language == Language.ENGLISH else
            "Bu kapsamlı analiz, adayın belirtilen pozisyon ile uyumluluğunu "
            "beceriler, deneyim ve gereksinimler açısından değerlendirmektedir."
        )
    
    def _build_footer(self, language: Language) -> List[Any]:
        """Build document footer"""
        footer_elements = []
        
        # Separator
        footer_elements.append(Spacer(1, 40))
        footer_elements.append(HRFlowable(
            width="100%",
            thickness=0.5,
            color=self.color_scheme.light_bg
        ))
        footer_elements.append(Spacer(1, 15))
        
        # Footer text
        footer_text = (
            "Generated by Selman Dedeakayoğulları's AI Portfolio Assistant | "
            "Visit portfolio for more information"
            if language == Language.ENGLISH else
            "Selman Dedeakayoğulları'nın AI Portföy Asistanı tarafından oluşturuldu | "
            "Daha fazla bilgi için portföyü ziyaret edin"
        )
        
        footer_elements.append(Paragraph(
            footer_text,
            self.style_manager.get_style('Footer')
        ))
        
        return footer_elements


# Utility functions for external use
def generate_compatibility_pdf(report_content: str,
                              job_title: str = "Unknown Position",
                              candidate_name: str = "Candidate",
                              language: str = "en") -> bytes:
    """
    Convenience function to generate PDF report.
    
    Args:
        report_content: The report text content
        job_title: Title of the job position
        candidate_name: Name of the candidate
        language: Language code ('en' or 'tr')
        
    Returns:
        PDF file as bytes
    """
    generator = JobCompatibilityPDFGenerator()
    return generator.generate_pdf(
        report_content=report_content,
        job_title=job_title,
        candidate_name=candidate_name,
        language=language
    )