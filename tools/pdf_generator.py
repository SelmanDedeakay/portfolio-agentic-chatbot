
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
import io
import streamlit as st
from datetime import datetime
import re

class GradientBackground(Flowable):
    """Custom flowable for gradient backgrounds"""
    def __init__(self, width, height, color1, color2):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color1 = color1
        self.color2 = color2
    
    def draw(self):
        steps = 20
        for i in range(steps):
            ratio = i / float(steps)
            r = self.color1[0] * (1 - ratio) + self.color2[0] * ratio
            g = self.color1[1] * (1 - ratio) + self.color2[1] * ratio
            b = self.color1[2] * (1 - ratio) + self.color2[2] * ratio
            
            self.canv.setFillColorRGB(r, g, b)
            y = self.height * i / steps
            h = self.height / steps
            self.canv.rect(0, y, self.width, h, fill=1, stroke=0)

class ScoreBox(Flowable):
    """Custom flowable for score display with visual appeal"""
    def __init__(self, score_text: str, width: float, height: float = 30):
        Flowable.__init__(self)
        self.score_text = score_text
        self.width = width
        self.height = height
        
    def draw(self):
        # Extract percentage from text
        percentages = re.findall(r'(\d+\.?\d*)%', self.score_text)
        score_value = float(percentages[0]) if percentages else 0
        
        # Determine color based on score
        if score_value >= 80:
            main_color = colors.HexColor('#27AE60')  # Green
            bg_color = colors.HexColor('#E8F8F5')
        elif score_value >= 60:
            main_color = colors.HexColor('#F39C12')  # Orange
            bg_color = colors.HexColor('#FEF5E7')
        else:
            main_color = colors.HexColor('#E74C3C')  # Red
            bg_color = colors.HexColor('#FADBD8')
        
        # Draw rounded rectangle background
        self.canv.setFillColor(bg_color)
        self.canv.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=1)
        self.canv.setStrokeColor(main_color)
        self.canv.setLineWidth(1.5)
        self.canv.roundRect(0, 0, self.width, self.height, 5, fill=0, stroke=1)
        
        # Progress bar
        bar_width = self.width - 40
        bar_height = 6
        bar_x = 20
        bar_y = 10
        
        # Background bar
        self.canv.setFillColor(colors.HexColor('#ECEFF1'))
        self.canv.roundRect(bar_x, bar_y, bar_width, bar_height, 3, fill=1, stroke=0)
        
        # Filled bar
        progress_width = max(0, min(bar_width, bar_width * (score_value / 100)))
        self.canv.setFillColor(main_color)
        self.canv.roundRect(bar_x, bar_y, progress_width, bar_height, 3, fill=1, stroke=0)
        
        # Text
        self.canv.setFont('DejaVuSans-Bold', 12)
        self.canv.setFillColor(colors.HexColor('#2C3E50'))
        self.canv.drawCentredString(self.width/2, self.height - 18, self.score_text)

class JobCompatibilityPDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', '../fonts/DejaVuSans.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', '../fonts/DejaVuSans-Bold.ttf'))
        except:
            pass
        
        self.colors = {
            'primary':   colors.HexColor('#2C3E50'),
            'secondary': colors.HexColor('#3498DB'),
            'accent':    colors.HexColor('#E74C3C'),
            'success':   colors.HexColor('#27AE60'),
            'warning':   colors.HexColor('#F39C12'),
            'text':      colors.HexColor('#34495E'),
            'light_bg':  colors.HexColor('#ECF0F1'),
            'white':     colors.white
        }
        self._setup_styles()

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle', parent=self.styles['Title'],
            fontSize=28, leading=36, spaceBefore=20, spaceAfter=30,
            textColor=self.colors['primary'], alignment=TA_CENTER,
            fontName='DejaVuSans-Bold'
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeading', parent=self.styles['Heading2'],
            fontSize=18, leading=24, spaceBefore=25, spaceAfter=15,
            textColor=self.colors['secondary'], fontName='DejaVuSans-Bold',
            borderPadding=(0,0,0,5), borderColor=self.colors['secondary']
        ))
        self.styles.add(ParagraphStyle(
            name='SubHeading', parent=self.styles['Heading3'],
            fontSize=14, leading=18, spaceBefore=15, spaceAfter=10,
            textColor=self.colors['text'], fontName='DejaVuSans-Bold',
            leftIndent=15
        ))
        self.styles.add(ParagraphStyle(
            name='CustomBody', parent=self.styles['Normal'],
            fontSize=11, leading=18, spaceBefore=6, spaceAfter=12,
            leftIndent=25, rightIndent=25, alignment=TA_JUSTIFY,
            fontName='DejaVuSans', textColor=self.colors['text'], wordWrap='CJK'
        ))
        self.styles.add(ParagraphStyle(
            name='ListItem', parent=self.styles['Normal'],
            fontSize=11, leading=16, spaceAfter=8,
            leftIndent=40, bulletIndent=25,
            fontName='DejaVuSans', textColor=self.colors['text'],
            bulletFontName='Symbol', bulletFontSize=8, bulletColor=self.colors['secondary']
        ))
        self.styles.add(ParagraphStyle(
            name='Footer', parent=self.styles['Normal'],
            fontSize=9, leading=12, alignment=TA_CENTER,
            textColor=colors.HexColor('#7F8C8D'),
            fontName='DejaVuSans'
        ))
        self.styles.add(ParagraphStyle(
            name='HighlightBox', parent=self.styles['Normal'],
            fontSize=11, leading=18, spaceBefore=15, spaceAfter=15,
            leftIndent=30, rightIndent=30,
            backColor=self.colors['light_bg'],
            borderColor=self.colors['secondary'],
            borderWidth=1, borderPadding=15, borderRadius=5,
            fontName='DejaVuSans', textColor=self.colors['text']
        ))

    def _create_header_section(self, candidate_name, job_title, language):
        """Üst kısmı koyu arkaplan ve beyaz metinle oluşturur."""
        header_data = []
        if language == "tr":
            header_data = [
                ["Aday:",      candidate_name],
                ["Pozisyon:",  job_title],
                ["Tarih:",     datetime.now().strftime("%d/%m/%Y")]
            ]
        else:
            header_data = [
                ["Candidate:", candidate_name],
                ["Position:",  job_title],
                ["Date:",      datetime.now().strftime("%d/%m/%Y")]
            ]

        header_table = Table(header_data, colWidths=[4*cm, 11*cm])
        header_table.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,-1), self.colors['primary']),
            ('TEXTCOLOR',   (0,0), (-1,-1), self.colors['white']),   # METİN BEYAZ
            ('ALIGN',       (0,0), (0,-1), 'RIGHT'),
            ('ALIGN',       (1,0), (1,-1), 'LEFT'),
            ('FONTNAME',    (0,0), (-1,-1), 'DejaVuSans'),
            ('TOPPADDING',  (0,0), (-1,-1), 8),
            ('BOTTOMPADDING',(0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
            ('RIGHTPADDING',(0,0), (-1,-1), 15),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ]))

        wrapper = Table([[header_table]], colWidths=[16*cm], rowHeights=[3*cm])
        wrapper.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,-1), self.colors['primary']),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING',(0,0), (-1,-1), 0),
            ('TOPPADDING',  (0,0), (-1,-1), 0),
            ('BOTTOMPADDING',(0,0),(-1,-1), 0),
        ]))
        return wrapper

    def _contains_score(self, line: str) -> bool:
        """Sadece gerçek yüzde değerlerini yakalar."""
        return bool(re.search(r'\b\d+(\.\d+)?%\b', line))

    def _is_main_heading(self, line: str) -> bool:
        # ... orijinal metodun tamamı ...
        line_clean = line.strip()
        if line_clean.startswith('##'): return True
        # (diğer kontroller aynı)
        section_keywords = [
            'overview','summary','analysis','skills','experience',
            'recommendations','conclusion','strengths','weaknesses',
            'özet','analiz','beceriler','deneyim','öneriler','sonuç'
        ]
        if any(k in line_clean.lower() for k in section_keywords) and len(line_clean.split())<=4:
            return True
        return False

    def _is_sub_heading(self, line: str) -> bool:
        # ... orijinal metodun tamamı ...
        line_clean = line.strip()
        if line_clean.endswith(':') and len(line_clean.split())<=6:
            if not re.search(r'\d{1,2}:\d{2}', line_clean):
                return True
        return False

    def _clean_heading(self, line: str) -> str:
        cleaned = line.replace('##','').replace('**','').strip()
        cleaned = re.sub(r'^\d+\.\s*','', cleaned)
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    def _rich_format(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'`(.*?)`', r'<font face="Courier" color="#2C3E50" backColor="#F8F9FA">\1</font>', text)
        text = re.sub(r'\b(\d+\.?\d*%)\b', r'<font color="#E74C3C"><b>\1</b></font>', text)
        text = re.sub(r'\b(20\d{2})\b', r'<font color="#3498DB">\1</font>', text)
        return text

    def _create_heading_decorator(self):
        d = Drawing(100, 10)
        d.add(Rect(0, 3, 30, 4, fillColor=self.colors['secondary'], strokeColor=None))
        d.add(Rect(35,3, 15, 4, fillColor=self.colors['accent'], strokeColor=None))
        d.add(Rect(55,3, 10, 4, fillColor=self.colors['warning'],strokeColor=None))
        return d

    def _parse_report_content(self, content: str, language: str):
        story = []
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        cur_para = []
        in_list = False

        for line in lines:
            # 1) ScoreBox
            if self._contains_score(line):
                if cur_para:
                    story.append(Paragraph(self._rich_format(' '.join(cur_para)), self.styles['CustomBody']))
                    cur_para = []
                story.append(Spacer(1, 10))
                # Sayfa genişliğini kullanıyoruz:
                story.append(ScoreBox(line, width=self.doc_width, height=30))
                story.append(Spacer(1, 15))
                continue

            # 2) Main heading
            if self._is_main_heading(line):
                if cur_para:
                    story.append(Paragraph(self._rich_format(' '.join(cur_para)), self.styles['CustomBody']))
                    cur_para = []
                if in_list:
                    in_list = False
                    story.append(Spacer(1, 10))

                story.append(Spacer(1, 20))
                story.append(self._create_heading_decorator())
                story.append(Paragraph(self._rich_format(self._clean_heading(line)), self.styles['SectionHeading']))
                story.append(HRFlowable(width="100%", thickness=2, color=self.colors['secondary'],
                                       spaceBefore=5, spaceAfter=15))
                continue

            # 3) Sub-heading
            if self._is_sub_heading(line):
                if cur_para:
                    story.append(Paragraph(self._rich_format(' '.join(cur_para)), self.styles['CustomBody']))
                    cur_para = []
                story.append(Spacer(1, 10))
                story.append(Paragraph(self._rich_format(line), self.styles['SubHeading']))
                story.append(Spacer(1, 5))
                continue

            # 4) List item
            if line.startswith(('•','-','*','►','▸')):
                if cur_para:
                    story.append(Paragraph(self._rich_format(' '.join(cur_para)), self.styles['CustomBody']))
                    cur_para = []
                if not in_list:
                    in_list = True
                    story.append(Spacer(1,5))
                txt = line.lstrip('•-*►▸ ').strip()
                bullet = '▸'
                story.append(Paragraph(
                    f'<font color="{self.colors["secondary"]}">{bullet}</font> {self._rich_format(txt)}',
                    self.styles['ListItem']
                ))
                continue

            # 5) Key-value
            if ':' in line and len(line.split(':')[0].split())<=3:
                if cur_para:
                    story.append(Paragraph(self._rich_format(' '.join(cur_para)), self.styles['CustomBody']))
                    cur_para = []
                k, v = line.split(':',1)
                story.append(Paragraph(f'<b>{k.strip()}:</b> {v.strip()}', self.styles['CustomBody']))
                continue

            # 6) Normal paragraph
            if in_list:
                in_list = False
                story.append(Spacer(1,10))
            cur_para.append(line)

        # Kalan paragraf
        if cur_para:
            story.append(Paragraph(self._rich_format(' '.join(cur_para)), self.styles['CustomBody']))

        return story

    def _add_page_number(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('DejaVuSans', 9)
        canvas.setFillColor(colors.HexColor('#7F8C8D'))
        page_num = canvas.getPageNumber()
        canvas.drawRightString(doc.pagesize[0]-2*cm, 2*cm, f"Page {page_num}")

        # Su damga
        canvas.setFillColor(colors.HexColor('#F0F0F0'))
        canvas.setFont('DejaVuSans', 60)
        canvas.saveState()
        canvas.translate(doc.pagesize[0]/2, doc.pagesize[1]/2)
        canvas.rotate(45)
        canvas.drawCentredString(0, 0, "AI ANALYSIS")
        canvas.restoreState()

        canvas.restoreState()

    def generate_pdf(self, report_content: str,
                     job_title: str="Unknown Position",
                     candidate_name: str="Candidate",
                     language: str="en") -> bytes:

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=1.5*cm, bottomMargin=2*cm
        )

        # Sayfa kullanılabilir genişliğini hesapla:
        pageW, pageH = A4
        self.doc_width = pageW - doc.leftMargin - doc.rightMargin

        story = []
        # Başlık
        title = "Job Compatibility Analysis Report" if language=="en" else "İş Uyumluluk Analizi Raporu"
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1,20))

        # Üst header
        story.append(self._create_header_section(candidate_name, job_title, language))
        story.append(Spacer(1,30))

        # Özet kutusu
        if "summary" in report_content.lower() or "özet" in report_content.lower():
            txt = (
                "This comprehensive analysis evaluates the candidate's compatibility "
                "with the specified position based on skills, experience, and requirements."
                if language=="en" else
                "Bu kapsamlı analiz, adayın belirtilen pozisyon ile uyumluluğunu "
                "beceriler, deneyim ve gereksinimler açısından değerlendirmektedir."
            )
            story.append(Paragraph(txt, self.styles['HighlightBox']))
            story.append(Spacer(1,20))

        # İçerik
        story.extend(self._parse_report_content(report_content, language))

        # Altbilgi
        story.append(Spacer(1,40))
        story.append(HRFlowable(width="100%", thickness=0.5, color=self.colors['light_bg']))
        story.append(Spacer(1,15))
        footer = (
            "Generated by Selman Dedeakayoğulları's AI Portfolio Assistant | Visit portfolio for more information"
            if language=="en" else
            "Selman Dedeakayoğulları'nın AI Portföy Asistanı tarafından oluşturuldu | Daha fazla bilgi için portföyü ziyaret edin"
        )
        story.append(Paragraph(footer, self.styles['Footer']))

        # PDF'i derle
        doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        buffer.seek(0)
        return buffer.getvalue()
