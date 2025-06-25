import os
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any


class EmailTool:
    """SMTP email functionality"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = st.secrets.get("GMAIL_EMAIL", os.getenv("GMAIL_EMAIL"))
        self.email_password = st.secrets.get("GMAIL_APP_PASSWORD", os.getenv("GMAIL_APP_PASSWORD"))
        self.recipient_email = st.secrets.get("RECIPIENT_EMAIL", os.getenv("RECIPIENT_EMAIL"))
        self.linkedin_url = "https://www.linkedin.com/in/selman-dedeakayo%C4%9Fullar%C4%B1-443b431a7/"
            
    def _send_confirmation_email(self, sender_email: str, sender_name: str, language: str = "en"):
        """Send confirmation email to the sender"""
        try:
            # Email signature
            signature = f"""
    Selman DEDEAKAYOĞULLARI
    AI Engineer & Researcher

    📧 Email: selmandedeakayogullari@gmail.com
    🔗 LinkedIn: {self.linkedin_url}
    🌐 Personal Website: https://selmandedeakay.github.io
    """

            # Email content based on language
            if language == "tr":
                subject = "Mesajınız Alındı"
                body = f"""
    Merhaba {sender_name},

    Portföy chatbotum aracılığıyla gönderdiğiniz mesajınızı aldım. En kısa sürede size geri dönüş yapacağım.

    Eğer acil bir durum varsa, benimle LinkedIn üzerinden iletişime geçebilirsiniz:
    {self.linkedin_url}

    Eğer böyle bir e-posta beklemiyorsanız, birisi yanlışlıkla sizin e-posta adresinizi girmiş olabilir. Lütfen bu mesajı görmezden gelin.
    
    Sevgiler,{signature}
    """
            else:  # English
                subject = "Your Message Has Been Received"
                body = f"""
    Hi {sender_name},

    I've received your message sent through my portfolio chatbot. I'll get back to you as soon as possible.

    If this is urgent, you can contact me directly on LinkedIn:
    {self.linkedin_url}

    If you did not expect this email, someone may have entered your email address by mistake. Please ignore this message.
    
    Best regards,{signature}
    """

            # Create confirmation message
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = sender_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send confirmation email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.sendmail(self.email_user, sender_email, msg.as_string())
            server.quit()
            
        except Exception as e:
            print(f"Failed to send confirmation email: {str(e)}")
    
    def send_email(self, sender_name: str, sender_email: str, subject: str, message: str) -> Dict[str, Any]:
        """Send email via SMTP"""
        try:
            # Debug info
            print(f"SMTP Server: {self.smtp_server}")
            print(f"SMTP Port: {self.smtp_port}")
            print(f"Email User: {self.email_user}")
            print(f"Recipient: {self.recipient_email}")
            
            # Create main message
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            msg['Reply-To'] = sender_email
            
            # Email body
            body = f"""
New contact from portfolio chatbot:

From: {sender_name}
Email: {sender_email}

Message:
{message}

---
Sent via Portfolio RAG Chatbot
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send main email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.set_debuglevel(1)
            server.starttls()
            server.login(self.email_user, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_user, self.recipient_email, text)
            server.quit()
            
            # Detect language from message (simple check for Turkish characters)
            language = "tr" if any(char in message.lower() for char in ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü']) else "en"
            
            # Send confirmation email
            self._send_confirmation_email(sender_email, sender_name, language)
            
            # Clear CAPTCHA after successful send
            if 'email_captcha' in st.session_state:
                del st.session_state.email_captcha
            
            return {
                "success": True,
                "message": f"Email sent successfully to {self.recipient_email}! Selman will get back to you soon."
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                "success": False,
                "message": "Email authentication failed. Please check EMAIL_USER and EMAIL_PASSWORD (use App Password for Gmail)."
            }
        except smtplib.SMTPException as e:
            return {
                "success": False,
                "message": f"SMTP error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send email: {str(e)}"
            }