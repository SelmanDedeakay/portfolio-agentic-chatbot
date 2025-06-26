import streamlit as st
from typing import Dict
import random
import string

def get_ui_text(language: str) -> Dict[str, str]:
    """Get UI text based on language"""
    if language == "tr":
        return {
            "email_review_title": "📧 **Lütfen e-postanızı göndermeden önce kontrol edin:**",
            "from_label": "Gönderen:",
            "email_label": "E-posta:",
            "message_label": "Mesaj:",
            "send_button": "✅ E-postayı Gönder",
            "cancel_button": "❌ İptal Et",
            "edit_button": "✏️ Mesajı Düzenle",
            "edit_title": "✏️ **E-postanızı düzenleyin:**",
            "name_field": "Adınız",
            "email_field": "E-posta Adresiniz",
            "message_field": "Mesaj",
            "save_button": "💾 Değişiklikleri Kaydet",
            "cancel_edit_button": "❌ Düzenlemeyi İptal Et",
            "email_sent": "✅ E-posta başarıyla gönderildi! Sizlere bir onay e-postası gönderdim. Lütfen gelen kutunuzu kontrol edin, eğer görmüyorsanız spam klasörünüze bakın. Selman size en kısa sürede geri dönecektir.",
            "email_failed": "❌ E-posta gönderilemedi: ",
            "email_cancelled": "E-posta iptal edildi. Başka bir konuda yardımcı olabileceğim bir şey var mı?",
            "email_prepared": "E-postanız hazırlandı. Lütfen göndermeden önce aşağıdaki detayları kontrol edin.",
            "captcha_label": "**CAPTCHA Doğrulama:**",
            "captcha_instructions": "Lütfen aşağıda gösterilen karakterleri girin (Kimse botların posta kutusunu işgal etmesini istemez, değil mi?):",
            "captcha_error": "❌ CAPTCHA doğrulaması başarısız. Lütfen tekrar deneyin.",
            "captcha_placeholder": "CAPTCHA'yı buraya girin",
        }
    else:  # English
        return {
            "email_review_title": "📧 **Please review your email before sending:**",
            "from_label": "From:",
            "email_label": "Email:",
            "message_label": "Message:",
            "send_button": "✅ Send Email",
            "cancel_button": "❌ Cancel",
            "edit_button": "✏️ Edit Message",
            "edit_title": "✏️ **Edit your email:**",
            "name_field": "Your Name",
            "email_field": "Your Email",
            "message_field": "Message",
            "save_button": "💾 Save Changes",
            "cancel_edit_button": "❌ Cancel Editing",
            "email_sent": "✅ Email sent successfully! You should have got a confirmation email. Please check your spam folder if you don't see it in your inbox. Selman will get back to you soon.",
            "email_failed": "❌ Failed to send email: ",
            "email_cancelled": "Email cancelled. Is there anything else I can help you with?",
            "email_prepared": "I've prepared your email. Please review the details below before sending.",
            "captcha_label": "**CAPTCHA Verification:**",
            "captcha_instructions": "Please enter the characters shown below (No one wants bots spamming his mailbox, right?):",
            "captcha_error": "❌ CAPTCHA verification failed. Please try again.",
            "captcha_placeholder": "Enter CAPTCHA here",
        }

def generate_captcha(length=6):
    """Generate a simple CAPTCHA string"""
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def render_email_verification_card(email_data: Dict[str, str], language: str):
    """Render email verification card within the chat message"""
    ui_text = get_ui_text(language)
    
    with st.container():
        st.info(ui_text["email_review_title"])
        
        # Display email details in a responsive format
        st.markdown(
            """
            <style>
            .email-details {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .email-row {
                display: flex;
                align-items: flex-start;
                gap: 15px;
            }
            .email-label {
                min-width: 80px;
                font-weight: bold;
                flex-shrink: 0;
            }
            .email-value {
                flex: 1;
                word-break: break-word;
            }
            @media (max-width: 768px) {
                .email-row {
                    gap: 10px;
                }
                .email-label {
                    min-width: 70px;
                    font-size: 14px;
                }
                .email-value {
                    font-size: 14px;
                }
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div class="email-details">
                <div class="email-row">
                    <div class="email-label">{ui_text["from_label"]}</div>
                    <div class="email-value">{email_data['sender_name']}</div>
                </div>
                <div class="email-row">
                    <div class="email-label">{ui_text["email_label"]}</div>
                    <div class="email-value">{email_data['sender_email']}</div>
                </div>
                <div class="email-row">
                    <div class="email-label">{ui_text["message_label"]}</div>
                    <div class="email-value">{email_data['message']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Generate and store CAPTCHA if not already present
        if 'email_captcha' not in st.session_state:
            st.session_state.email_captcha = generate_captcha()
        
        # Display CAPTCHA verification
        st.markdown("---")
        st.markdown(ui_text["captcha_label"])
        st.markdown(ui_text["captcha_instructions"])
        
        # Display CAPTCHA in a styled box
        st.markdown(
            f"""
            <div style="
                background: #f0f2f6;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                font-family: monospace;
                font-size: 24px;
                letter-spacing: 5px;
                margin: 10px 0;
                user-select: none;
            ">{st.session_state.email_captcha}</div>
            """,
            unsafe_allow_html=True
        )
        
        captcha_input = st.text_input(
            ui_text["captcha_placeholder"],
            key="email_captcha_input",
            label_visibility="collapsed"
        )
        
        # Action buttons - Equal grid layout
        col1, col2, col3 = st.columns(3)  # Equal columns (1:1:1 ratio)
        
        with col1:
            if st.button(ui_text["send_button"], type="primary", key="send_email_btn", use_container_width=True):
                if captcha_input.upper() == st.session_state.email_captcha:
                    st.session_state.email_action = "send"
                    st.rerun()
                else:
                    st.error(ui_text["captcha_error"])
        
        with col2:
            if st.button(ui_text["cancel_button"], key="cancel_email_btn", use_container_width=True):
                st.session_state.email_action = "cancel"
                st.rerun()
        
        with col3:
            if st.button(ui_text["edit_button"], key="edit_email_btn", use_container_width=True):
                st.session_state.email_action = "edit"
                st.rerun()

def render_email_editor_card(email_data: Dict[str, str], language: str):
    """Render email editor card within the chat message"""
    ui_text = get_ui_text(language)
    
    with st.container():
        st.info(ui_text["edit_title"])
        
        # Editable fields
        with st.form("email_editor", clear_on_submit=False):
            sender_name = st.text_input(ui_text["name_field"], value=email_data['sender_name'])
            sender_email = st.text_input(ui_text["email_field"], value=email_data['sender_email'])
            message = st.text_area(ui_text["message_field"], value=email_data['message'], height=150)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.form_submit_button(ui_text["save_button"], type="primary"):
                    # Update email data
                    st.session_state.pending_email = {
                        'sender_name': sender_name,
                        'sender_email': sender_email,
                        'subject': 'New Message from Portfolio Bot',
                        'message': message
                    }
                    st.session_state.editing_email = False
                    st.session_state.email_action = None
                    st.rerun()
            
            with col2:
                if st.form_submit_button(ui_text["cancel_edit_button"]):
                    st.session_state.editing_email = False
                    st.session_state.email_action = None
                    st.rerun()