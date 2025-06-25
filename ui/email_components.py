import streamlit as st
from typing import Dict


def get_ui_text(language: str) -> Dict[str, str]:
    """Get UI text based on language"""
    if language == "tr":
        return {
            "email_review_title": "📧 **Lütfen e-postanızı göndermeden önce kontrol edin:**",
            "from_label": "**Gönderen:**",
            "email_label": "**E-posta:**",
            "message_label": "**Mesaj:**",
            "send_button": "✅ E-postayı Gönder",
            "cancel_button": "❌ İptal Et",
            "edit_button": "✏️ Mesajı Düzenle",
            "edit_title": "✏️ **E-postanızı düzenleyin:**",
            "name_field": "Adınız",
            "email_field": "E-posta Adresiniz",
            "message_field": "Mesaj",
            "save_button": "💾 Değişiklikleri Kaydet",
            "cancel_edit_button": "❌ Düzenlemeyi İptal Et",
            "email_sent": "✅ E-posta başarıyla gönderildi! Selman size yakında dönüş yapacak.",
            "email_failed": "❌ E-posta gönderilemedi: ",
            "email_cancelled": "E-posta iptal edildi. Başka bir konuda yardımcı olabileceğim bir şey var mı?",
            "email_prepared": "E-postanız Selman'a hazırlandı. Lütfen göndermeden önce aşağıdaki detayları kontrol edin."
        }
    else:  # English
        return {
            "email_review_title": "📧 **Please review your email before sending:**",
            "from_label": "**From:**",
            "email_label": "**Email:**",
            "message_label": "**Message:**",
            "send_button": "✅ Send Email",
            "cancel_button": "❌ Cancel",
            "edit_button": "✏️ Edit Message",
            "edit_title": "✏️ **Edit your email:**",
            "name_field": "Your Name",
            "email_field": "Your Email",
            "message_field": "Message",
            "save_button": "💾 Save Changes",
            "cancel_edit_button": "❌ Cancel Editing",
            "email_sent": "✅ Email sent successfully! Selman will get back to you soon.",
            "email_failed": "❌ Failed to send email: ",
            "email_cancelled": "Email cancelled. Is there anything else I can help you with?",
            "email_prepared": "I've prepared your email to Selman. Please review the details below before sending."
        }


def render_email_verification_card(email_data: Dict[str, str], language: str):
    """Render email verification card within the chat message"""
    ui_text = get_ui_text(language)
    
    with st.container():
        st.info(ui_text["email_review_title"])
        
        # Display email details in a nice format
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(ui_text["from_label"])
            st.markdown(ui_text["email_label"])
            st.markdown(ui_text["message_label"])
        
        with col2:
            st.markdown(f"{email_data['sender_name']}")
            st.markdown(f"{email_data['sender_email']}")
            st.markdown(f"{email_data['message']}")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button(ui_text["send_button"], type="primary", key="send_email_btn"):
                st.session_state.email_action = "send"
                st.rerun()
        
        with col2:
            if st.button(ui_text["cancel_button"], key="cancel_email_btn"):
                st.session_state.email_action = "cancel"
                st.rerun()
        
        with col3:
            if st.button(ui_text["edit_button"], key="edit_email_btn"):
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
            
            col1, col2 = st.columns(2)
            
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