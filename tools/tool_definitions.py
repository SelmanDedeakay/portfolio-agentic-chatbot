from google.genai.types import Tool, FunctionDeclaration
from typing import List, Any, Dict
import streamlit as st


class ToolDefinitions:
    """Tool definitions for function calling"""
    
    @staticmethod
    def get_email_tool_definition() -> Tool:
        """Get email tool definition for function calling"""
        prepare_email_func = FunctionDeclaration(
            name="prepare_email",
            description="Prepare an email to Selman when someone wants to contact him. This function prepares the email for review before sending.",
            parameters={
                "type": "object",
                "properties": {
                    "sender_name": {
                        "type": "string",
                        "description": "Name of the person sending the email"
                    },
                    "sender_email": {
                        "type": "string",
                        "description": "Email address of the sender"
                    },
                    "message": {
                        "type": "string",
                        "description": "The message content to send to Selman"
                    }
                },
                "required": ["sender_name", "sender_email", "message"]
            }
        )
        
        return Tool(function_declarations=[prepare_email_func])
    
    @staticmethod
    def get_all_tools() -> List[Tool]:
        """Get all available tools"""
        return [
            ToolDefinitions.get_email_tool_definition()
        ]
    
    @staticmethod
    def execute_tool(tool_name: str, tool_args: Dict) -> Dict[str, Any]:
        """Execute the requested tool"""
        if tool_name == "prepare_email":
            # Add default subject
            tool_args['subject'] = "New Message from Portfolio Bot"
            # Store email data for verification
            st.session_state.pending_email = tool_args
            return {
                "success": True,
                "message": "Email prepared for review",
                "data": tool_args
            }
        else:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}"
            }