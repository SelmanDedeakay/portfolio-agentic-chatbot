# Add this to your tools/tool_definitions.py

from google.genai.types import Tool, FunctionDeclaration
from typing import List, Any, Dict
import streamlit as st
from tools.social_media_tool import SocialMediaAggregator


class ToolDefinitions:
    """Tool definitions for function calling"""
    
    def __init__(self):
        self.social_media_aggregator = SocialMediaAggregator()
    
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
    def get_social_media_tool_definition() -> Tool:
        """Get social media aggregator tool definition"""
        get_posts_func = FunctionDeclaration(
            name="get_recent_posts",
            description="Get Selman's recent posts from LinkedIn and Medium. Use this when someone asks about his latest posts, articles, or social media activity.",
            parameters={
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "Specific platform to get posts from (optional). Options: 'medium', 'linkedin', 'all'",
                        "enum": ["medium", "linkedin", "all"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of posts to retrieve (default: 5)",
                        "default": 5
                    },
                    "search_query": {
                        "type": "string",
                        "description": "Optional search query to filter posts by topic"
                    }
                },
                "required": []
            }
        )
        
        return Tool(function_declarations=[get_posts_func])
    
    def get_all_tools(self) -> List[Tool]:
        """Get all available tools"""
        return [
            self.get_email_tool_definition(),
            self.get_social_media_tool_definition()
        ]
    
    def execute_tool(self, tool_name: str, tool_args: Dict) -> Dict[str, Any]:
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
        
        elif tool_name == "get_recent_posts":
            try:
                platform = tool_args.get('platform', 'all')
                limit = tool_args.get('limit', 5)
                search_query = tool_args.get('search_query', '')
                
                # Get posts based on platform
                if platform == 'medium':
                    posts = self.social_media_aggregator.get_medium_posts(limit)
                elif platform == 'linkedin':
                    posts = self.social_media_aggregator.get_linkedin_posts_fallback()[:limit]
                else:  # all
                    posts = self.social_media_aggregator.get_all_posts(limit_per_platform=limit//2 + 1)
                
                # Filter by search query if provided
                if search_query and posts:
                    formatted_posts = self.social_media_aggregator.get_post_summary(search_query, posts)
                else:
                    # Detect language from session state
                    language = "tr" if any("tr" in str(msg).lower() for msg in st.session_state.get("messages", [])) else "en"
                    formatted_posts = self.social_media_aggregator.format_posts_for_chat(posts, language)
                
                return {
                    "success": True,
                    "message": "Posts retrieved successfully",
                    "data": {
                        "posts": posts,
                        "formatted_response": formatted_posts,
                        "count": len(posts)
                    }
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error retrieving posts: {str(e)}"
                }
        
        else:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}"
            }