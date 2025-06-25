import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import streamlit as st
import re
from urllib.parse import urljoin
import time


class SocialMediaAggregator:
    """Aggregate posts from Medium only"""
    
    def __init__(self):
        self.medium_username = "selmandedeakayogullari"  # From CV data
        self.cache_duration = 3600  # 1 hour cache
        
    def get_medium_posts(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch latest Medium articles via RSS feed"""
        try:
            # Medium RSS feed URL
            rss_url = f"https://medium.com/@{self.medium_username}/feed"
            
            # Check cache first
            cache_key = f"medium_posts_{self.medium_username}"
            if self._is_cache_valid(cache_key):
                return st.session_state.get(cache_key, [])
            
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            
            posts = []
            for entry in feed.entries[:limit]:
                # Clean up Medium's description (remove HTML tags)
                description = re.sub(r'<[^>]+>', '', entry.summary)
                description = description[:200] + "..." if len(description) > 200 else description
                
                # Extract publication date
                pub_date = entry.get('published_parsed')
                if pub_date:
                    pub_datetime = datetime(*pub_date[:6])
                    time_ago = self._get_time_ago(pub_datetime)
                else:
                    time_ago = "Unknown"
                
                posts.append({
                    'platform': 'Medium',
                    'title': entry.title,
                    'description': description,
                    'url': entry.link,
                    'published': time_ago,
                    'published_date': pub_datetime if pub_date else None
                })
            
            # Cache results
            st.session_state[cache_key] = posts
            st.session_state[f"{cache_key}_timestamp"] = time.time()
            
            return posts
            
        except Exception as e:
            st.error(f"Error fetching Medium posts: {e}")
            return []
    
    def get_all_posts(self, limit_per_platform: int = 3) -> List[Dict[str, Any]]:
        """Get Medium posts only"""
        # Get Medium posts
        medium_posts = self.get_medium_posts(limit_per_platform)
        
        # Sort by date (most recent first)
        medium_posts.sort(
            key=lambda x: x.get('published_date', datetime.min), 
            reverse=True
        )
        
        return medium_posts
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        timestamp_key = f"{cache_key}_timestamp"
        if timestamp_key not in st.session_state:
            return False
        
        last_update = st.session_state[timestamp_key]
        return (time.time() - last_update) < self.cache_duration
    
    def _get_time_ago(self, pub_date: datetime) -> str:
        """Convert publication date to human-readable time ago"""
        now = datetime.now()
        diff = now - pub_date
        
        if diff.days > 0:
            if diff.days == 1:
                return "1 day ago"
            elif diff.days < 7:
                return f"{diff.days} days ago"
            elif diff.days < 30:
                weeks = diff.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            else:
                months = diff.days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"
        else:
            hours = diff.seconds // 3600
            if hours > 0:
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            else:
                minutes = diff.seconds // 60
                return f"{minutes} minute{'s' if minutes > 1 else ''} ago" if minutes > 0 else "Just now"
    
    def format_posts_for_chat(self, posts: List[Dict[str, Any]], language: str = "en") -> str:
        """Format posts for chatbot response"""
        if not posts:
            return "No recent posts found." if language == "en" else "Son paylaşım bulunamadı."
        
        title = "📱 **Selman's Latest Posts:**\n\n" if language == "en" else "📱 **Selman'ın Son Paylaşımları:**\n\n"
        
        formatted = title
        for post in posts:
            platform_emoji = "📝"
            formatted += f"{platform_emoji} **{post['platform']}** - {post['published']}\n"
            formatted += f"**{post['title']}**\n"
            formatted += f"{post['description']}\n"
            formatted += f"[Read more]({post['url']})\n\n"
        
        return formatted
    
    def get_post_summary(self, query: str, posts: List[Dict[str, Any]]) -> str:
        """Get relevant posts based on query"""
        if not posts:
            return "No posts available to search."
        
        # Simple keyword matching
        query_lower = query.lower()
        relevant_posts = []
        
        for post in posts:
            if (query_lower in post['title'].lower() or 
                query_lower in post['description'].lower()):
                relevant_posts.append(post)
        
        if relevant_posts:
            return self.format_posts_for_chat(relevant_posts)
        else:
            return self.format_posts_for_chat(posts[:2])  # Show recent posts if no match