import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import streamlit as st
import re
import time
import hashlib
from bs4 import BeautifulSoup


class SocialMediaAggregator:
    """Responsive design with mobile optimization"""
    
    def __init__(self):
        self.medium_username = "selmandedeakayogullari"
        self.cache_duration = 1800
        
    def get_medium_posts(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch Medium posts with real images"""
        try:
            rss_url = f"https://medium.com/@{self.medium_username}/feed"
            
            # Check cache
            cache_key = f"medium_posts_{self.medium_username}"
            if self._is_cache_valid(cache_key):
                return st.session_state.get(cache_key, [])
            
            feed = feedparser.parse(rss_url)
            posts = []
            
            for i, entry in enumerate(feed.entries[:limit]):
                # Simple processing
                pub_date = entry.get('published_parsed')
                if pub_date:
                    pub_datetime = datetime(*pub_date[:6])
                    time_ago = self._get_time_ago(pub_datetime)
                else:
                    time_ago = "Recent"
                    pub_datetime = datetime.now()
                
                # Extract real image from Medium post
                image_url = self._extract_medium_image(entry)
                
                # Extract reading time if available
                reading_time = self._extract_reading_time(entry)
                
                posts.append({
                    'platform': 'Medium',
                    'title': entry.title.strip(),
                    'url': entry.link,
                    'published': time_ago,
                    'published_date': pub_datetime,
                    'thumbnail': image_url,
                    'reading_time': reading_time,
                    'author': 'Selman Dedeakayoğulları'
                })
            
            # Cache results
            st.session_state[cache_key] = posts
            st.session_state[f"{cache_key}_timestamp"] = time.time()
            
            return posts
            
        except Exception as e:
            st.error(f"Error: {e}")
            return self._get_demo_posts()
    
    def _extract_medium_image(self, entry) -> str:
        """Extract real image from Medium RSS entry"""
        try:
            # Method 1: Check media_thumbnail
            if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
                return entry.media_thumbnail[0]['url']
            
            # Method 2: Check enclosures for images
            if hasattr(entry, 'enclosures') and entry.enclosures:
                for enclosure in entry.enclosures:
                    if 'image' in enclosure.get('type', ''):
                        return enclosure['href']
            
            # Method 3: Parse content for images
            content = entry.get('summary', '') or entry.get('content', [{}])[0].get('value', '')
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                img_tags = soup.find_all('img')
                for img in img_tags:
                    src = img.get('src', '')
                    if 'medium.com' in src and src.startswith('http'):
                        return src
            
            # Method 4: Try to get from Medium API-like URL
            post_id = self._extract_post_id_from_url(entry.link)
            if post_id:
                possible_img = f"https://miro.medium.com/v2/resize:fit:1200/1*{post_id}.jpeg"
                return possible_img
            
        except Exception as e:
            print(f"Error extracting image: {e}")
        
        return self._create_card_image(entry.title, 0)
    
    def _extract_post_id_from_url(self, url: str) -> Optional[str]:
        """Extract post ID from Medium URL"""
        try:
            import re
            match = re.search(r'-([a-f0-9]{12,})$', url)
            if match:
                return match.group(1)[:12]
        except:
            pass
        return None
    
    def _extract_reading_time(self, entry) -> str:
        """Extract reading time from Medium entry"""
        try:
            content = entry.get('summary', '') or entry.get('content', [{}])[0].get('value', '')
            if content:
                import re
                reading_time_match = re.search(r'(\d+)\s*min\s*read', content, re.IGNORECASE)
                if reading_time_match:
                    return f"{reading_time_match.group(1)} min"
            
            if content:
                word_count = len(content.split())
                estimated_time = max(1, word_count // 200)
                return f"{estimated_time} min"
        except:
            pass
        return "5 min"
    
    def _create_card_image(self, title: str, index: int) -> str:
        """Create beautiful gradient images"""
        gradients = [
            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", 
            "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
            "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
            "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)"
        ]
        
        icons = ["💻", "🚀", "⚡", "🎯", "🔥", "✨"]
        
        hash_val = abs(hash(title)) 
        gradient = gradients[hash_val % len(gradients)]
        icon = icons[hash_val % len(icons)]
        
        svg = f'''
        <svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="grad{hash_val}" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:rgb(102,126,234);stop-opacity:1" />
                    <stop offset="100%" style="stop-color:rgb(118,75,162);stop-opacity:1" />
                </linearGradient>
            </defs>
            <rect width="100%" height="100%" fill="url(#grad{hash_val})"/>
            <text x="50%" y="50%" text-anchor="middle" dy="0.3em" font-size="48" fill="white" opacity="0.9">{icon}</text>
            <rect width="100%" height="100%" fill="rgba(0,0,0,0.1)"/>
        </svg>
        '''
        
        import base64
        svg_bytes = svg.encode('utf-8')
        svg_base64 = base64.b64encode(svg_bytes).decode('utf-8')
        return f"data:image/svg+xml;base64,{svg_base64}"
    
    def _get_demo_posts(self) -> List[Dict[str, Any]]:
        """Demo posts if RSS fails"""
        return [
            {
                'platform': 'Medium',
                'title': 'Building AI Portfolio Assistant with RAG',
                'url': 'https://medium.com/@selmandedeakayogullari',
                'published': '2 days ago',
                'published_date': datetime.now() - timedelta(days=2),
                'thumbnail': self._create_card_image('AI Portfolio Assistant', 0),
                'reading_time': '5 min',
                'author': 'Selman Dedeakayoğulları'
            },
            {
                'platform': 'Medium', 
                'title': 'Advanced Python Techniques for Data Science',
                'url': 'https://medium.com/@selmandedeakayogullari',
                'published': '1 week ago',
                'published_date': datetime.now() - timedelta(weeks=1),
                'thumbnail': self._create_card_image('Python Data Science', 1),
                'reading_time': '7 min',
                'author': 'Selman Dedeakayoğulları'
            }
        ]
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check cache validity"""
        timestamp_key = f"{cache_key}_timestamp"
        if timestamp_key not in st.session_state:
            return False
        last_update = st.session_state[timestamp_key]
        return (time.time() - last_update) < self.cache_duration
    
    def _get_time_ago(self, pub_date: datetime) -> str:
        """Simple time ago calculation"""
        now = datetime.now()
        diff = now - pub_date
        
        if diff.days > 30:
            return f"{diff.days // 30}mo ago"
        elif diff.days > 7:
            return f"{diff.days // 7}w ago"
        elif diff.days > 0:
            return f"{diff.days}d ago"
        else:
            hours = diff.seconds // 3600
            return f"{hours}h ago" if hours > 0 else "now"
    
    def get_all_posts(self, limit_per_platform: int = 6) -> List[Dict[str, Any]]:
        """Get all posts"""
        return self.get_medium_posts(limit_per_platform)
    
    def render_posts_cards(self, posts: List[Dict[str, Any]], language: str = "en") -> None:
        """Render responsive cards with mobile optimization and dark mode support"""
        if not posts:
            st.info("📭 No posts found")
            return
        
        # Beautiful header with gradient
        header_text = "Latest Articles" if language == "en" else "Son Makaleler"
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 16px 0;
            text-align: center;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        ">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 20px;
                font-weight: 600;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            ">
                ✍️ {header_text}
            </h3>
            <p style="
                color: rgba(255,255,255,0.9);
                margin: 8px 0 0 0;
                font-size: 14px;
            ">
                Medium'da yayınlanan yazılarım
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Geliştirilmiş dark mode CSS
        st.markdown("""
    <style>
    .responsive-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 0;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
        overflow: hidden;
        border: 1px solid #e1e5e9;
    }

    /* Dark mode support - daha spesifik selectors */
    .stApp[data-theme="dark"] .responsive-card,
    [data-theme="dark"] .responsive-card,
    .stApp.dark-theme .responsive-card,
    .dark-theme .responsive-card,
    body[data-theme="dark"] .responsive-card,
    html[data-theme="dark"] .responsive-card {
        background: #1e1e1e !important;
        border: 1px solid #333 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    /* Media query ile dark mode algılama */
    @media (prefers-color-scheme: dark) {
        .responsive-card {
            background: #1e1e1e !important;
            border: 1px solid #333 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        }
    }

    .responsive-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    /* Dark mode hover effects */
    .stApp[data-theme="dark"] .responsive-card:hover,
    [data-theme="dark"] .responsive-card:hover,
    .stApp.dark-theme .responsive-card:hover,
    .dark-theme .responsive-card:hover,
    body[data-theme="dark"] .responsive-card:hover,
    html[data-theme="dark"] .responsive-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
    }

    @media (prefers-color-scheme: dark) {
        .responsive-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
        }
    }

    .card-img {
        width: 100%;
        height: 160px;
        object-fit: cover;
        transition: transform 0.3s ease;
    }

    .responsive-card:hover .card-img {
        transform: scale(1.05);
    }

    .card-body {
        padding: 16px;
    }

    .card-title {
        font-size: 16px;
        font-weight: 600;
        margin: 0 0 8px 0;
        color: #1a1a1a;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        word-wrap: break-word;
        hyphens: auto;
    }

    /* Dark mode text colors - daha güçlü selectors */
    .stApp[data-theme="dark"] .card-title,
    [data-theme="dark"] .card-title,
    .stApp.dark-theme .card-title,
    .dark-theme .card-title,
    body[data-theme="dark"] .card-title,
    html[data-theme="dark"] .card-title {
        color: #ffffff !important;
    }

    @media (prefers-color-scheme: dark) {
        .card-title {
            color: #ffffff !important;
        }
    }

    .card-meta {
        font-size: 14px;
        color: #666;
        margin: 8px 0;
    }

    /* Dark mode meta text */
    .stApp[data-theme="dark"] .card-meta,
    [data-theme="dark"] .card-meta,
    .stApp.dark-theme .card-meta,
    .dark-theme .card-meta,
    body[data-theme="dark"] .card-meta,
    html[data-theme="dark"] .card-meta {
        color: #cccccc !important;
    }

    @media (prefers-color-scheme: dark) {
        .card-meta {
            color: #cccccc !important;
        }
    }

    .read-link {
        display: block;
        width: 100%;
        background: #0066cc;
        color: white !important;
        padding: 12px 16px;
        border-radius: 6px;
        text-decoration: none !important;
        font-size: 14px;
        font-weight: 500;
        text-align: center;
        box-sizing: border-box;
        margin-top: 12px;
        border: none;
        outline: none;
    }

    .read-link:hover {
        background: #0052a3 !important;
        text-decoration: none !important;
        color: white !important;
    }

    .read-link:visited,
    .read-link:focus,
    .read-link:active {
        color: white !important;
        text-decoration: none !important;
    }

    /* Global link override for this component */
    .responsive-card a {
        text-decoration: none !important;
    }

    .responsive-card a:hover,
    .responsive-card a:visited,
    .responsive-card a:focus,
    .responsive-card a:active {
        text-decoration: none !important;
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
        .responsive-card {
            margin: 12px 0;
        }
        .card-img {
            height: 120px;
        }
        .card-body {
            padding: 12px;
        }
        .card-title {
            font-size: 15px;
            line-height: 1.3;
            -webkit-line-clamp: 3;
            max-height: 3.9em;
        }
        .card-meta {
            font-size: 13px;
            margin: 6px 0;
        }
        .read-link {
            padding: 10px 12px;
            font-size: 13px;
        }
    }

    /* Extra small screens */
    @media (max-width: 480px) {
        .responsive-card {
            margin: 8px 0;
            border-radius: 8px;
        }
        .card-img {
            height: 100px;
        }
        .card-body {
            padding: 10px;
        }
        .card-title {
            font-size: 14px;
            line-height: 1.25;
            margin: 0 0 6px 0;
        }
        .card-meta {
            font-size: 12px;
            margin: 6px 0 8px 0;
        }
        .read-link {
            padding: 8px 12px;
            font-size: 12px;
            margin-top: 8px;
        }
    }

    /* Mobile single column layout */
    @media (max-width: 480px) {
        .element-container .stColumns {
            flex-direction: column !important;
        }
        .element-container .stColumns > div {
            width: 100% !important;
            margin-right: 0 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
        
        # Rest of the function remains the same...
        col1, col2 = st.columns(2, gap="medium")
        
        for i, post in enumerate(posts):
            col = col1 if i % 2 == 0 else col2
            
            with col:
                read_text = "Read Article" if language == "en" else "Makaleyi Oku"
                safe_title = post['title'].replace('"', '&quot;').replace("'", '&#39;')
                
                card_html = f"""
                <div class="responsive-card">
                    <img src="{post['thumbnail']}" 
                        class="card-img" 
                        alt="{safe_title}"
                        loading="lazy"
                        onerror="this.src='data:image/svg+xml;base64,{self._create_fallback_image()}'">
                    <div class="card-body">
                        <h4 class="card-title" title="{safe_title}">{post['title']}</h4>
                        <p class="card-meta">📝 {post['platform']} • {post['published']} • {post['reading_time']}</p>
                        <a href="{post['url']}" target="_blank" class="read-link">{read_text} →</a>
                    </div>
                </div>
                """
                
                st.markdown(card_html, unsafe_allow_html=True)
    
    def _create_fallback_image(self) -> str:
        """Create fallback image"""
        svg = '''
        <svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f0f2f5"/>
            <text x="50%" y="50%" text-anchor="middle" dy="0.3em" font-size="24" fill="#666">📄</text>
        </svg>
        '''
        import base64
        return base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    
    def format_posts_for_chat(self, posts: List[Dict[str, Any]], language: str = "en") -> str:
        """Simple chat format"""
        if not posts:
            return "📭 No posts found"
        
        title = "✍️ **Latest Articles**\n\n" if language == "en" else "✍️ **Son Makaleler**\n\n"
        formatted = title
        
        for i, post in enumerate(posts[:4], 1):
            formatted += f"**{i}. {post['title']}**\n"
            formatted += f"📝 {post['platform']} • {post['published']} • {post['reading_time']}\n"
            formatted += f"[🔗 Read Article]({post['url']})\n\n"
        
        return formatted
    
    def get_post_summary(self, query: str, posts: List[Dict[str, Any]], language: str = "en") -> str:
        """Simple search"""
        if not posts:
            return "📭 No posts available"
        return self.format_posts_for_chat(posts[:3], language)
    
    def clear_cache(self) -> None:
        """Clear cache"""
        keys_to_remove = [key for key in st.session_state.keys() 
                         if key.startswith('medium_posts_')]
        for key in keys_to_remove:
            del st.session_state[key]