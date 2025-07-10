# utils/config.py

# Color palette for consistent branding
COLORS = {
    "primary": "#0072B2",   # Blue
    "secondary": "#009E73", # Green
    "accent": "#D55E00",    # Orange
    "neutral": "#F0E442",   # Yellow
    "background": "#FAFAFA",
    "text": "#333333",
    "light_gray": "#DDDDDD",
    "dark_gray": "#555555"
}

def get_custom_css():
    """Returns custom CSS for a professional look and feel."""
    return f"""
    <style>
        /* Main app styling */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        /* Sidebar styling */
        .st-emotion-cache-16txtl3 {{
            background-color: #FFFFFF;
        }}
        /* Titles and headers */
        h1, h2, h3 {{
            color: {COLORS['dark_gray']};
        }}
        /* Custom info/success boxes */
        .st-emotion-cache-1wivap2 {{
            border-left: 5px solid {COLORS['primary']};
            background-color: #E6F7FF;
        }}
        .st-emotion-cache-1wivap2 .st-emotion-cache-1ghh1go {{
            color: {COLORS['dark_gray']};
        }}
        .stAlert[data-baseweb="alert"] {{
            border-radius: 5px;
        }}
        .st-emotion-cache-1aehpv3 {{
             border-left: 5px solid {COLORS['secondary']};
             background-color: #E6FFFA;
        }}
    </style>
    """
