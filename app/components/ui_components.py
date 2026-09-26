import sys
from pathlib import Path
import streamlit as st

def apply_custom_theme():
    """
    Applies SmartRoute-AI (NCKH 2026) inspired UI theme with ultra-high CSS specificity:
    - Completely eradicates all Streamlit red/pink underline indicators across all shadow DOM trees
    - Full 9999px Pill Navigation Buttons with Emerald Green active state (#059669)
    - Styled Segmented Radio Buttons for Model & Input Mode Selection
    - Icon Circle Headers for Hero Banners & Section Cards
    """
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }

            /* Hide Sidebar Completely for Top-Header Web App Layout */
            [data-testid="stSidebar"] {
                display: none !important;
            }
            [data-testid="stSidebarNav"] {
                display: none !important;
            }
            .stAppHeader {
                display: none !important;
            }
            
            /* Main Content Container Padding & Width */
            div.block-container {
                padding-top: 0.8rem !important;
                padding-bottom: 3rem !important;
                max-width: 1260px !important;
            }

            /* Global Background */
            .stApp {
                background-color: #F8FAFC;
            }

            /* Top Main Header Bar (SmartRoute-AI Style) */
            .top-header-bar {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                padding: 12px 24px;
                margin-top: -0.5rem;
                margin-bottom: 20px;
                border-radius: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 4px 15px -2px rgba(5, 150, 105, 0.05);
            }

            .top-header-left {
                display: flex;
                align-items: center;
                gap: 14px;
            }

            .top-header-icon {
                width: 44px;
                height: 44px;
                background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.4rem;
                color: #FFFFFF;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
            }

            .top-header-title-box {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .top-header-logo {
                font-size: 1.35rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                color: #0F172A;
            }

            .top-header-logo span {
                color: #059669;
            }

            .top-header-badge {
                background-color: #ECFDF5;
                color: #059669;
                border: 1px solid #A7F3D0;
                font-size: 0.72rem;
                font-weight: 800;
                padding: 2px 10px;
                border-radius: 9999px;
                letter-spacing: 0.04em;
            }

            .top-header-sublogo {
                font-size: 0.8rem;
                color: #64748B;
                font-weight: 500;
                margin-top: 1px;
            }

            .top-header-right {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .status-badge-yellow {
                background-color: #FEF3C7;
                color: #92400E;
                border: 1px solid #FDE68A;
                padding: 6px 14px;
                border-radius: 9999px;
                font-size: 0.78rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .status-badge-green {
                background-color: #ECFDF5;
                color: #059669;
                border: 1px solid #A7F3D0;
                padding: 6px 14px;
                border-radius: 9999px;
                font-size: 0.78rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                background-color: #10B981;
                border-radius: 50%;
                display: inline-block;
            }

            /* ERADICATE Streamlit's Default Red/Pink/Orange Active Tab Line & Borders Permanently */
            html body .stApp [data-baseweb="tab-highlight-title"],
            html body .stApp [data-baseweb="tab-border"],
            html body .stApp [data-baseweb="tab-highlight"],
            html body .stApp .stTabs [data-baseweb="tab-highlight-title"],
            html body .stApp .stTabs [data-baseweb="tab-border"],
            html body .stApp .stTabs [data-baseweb="tab-highlight"],
            html body .stApp div[data-baseweb="tab-list"]::after,
            html body .stApp button[data-baseweb="tab"]::after,
            html body .stApp div[data-baseweb="tab-list"] > div {
                display: none !important;
                opacity: 0 !important;
                height: 0px !important;
                min-height: 0px !important;
                max-height: 0px !important;
                width: 0px !important;
                border: none !important;
                border-bottom: none !important;
                background: transparent !important;
                background-color: transparent !important;
                visibility: hidden !important;
            }

            /* Streamlit Tabs Container & 100% Full Pill Navigation Styling */
            html body .stApp .stTabs [data-baseweb="tab-list"],
            html body .stApp div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 8px !important;
                background-color: transparent !important;
                padding-bottom: 6px !important;
                border-bottom: none !important;
            }

            html body .stApp .stTabs [data-baseweb="tab"],
            html body .stApp .stTabs button,
            html body .stApp .stTabs [role="tab"],
            html body .stApp div[data-testid="stTabs"] button,
            html body .stApp div[data-testid="stTabs"] [data-baseweb="tab"],
            html body .stApp div[data-testid="stTabs"] [role="tab"],
            html body .stApp button[data-baseweb="tab"],
            html body .stApp button[role="tab"] {
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                border-radius: 9999px !important;
                border-top-left-radius: 9999px !important;
                border-top-right-radius: 9999px !important;
                border-bottom-left-radius: 9999px !important;
                border-bottom-right-radius: 9999px !important;
                padding: 8px 22px !important;
                font-size: 0.88rem !important;
                font-weight: 600 !important;
                color: #334155 !important;
                transition: all 0.2s ease !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
                margin-bottom: 2px !important;
                outline: none !important;
                overflow: hidden !important;
                -webkit-border-radius: 9999px !important;
                -moz-border-radius: 9999px !important;
            }

            html body .stApp .stTabs [data-baseweb="tab"]:hover,
            html body .stApp .stTabs button:hover,
            html body .stApp .stTabs [role="tab"]:hover,
            html body .stApp div[data-testid="stTabs"] button:hover,
            html body .stApp button[data-baseweb="tab"]:hover {
                background-color: #ECFDF5 !important;
                color: #059669 !important;
                border-color: #A7F3D0 !important;
                border-radius: 9999px !important;
                border-top-left-radius: 9999px !important;
                border-top-right-radius: 9999px !important;
                border-bottom-left-radius: 9999px !important;
                border-bottom-right-radius: 9999px !important;
            }

            html body .stApp .stTabs [aria-selected="true"],
            html body .stApp .stTabs button[aria-selected="true"],
            html body .stApp .stTabs [data-baseweb="tab"][aria-selected="true"],
            html body .stApp div[data-testid="stTabs"] button[aria-selected="true"],
            html body .stApp div[data-testid="stTabs"] [aria-selected="true"],
            html body .stApp button[aria-selected="true"] {
                background-color: #059669 !important;
                color: #FFFFFF !important;
                border-color: #059669 !important;
                font-weight: 700 !important;
                box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
                border-radius: 9999px !important;
                border-top-left-radius: 9999px !important;
                border-top-right-radius: 9999px !important;
                border-bottom-left-radius: 9999px !important;
                border-bottom-right-radius: 9999px !important;
            }

            html body .stApp .stTabs [aria-selected="true"] p,
            html body .stApp .stTabs [aria-selected="true"] span,
            html body .stApp .stTabs [aria-selected="true"] div,
            html body .stApp button[aria-selected="true"] p,
            html body .stApp button[aria-selected="true"] span,
            html body .stApp button[aria-selected="true"] div {
                color: #FFFFFF !important;
            }

            /* Segmented Control Styling for Radio Buttons (SmartRoute-AI Style) */
            div[data-testid="stRadio"] > label {
                font-size: 0.88rem !important;
                font-weight: 700 !important;
                color: #0F172A !important;
                margin-bottom: 8px !important;
            }

            div[data-testid="stRadio"] [role="radiogroup"] {
                gap: 8px !important;
                background-color: #FFFFFF !important;
                padding: 6px !important;
                border-radius: 9999px !important;
                border: 1px solid #E2E8F0 !important;
                display: inline-flex !important;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
            }

            /* Hero Banner Card with Icon Circle Header */
            .hero-container {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 22px 28px;
                color: #0F172A;
                margin-bottom: 20px;
                box-shadow: 0 6px 20px -4px rgba(0, 0, 0, 0.04);
            }

            .hero-header-flex {
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 8px;
            }

            .hero-icon-circle {
                width: 44px;
                height: 44px;
                background-color: #ECFDF5;
                border: 1px solid #A7F3D0;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.35rem;
                color: #059669;
                flex-shrink: 0;
            }

            .hero-badge {
                display: inline-block;
                background-color: #ECFDF5;
                border: 1px solid #A7F3D0;
                color: #059669;
                padding: 2px 10px;
                border-radius: 9999px;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 4px;
            }

            .hero-title {
                font-size: 1.65rem;
                font-weight: 800;
                color: #0F172A;
                line-height: 1.3;
                letter-spacing: -0.01em;
            }

            .hero-subtitle {
                font-size: 0.92rem;
                color: #64748B;
                max-width: 900px;
                line-height: 1.5;
                margin-top: 4px;
            }

            /* Metric Cards */
            .metric-card {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 14px;
                padding: 18px 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
            }
            
            .metric-card-title {
                font-size: 0.78rem;
                font-weight: 700;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 6px;
            }
            
            .metric-card-value {
                font-size: 1.85rem;
                font-weight: 800;
                color: #0F172A;
            }

            .metric-card-subtitle {
                font-size: 0.78rem;
                color: #059669;
                font-weight: 600;
                margin-top: 4px;
            }

            /* Prediction Result Card */
            .result-card {
                background-color: #FFFFFF;
                border: 1px solid #A7F3D0;
                border-left: 6px solid #059669;
                border-radius: 14px;
                padding: 20px 24px;
                margin-bottom: 16px;
                box-shadow: 0 4px 15px rgba(5, 150, 105, 0.06);
            }

            .result-title {
                font-size: 1.5rem;
                font-weight: 800;
                color: #0F172A;
                margin-bottom: 6px;
            }

            .result-detail {
                font-size: 0.92rem;
                color: #475569;
                margin-bottom: 4px;
            }

            /* Section Header */
            .section-header {
                font-size: 1.2rem;
                font-weight: 700;
                color: #0F172A;
                border-bottom: 2px solid #E2E8F0;
                padding-bottom: 8px;
                margin-top: 24px;
                margin-bottom: 16px;
            }

            /* Styled Alerts (SmartRoute-AI Style) */
            .alert-info-box {
                background-color: #F0FDF4;
                border: 1px solid #BBF7D0;
                border-left: 4px solid #16A34A;
                color: #14532D;
                padding: 14px 18px;
                border-radius: 12px;
                margin-bottom: 16px;
                font-size: 0.92rem;
            }

            .alert-warning-box {
                background-color: #FFFBEB;
                border: 1px solid #FDE68A;
                border-left: 4px solid #D97706;
                color: #92400E;
                padding: 14px 18px;
                border-radius: 12px;
                margin-bottom: 16px;
                font-size: 0.92rem;
            }

            .alert-success-box {
                background-color: #ECFDF5;
                border: 1px solid #A7F3D0;
                border-left: 4px solid #059669;
                color: #065F46;
                padding: 14px 18px;
                border-radius: 12px;
                margin-bottom: 16px;
                font-size: 0.92rem;
            }

            /* Primary Button Styling */
            .stButton > button {
                background-color: #059669 !important;
                color: #FFFFFF !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 10px 24px !important;
                font-weight: 700 !important;
                transition: all 0.2s ease !important;
            }

            .stButton > button:hover {
                background-color: #047857 !important;
                box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
            }

            /* SVG Vector Icon Helper Styling */
            .top-header-icon svg {
                width: 24px;
                height: 24px;
                stroke: #FFFFFF;
            }
            .hero-icon-circle svg {
                width: 22px;
                height: 22px;
                stroke: #059669;
            }
        </style>
    """, unsafe_allow_html=True)

SVG_ICONS = {
    # Traffic Warning Sign Icon
    "traffic": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    
    # Target / Prediction Icon
    "target": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/><line x1="12" y1="2" x2="12" y2="4"/><line x1="12" y1="20" x2="12" y2="22"/><line x1="2" y1="12" x2="4" y2="12"/><line x1="20" y1="12" x2="22" y2="12"/></svg>',
    
    # Dataset / Database Icon
    "dataset": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    
    # Metrics / Evaluation Chart Icon
    "metrics": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
    
    # Model Comparison / Scale Icon
    "compare": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 16l3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1z"/><path d="M2 16l3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/></svg>',
    
    # History / Log Icon
    "history": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>'
}


def render_top_header():
    traffic_svg = SVG_ICONS["traffic"]
    st.markdown(f"""
        <div class="top-header-bar">
            <div class="top-header-left">
                <div class="top-header-icon">{traffic_svg}</div>
                <div>
                    <div class="top-header-title-box">
                        <div class="top-header-logo">Sigange<span>-AI</span></div>
                        <div class="top-header-badge">KHOA CNTT</div>
                    </div>
                    <div class="top-header-sublogo">Hệ thống Nhận diện & Phân loại Biển báo Giao thông Tự động</div>
                </div>
            </div>
            <div class="top-header-right">
                <div class="status-badge-yellow">Ghi chú học thuật</div>
                <div class="status-badge-green"><span class="status-dot"></span> AI Engine Ready</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, badge_text: str = "HỆ THỐNG PHÂN LOẠI BIỂN BÁO GIAO THÔNG", icon: str = "traffic", **kwargs):
    icon_html = SVG_ICONS.get(icon, icon) if icon in SVG_ICONS else (SVG_ICONS.get(kwargs.get('icon_name', ''), icon) if not str(icon).startswith('<svg') else icon)
    if not icon_html or not str(icon_html).startswith('<svg'):
        icon_html = SVG_ICONS["traffic"]

    st.markdown(f"""
        <div class="hero-container">
            <div class="hero-header-flex">
                <div class="hero-icon-circle">{icon_html}</div>
                <div>
                    <div class="hero-badge">{badge_text}</div>
                    <div class="hero-title">{title}</div>
                </div>
            </div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = ""):
    subtitle_html = f'<div class="metric-card-subtitle">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">{title}</div>
            <div class="metric-card-value">{value}</div>
            {subtitle_html}
        </div>
    """, unsafe_allow_html=True)


def render_alert(text: str, alert_type: str = "info"):
    css_class = f"alert-{alert_type}-box"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)
