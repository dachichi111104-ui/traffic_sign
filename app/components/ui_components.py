import sys
from pathlib import Path
import streamlit as st

def apply_custom_theme():
    """
    Applies SmartRoute-AI (NCKH 2026) inspired UI theme:
    - Removes all default Streamlit red/pink underline indicators
    - Rounded 9999px Full Pill Buttons for Horizontal Tabs
    - Clean Header Bar with Status Pills & AI Engine Badge
    - Ultra-clean Cards with Icon Circle Headers (#ECFDF5 / #059669)
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

            /* REMOVE Streamlit's Default Red/Pink Underline Active Tab Line */
            div[data-baseweb="tab-highlight-title"] {
                display: none !important;
                background-color: transparent !important;
                height: 0px !important;
            }
            
            div[data-baseweb="tab-border"] {
                display: none !important;
                height: 0px !important;
            }

            /* Streamlit Tabs - Pill Shape (SmartRoute-AI Navigation Steps) */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px !important;
                background-color: transparent !important;
                padding-bottom: 8px !important;
                border-bottom: none !important;
            }

            .stTabs [data-baseweb="tab"] {
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                border-radius: 9999px !important;
                padding: 8px 20px !important;
                font-size: 0.88rem !important;
                font-weight: 600 !important;
                color: #334155 !important;
                transition: all 0.2s ease !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
            }

            .stTabs [data-baseweb="tab"]:hover {
                background-color: #ECFDF5 !important;
                color: #059669 !important;
                border-color: #A7F3D0 !important;
            }

            .stTabs [aria-selected="true"] {
                background-color: #059669 !important;
                color: #FFFFFF !important;
                border-color: #059669 !important;
                font-weight: 700 !important;
                box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
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
        </style>
    """, unsafe_allow_html=True)


def render_top_header():
    st.markdown("""
        <div class="top-header-bar">
            <div class="top-header-left">
                <div class="top-header-icon">🚦</div>
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


def render_hero_banner(title: str, subtitle: str, badge_text: str = "HỆ THỐNG PHÂN LOẠI BIỂN BÁO GIAO THÔNG", icon: str = "🚦"):
    st.markdown(f"""
        <div class="hero-container">
            <div class="hero-header-flex">
                <div class="hero-icon-circle">{icon}</div>
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
