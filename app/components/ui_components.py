import sys
from pathlib import Path
import streamlit as st

def apply_custom_theme():
    """
    Applies custom high-end Academic & Professional CSS styling:
    - Hides default Streamlit sidebar completely to match top-header web apps (ProjectHub / FoodGo)
    - Deep Navy Header (#0F172A to #1E3A8A)
    - Clean Sans-Serif typography (Inter)
    - Soft Shadow Cards with Accent Borders
    - ZERO EMOJIS, pure professional presentation
    """
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
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
                padding-top: 1rem !important;
                padding-bottom: 3rem !important;
                max-width: 1240px !important;
            }

            /* Global Background */
            .stApp {
                background-color: #F8FAFC;
            }

            /* Top Main Header Bar (ProjectHub Style) */
            .top-header-bar {
                background-color: #0F172A;
                border-bottom: 3px solid #2563EB;
                padding: 14px 28px;
                margin-top: -1rem;
                margin-bottom: 24px;
                border-radius: 0 0 10px 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                color: #FFFFFF;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
            }

            .top-header-logo {
                font-size: 1.15rem;
                font-weight: 800;
                letter-spacing: 0.05em;
                color: #FFFFFF;
                text-transform: uppercase;
            }

            .top-header-sublogo {
                font-size: 0.78rem;
                color: #94A3B8;
                font-weight: 500;
            }

            .top-header-user {
                background-color: #1E293B;
                border: 1px solid #334155;
                padding: 6px 16px;
                border-radius: 6px;
                font-size: 0.82rem;
                font-weight: 600;
                color: #F8FAFC;
            }

            /* Hero Banner */
            .hero-container {
                background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
                border-radius: 10px;
                padding: 28px 32px;
                color: #FFFFFF;
                margin-bottom: 24px;
                box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.15);
            }

            .hero-badge {
                display: inline-block;
                background-color: rgba(59, 130, 246, 0.2);
                border: 1px solid rgba(147, 197, 253, 0.4);
                color: #93C5FD;
                padding: 3px 12px;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 10px;
            }

            .hero-title {
                font-size: 2.1rem;
                font-weight: 800;
                color: #FFFFFF;
                margin-bottom: 6px;
                line-height: 1.25;
            }

            .hero-subtitle {
                font-size: 0.98rem;
                color: #CBD5E1;
                max-width: 850px;
                line-height: 1.5;
            }

            /* Metric Cards */
            .metric-card {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 18px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
                text-align: center;
            }
            
            .metric-card-title {
                font-size: 0.8rem;
                font-weight: 700;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 6px;
            }
            
            .metric-card-value {
                font-size: 1.8rem;
                font-weight: 800;
                color: #0F172A;
            }

            .metric-card-subtitle {
                font-size: 0.78rem;
                color: #2563EB;
                font-weight: 600;
                margin-top: 4px;
            }

            /* Prediction Result Card */
            .result-card {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-left: 6px solid #2563EB;
                border-radius: 8px;
                padding: 18px 24px;
                margin-bottom: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
            }

            .result-title {
                font-size: 1.45rem;
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
                font-size: 1.25rem;
                font-weight: 700;
                color: #0F172A;
                border-bottom: 2px solid #E2E8F0;
                padding-bottom: 8px;
                margin-top: 24px;
                margin-bottom: 16px;
            }

            /* Styled Alerts (No Emoji) */
            .alert-info-box {
                background-color: #EFF6FF;
                border: 1px solid #BFDBFE;
                border-left: 4px solid #2563EB;
                color: #1E40AF;
                padding: 14px 18px;
                border-radius: 6px;
                margin-bottom: 16px;
                font-size: 0.92rem;
            }

            .alert-warning-box {
                background-color: #FFFBEB;
                border: 1px solid #FDE68A;
                border-left: 4px solid #D97706;
                color: #92400E;
                padding: 14px 18px;
                border-radius: 6px;
                margin-bottom: 16px;
                font-size: 0.92rem;
            }

            .alert-success-box {
                background-color: #ECFDF5;
                border: 1px solid #A7F3D0;
                border-left: 4px solid #059669;
                color: #065F46;
                padding: 14px 18px;
                border-radius: 6px;
                margin-bottom: 16px;
                font-size: 0.92rem;
            }
        </style>
    """, unsafe_allow_html=True)


def render_top_header():
    st.markdown("""
        <div class="top-header-bar">
            <div>
                <div class="top-header-logo">SIGANGE — TRAFFIC SIGN RECOGNITION</div>
                <div class="top-header-sublogo">KHOA CÔNG NGHỆ THÔNG TIN / ĐIỆN ĐIỆN TỬ — HỌC PHẦN MÁY HỌC</div>
            </div>
            <div class="top-header-user">
                BÁO CÁO DỰ ÁN MÁY HỌC
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, badge_text: str = "HỆ THỐNG PHÂN LOẠI BIỂN BÁO GIAO THÔNG"):
    st.markdown(f"""
        <div class="hero-container">
            <div class="hero-badge">{badge_text}</div>
            <div class="hero-title">{title}</div>
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
