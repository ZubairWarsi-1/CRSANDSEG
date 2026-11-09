"""
Customer360 - UI Components and Theming Module
Provides consistent professional design, custom CSS, KPI metric cards, and Plotly chart themes.
"""

from typing import Optional, Dict, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# Modern corporate palette
COLOR_PRIMARY = "#2563EB"       # Indigo / Royal Blue
COLOR_SECONDARY = "#0EA5E9"     # Cyan / Sky
COLOR_ACCENT = "#6366F1"        # Purple
COLOR_SUCCESS = "#10B981"       # Emerald
COLOR_WARNING = "#F59E0B"       # Amber
COLOR_DANGER = "#EF4444"        # Rose / Red
COLOR_NEUTRAL_DARK = "#0F172A"  # Slate 900
COLOR_NEUTRAL_LIGHT = "#F8FAFC" # Slate 50
COLOR_CARD_BORDER = "#E2E8F0"   # Slate 200


def apply_custom_css():
    """
    Inject custom styles for an executive, modern SaaS feel.
    """
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            html, body, [class*="css"] {{
                font-family: 'Inter', sans-serif;
            }}
            
            .stApp {{
                background-color: #F8FAFC;
                color: #0F172A;
            }}

            /* Header Title Styling */
            .main-header {{
                background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
                padding: 1.5rem 2rem;
                border-radius: 12px;
                color: #FFFFFF;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            }}
            .main-header h1 {{
                font-size: 1.8rem;
                font-weight: 700;
                margin: 0;
                color: #FFFFFF;
            }}
            .main-header p {{
                font-size: 0.95rem;
                margin: 0.25rem 0 0 0;
                color: #E2E8F0;
            }}

            /* Modern KPI Cards */
            .kpi-container {{
                background: #FFFFFF;
                border: 1px solid {COLOR_CARD_BORDER};
                border-radius: 10px;
                padding: 1.2rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.03);
                transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
            }}
            .kpi-container:hover {{
                box-shadow: 0 4px 10px rgba(0,0,0,0.06);
            }}
            .kpi-label {{
                font-size: 0.85rem;
                font-weight: 600;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.35rem;
            }}
            .kpi-value {{
                font-size: 1.85rem;
                font-weight: 700;
                color: #0F172A;
                line-height: 1.2;
            }}
            .kpi-subtext {{
                font-size: 0.8rem;
                color: #94A3B8;
                margin-top: 0.3rem;
            }}

            /* Badges */
            .badge {{
                display: inline-block;
                padding: 0.2rem 0.6rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 600;
            }}
            .badge-success {{ background-color: #DEF7EC; color: #03543F; }}
            .badge-warning {{ background-color: #FEF3C7; color: #92400E; }}
            .badge-danger {{ background-color: #FDE8E8; color: #9B1C1C; }}
            .badge-info {{ background-color: #E1EFFE; color: #1E429F; }}

            /* Sidebar custom brand */
            .sidebar-brand {{
                padding: 1rem 0;
                border-bottom: 1px solid #E2E8F0;
                margin-bottom: 1rem;
            }}
            .sidebar-title {{
                font-size: 1.25rem;
                font-weight: 700;
                color: #1E3A8A;
                margin: 0;
            }}
            .sidebar-subtitle {{
                font-size: 0.8rem;
                color: #64748B;
                margin: 0;
            }}

            /* Section cards */
            .content-card {{
                background: #FFFFFF;
                border: 1px solid {COLOR_CARD_BORDER};
                border-radius: 10px;
                padding: 1.25rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


def render_page_header(title: str, subtitle: str, badge: Optional[str] = None):
    """
    Renders a unified modern page header banner.
    """
    badge_html = f'<span class="badge badge-info" style="margin-left: 10px; vertical-align: middle;">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{title} {badge_html}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_kpi(label: str, value: Any, subtext: str = "", delta: Optional[str] = None, delta_color: str = "normal"):
    """
    Render a clean KPI tile.
    """
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_plotly_layout(title: str = "", height: int = 400) -> Dict[str, Any]:
    """
    Returns standard styling attributes for interactive Plotly charts.
    """
    return {
        "title": {"text": title, "font": {"family": "Inter, sans-serif", "size": 15, "color": "#0F172A"}},
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": {"family": "Inter, sans-serif", "color": "#475569"},
        "margin": {"l": 40, "r": 30, "t": 50, "b": 40},
        "height": height,
        "xaxis": {
            "gridcolor": "#F1F5F9",
            "linecolor": "#E2E8F0",
            "zerolinecolor": "#E2E8F0"
        },
        "yaxis": {
            "gridcolor": "#F1F5F9",
            "linecolor": "#E2E8F0",
            "zerolinecolor": "#E2E8F0"
        },
        "legend": {"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5}
    }


def init_session_state():
    """
    Initialize global state across Streamlit pages.
    """
    if "current_dataset" not in st.session_state:
        # Load default Telco dataset
        from src.data_loader import get_sample_telco_data
        df, is_syn = get_sample_telco_data()
        st.session_state["current_dataset"] = df
        st.session_state["dataset_name"] = "IBM Telco Customer Churn"
        st.session_state["is_synthetic"] = is_syn
        st.session_state["dataset_type"] = "churn" # 'churn' or 'transaction'

    if "rfm_data" not in st.session_state:
        st.session_state["rfm_data"] = None

    if "kmeans_data" not in st.session_state:
        st.session_state["kmeans_data"] = None

    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}

    if "active_model_name" not in st.session_state:
        st.session_state["active_model_name"] = None

    if "retention_actions" not in st.session_state:
        st.session_state["retention_actions"] = {}

    if "predictions_df" not in st.session_state:
        st.session_state["predictions_df"] = None


def render_sidebar():
    """
    Render standard sidebar branding and dataset selector.
    """
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-title">⚡ Customer360</div>
                <div class="sidebar-subtitle">Customer Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("Active Dataset")
        curr_name = st.session_state.get("dataset_name", "Unknown")
        is_syn = st.session_state.get("is_synthetic", False)
        ds_source = st.session_state.get("dataset_source_class", "Synthetic Test Data" if is_syn else "Benchmark Sample Dataset")
        
        st.info(f"**{curr_name}**\n{len(st.session_state.get('current_dataset', [])):,} records\n\n*{ds_source}*")

        # Quick switcher between pre-loaded datasets
        ds_choice = st.selectbox(
            "Switch Demo Dataset",
            ["IBM Telco Churn (Subscription)", "Retail Transactions (RFM)", "Keep Current"],
            index=2
        )

        if ds_choice == "IBM Telco Churn (Subscription)":
            from src.data_loader import get_sample_telco_data
            df, is_syn = get_sample_telco_data()
            st.session_state["current_dataset"] = df
            st.session_state["dataset_name"] = "IBM Telco Customer Churn"
            st.session_state["is_synthetic"] = is_syn
            st.session_state["dataset_type"] = "churn"
            st.session_state["predictions_df"] = None
            st.rerun()

        elif ds_choice == "Retail Transactions (RFM)":
            from src.data_loader import get_sample_retail_data
            df, is_syn = get_sample_retail_data()
            st.session_state["current_dataset"] = df
            st.session_state["dataset_name"] = "Retail Transactions"
            st.session_state["is_synthetic"] = is_syn
            st.session_state["dataset_type"] = "transaction"
            st.session_state["predictions_df"] = None
            st.rerun()

        st.divider()
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #94A3B8; text-align: center;">
                Customer360 v1.0.0<br>Production-Grade ML Intelligence
            </div>
            """,
            unsafe_allow_html=True
        )

