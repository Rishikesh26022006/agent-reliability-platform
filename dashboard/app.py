"""
Streamlit dashboard — Agent Reliability Monitor
Shows real-time risk scores, failure categories, and trajectory viewer.
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import requests

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Reliability Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
.metric-card {
    background: linear-gradient(135deg, #1e2330, #252d3d);
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #2d3748;
    margin-bottom: 12px;
}
.risk-high   { color: #fc8181; font-weight: 700; }
.risk-medium { color: #fbd38d; font-weight: 700; }
.risk-low    { color: #68d391; font-weight: 700; }
.traj-row:hover { background: #1a202c; }
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://em-content.zobj.net/source/google/387/shield_1f6e1-fe0f.png", width=64)
st.sidebar.title("Reliability Monitor")
st.sidebar.caption("AI Agent self-healing dashboard")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🎯 Live Test", "📋 Trajectories", "🔬 Trajectory Detail"])
refresh = st.sidebar.button("🔄 Refresh")

# ── Helper ────────────────────────────────────────────────────────────────────
def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=8)
        return r.json()
    except Exception as e:
        st.error(f"API unreachable: {e}\n\nMake sure the FastAPI server is running:\n`C:\\arp_venv\\Scripts\\uvicorn api.main:app --reload`")
        return None


def risk_badge(score: float) -> str:
    if score is None:
        return "—"
    if score >= 0.65:
        return f'<span class="risk-high">🔴 {score:.3f}</span>'
    if score >= 0.35:
        return f'<span class="risk-medium">🟡 {score:.3f}</span>'
    return f'<span class="risk-low">🟢 {score:.3f}</span>'


# ── Overview ─────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Agent Reliability Overview")

    data = api_get("/metrics")
    if data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trajectories", data["total_trajectories"])
        col2.metric("Labeled", data["labeled_trajectories"])
        col3.metric("Failures Found", data["failure_count"])
        col4.metric("Failure Rate", f"{data['failure_rate']*100:.1f}%")

        st.divider()
        st.subheader("Failure Category Breakdown")
        cats = data.get("failure_categories", {})
        if cats:
            df = pd.DataFrame(list(cats.items()), columns=["Category", "Count"]).sort_values("Count", ascending=False)
            st.bar_chart(df.set_index("Category"))
        else:
            st.info("No labeled failures yet.")

    # Recent trajectories mini-feed
    st.divider()
    st.subheader("Recent Trajectories")
    traj_data = api_get("/trajectories?limit=10")
    if traj_data:
        rows = traj_data.get("trajectories", [])
        if rows:
            df = pd.DataFrame(rows)[["trajectory_id", "query", "timestamp", "step_count", "label_success", "label_category"]]
            df.columns = ["ID", "Query", "Timestamp", "Steps", "Success", "Category"]
            df["Query"] = df["Query"].str[:60] + "..."
            st.dataframe(df, use_container_width=True, hide_index=True)


# ── Live Test ─────────────────────────────────────────────────────────────────
elif page == "🎯 Live Test":
    st.title("🎯 Live Agent Test with Guardrails")
    st.caption("Send a support query through the full reliability pipeline")

    presets = [
        "My order 8891 arrived broken. I'd like a refund please.",
        "I've been waiting 3 weeks for my order 8891 and it's still not here!",
        "I changed my mind about my purchase. Can I get a refund?",
        "I think order 4432 was placed fraudulently on my account.",
        "What's the status of my order 4432?",
    ]
    preset = st.selectbox("Quick presets", ["— custom —"] + presets)
    query = st.text_area("Query", value=preset if preset != "— custom —" else "", height=80)

    if st.button("▶ Run Agent", type="primary") and query.strip():
        with st.spinner("Running agent with reliability guardrails..."):
            try:
                t0 = time.time()
                resp = requests.post(f"{API_BASE}/handle_ticket", json={"query": query}, timeout=60)
                result = resp.json()
            except Exception as e:
                st.error(str(e))
                result = None

        if result:
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{result['risk_score']:.3f}")
            col2.metric("Confidence", result["risk_confidence"].upper())
            col3.metric("Correction Triggered", "✅ Yes" if result["correction_triggered"] else "❌ No")

            st.markdown("**Agent Response:**")
            st.info(result["final_response"])

            if result["corrections"]:
                st.markdown("**Corrections Triggered:**")
                for c in result["corrections"]:
                    st.warning(f"**{c['strategy'].upper()}** (risk={c['risk_score']:.3f}) — {c['message']}")

            with st.expander("Full result JSON"):
                st.json(result)


# ── Trajectories ──────────────────────────────────────────────────────────────
elif page == "📋 Trajectories":
    st.title("📋 All Trajectories")
    limit = st.slider("Show last N trajectories", 10, 100, 50)
    data = api_get(f"/trajectories?limit={limit}")
    if data:
        rows = data.get("trajectories", [])
        if rows:
            df = pd.DataFrame(rows)
            df["Query"] = df["query"].str[:70] + "..."
            df = df[["trajectory_id", "Query", "timestamp", "step_count", "label_success", "label_category"]]
            df.columns = ["ID", "Query", "Timestamp", "Steps", "Labeled Success", "Failure Category"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No trajectories found.")


# ── Trajectory Detail ─────────────────────────────────────────────────────────
elif page == "🔬 Trajectory Detail":
    st.title("🔬 Trajectory Detail Viewer")
    traj_id = st.text_input("Trajectory ID (e.g. traj_7468641a)")
    if traj_id.strip():
        data = api_get(f"/trajectories/{traj_id.strip()}")
        if data:
            st.markdown(f"**Query:** {data.get('query', '')}")
            st.markdown(f"**Model:** `{data.get('model_used', '')}`")
            st.markdown(f"**Timestamp:** {data.get('timestamp', '')}")

            label = data.get("label", {})
            if label:
                col1, col2, col3 = st.columns(3)
                col1.metric("Outcome", "✅ Success" if label.get("success") else "❌ Failure")
                col2.metric("Category", label.get("failure_category", "—") or "—")
                col3.metric("Confidence", label.get("confidence", "—") or "—")
                if label.get("reasoning"):
                    with st.expander("Label Reasoning"):
                        st.write(label["reasoning"])

            st.divider()
            st.subheader("Trajectory Steps")
            for i, step in enumerate(data.get("steps", [])):
                stype = step.get("type", "")
                if stype == "human":
                    st.markdown(f"**{i+1}. 👤 Customer:** {step.get('content', '')}")
                elif stype == "tool_call":
                    st.markdown(f"**{i+1}. 🔧 Tool Call:** `{step.get('tool', '')}({json.dumps(step.get('args', {}))})`")
                elif stype == "tool_result":
                    st.markdown(f"**{i+1}. 📦 Tool Result:** {step.get('content', '')}")
                elif stype == "ai_response":
                    st.markdown(f"**{i+1}. 🤖 Agent:** {step.get('content', '')}")
                st.divider()
