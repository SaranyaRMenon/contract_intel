import sys
import tempfile
import json
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from contract_intel import extraction, ingestion, pipeline, rag, risk, summarize
from contract_intel.chat import ChatMessage, chat as chat_fn

st.set_page_config(page_title="Contract Intelligence", layout="wide", page_icon="📄")

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 10px; margin-bottom: 4px; }
    .analysis-btn { margin-bottom: 8px; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Contract Intelligence System")
st.caption("Upload contracts and analyze them with AI — extraction, risk detection, and a smart assistant.")


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


@st.cache_data(show_spinner=False)
def get_contracts() -> list[dict]:
    return pipeline.list_contracts()


def load_contract_text(contract_id: str) -> str:
    path = pipeline._contract_path(contract_id)
    if path is None:
        raise FileNotFoundError(f"Contract '{contract_id}' not found.")
    return ingestion.full_text(ingestion.load_document(path))


if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[ChatMessage] = []
if "active_contract" not in st.session_state:
    st.session_state.active_contract: str | None = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result: dict | None = None


with st.sidebar:
    st.header("📁 Contract Manager")

    uploaded_file = st.file_uploader(
        "Upload a contract",
        type=["txt", "md", "pdf", "docx"],
        help="Supported: TXT, MD, PDF, DOCX",
    )
    with st.expander("Optional: set contract ID"):
        custom_id = st.text_input("Contract ID", help="Leave blank to use filename")

    if st.button("⬆️ Upload & Ingest", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a file first.")
        else:
            temp_path = save_uploaded_file(uploaded_file)
            try:
                cid = custom_id.strip() if custom_id else Path(uploaded_file.name).stem
                result = pipeline.ingest_file(temp_path, contract_id=cid or None)
                st.success(f"✓ Ingested: **{result.contract_id}** ({result.chunk_count} chunks)")
                get_contracts.clear()
                st.session_state.active_contract = result.contract_id
                st.session_state.chat_history = []
                st.session_state.analysis_result = None
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
            finally:
                temp_path.unlink(missing_ok=True)

    st.divider()

    contracts = get_contracts()
    contract_ids = [c["contract_id"] for c in contracts]

    if contract_ids:
        st.subheader("Switch Contract")
        selected = st.selectbox(
            "Active contract",
            contract_ids,
            index=contract_ids.index(st.session_state.active_contract)
            if st.session_state.active_contract in contract_ids
            else 0,
            label_visibility="collapsed",
        )
        if selected != st.session_state.active_contract:
            st.session_state.active_contract = selected
            st.session_state.chat_history = []
            st.session_state.analysis_result = None
            st.rerun()

        if st.session_state.active_contract:
            st.success(f"Active: **{st.session_state.active_contract}**")

        if st.button("🗑️ Delete active contract", use_container_width=True):
            if st.session_state.active_contract:
                pipeline.delete_contract(st.session_state.active_contract)
                get_contracts.clear()
                st.session_state.active_contract = None
                st.session_state.chat_history = []
                st.session_state.analysis_result = None
                st.rerun()
    else:
        st.info("No contracts ingested yet.")

    st.divider()
    if st.button("🧹 Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


if not st.session_state.active_contract:
    st.info("👈 Upload a contract using the sidebar to get started.")
    st.stop()


st.subheader(f"Analyzing: `{st.session_state.active_contract}`")

st.markdown("### 📊 Fixed Analysis Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📋 Summarize", use_container_width=True):
        with st.spinner("Generating summary..."):
            try:
                text = load_contract_text(st.session_state.active_contract)
                result = summarize.summarize(text)
                st.session_state.analysis_result = {"type": "summary", "data": result}
            except Exception as exc:
                st.error(f"Summary failed: {exc}")

with col2:
    if st.button("🔍 Extract Clauses", use_container_width=True):
        with st.spinner("Extracting clauses..."):
            try:
                text = load_contract_text(st.session_state.active_contract)
                result = extraction.extract(text).model_dump()
                st.session_state.analysis_result = {"type": "extraction", "data": result}
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")

with col3:
    if st.button("⚠️ Detect Risks", use_container_width=True):
        with st.spinner("Scanning for risks..."):
            try:
                text = load_contract_text(st.session_state.active_contract)
                risks = risk.detect_risks(text)
                st.session_state.analysis_result = {"type": "risks", "data": [r.model_dump() for r in risks]}
            except Exception as exc:
                st.error(f"Risk detection failed: {exc}")


if st.session_state.analysis_result:
    r = st.session_state.analysis_result
    with st.expander("📄 Analysis Result", expanded=True):
        if r["type"] == "summary":
            st.markdown(r["data"])

        elif r["type"] == "extraction":
            for key, value in r["data"].items():
                if not value:
                    continue
                st.markdown(f"#### {key.replace('_', ' ').title()}")
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if v is not None:
                                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                            st.markdown("---")
                        else:
                            st.write(item)
                else:
                    st.write(value)

        elif r["type"] == "risks":
            risks_list = r["data"]
            if not risks_list:
                st.success("No risks detected.")
            else:
                for risk_item in risks_list:
                    icon = "🔴" if risk_item["severity"] == "high" else "🟠" if risk_item["severity"] == "medium" else "🟡"
                    with st.expander(f"{icon} {risk_item['title']} ({risk_item['severity'].upper()})", expanded=False):
                        st.markdown(f"**Category:** {risk_item['category']}")
                        st.markdown(f"**Explanation:** {risk_item['explanation']}")


st.divider()
st.markdown("### 💬 AI Contract Assistant")
st.caption("Ask anything about the contract — uses RAG retrieval, agent tools, and remembers your conversation.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg.role):
        st.markdown(msg.content)

user_input = st.chat_input("Ask anything about this contract...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chat_fn(
                    user_message=user_input,
                    history=st.session_state.chat_history,
                    contract_id=st.session_state.active_contract,
                )
                st.session_state.chat_history = response.history
                st.markdown(response.reply)
                if response.tool_calls_made:
                    st.caption(f"🔧 Tools used: {', '.join(set(response.tool_calls_made))}")
            except Exception as exc:
                st.error(f"Chat error: {exc}")
