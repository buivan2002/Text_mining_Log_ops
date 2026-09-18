"""Dashboard Streamlit: nhập log -> xem cây sự cố."""

import streamlit as st

from src.pipeline.orchestrator import PipelineOrchestrator

st.title("Incident Mining Pipeline")

orchestrator = PipelineOrchestrator()

log_text = st.text_area("Nhập nội dung log / postmortem")

if st.button("Phân tích"):
    result = orchestrator.run(log_text)
    st.json(result)
