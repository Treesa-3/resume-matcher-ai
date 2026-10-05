import streamlit as st

# ---- Page setup ----
st.set_page_config(page_title="Resume Matcher AI", page_icon="📄", layout="centered")

st.title("📄 Resume Matcher AI")
st.caption("Paste your resume and a job description to see how well they match.")

st.divider()

# ---- Feature 1: Input (resume + JD) ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    resume_text = st.text_area(
        "Paste your resume text here",
        height=250,
        placeholder="Paste resume content..."
    )

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="Paste job description..."
    )

analyze_clicked = st.button("Analyze Match", type="primary", use_container_width=True)

st.divider()

# ---- Feature 2: Match Score (placeholder) ----
st.subheader("Match Score")
if analyze_clicked:
    st.info("Scoring logic not built yet — this is just the UI skeleton.")
    st.progress(0)  # placeholder, will show real % later
else:
    st.write("Your score will appear here after clicking **Analyze Match**.")

st.divider()

# ---- Feature 3: Missing Keywords (placeholder) ----
st.subheader("Suggested Keywords to Add")
if analyze_clicked:
    st.write("Keyword suggestions will appear here, ranked by priority.")
else:
    st.write("Keyword list will appear here after analysis.")

st.divider()
st.caption("Built by Treesa Renny · Resume Matcher AI")