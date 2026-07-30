import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="LangChain Demo",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0F172A,#1E3A8A);
}

.main-title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:30px;
}

.response-box{
    background:white;
    padding:20px;
    border-radius:12px;
    border-left:6px solid #2563EB;
    color:black;
}

.stButton>button{
    width:100%;
    background:#2563EB;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Heading
# -----------------------------
st.markdown(
    "<div class='main-title'>🤖 LangChain + LangSmith Demo</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Prompt → Model → Output Parser</div>",
    unsafe_allow_html=True
)

# -----------------------------
# User Input
# -----------------------------
topic = st.text_input(
    "Enter Topic",
    placeholder="Example: LangSmith"
)

col1, col2 = st.columns(2)

# -----------------------------
# Generate Button
# -----------------------------
with col1:

    if st.button("🚀 Generate Response"):

        if topic:

            prompt = ChatPromptTemplate.from_template(
                "Explain {topic} in two simple sentences."
            )

            model = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0
            )

            parser = StrOutputParser()

            chain = prompt | model | parser

            with st.spinner("Generating response..."):

                response = chain.invoke(
                    {
                        "topic": topic
                    }
                )

            st.markdown("### AI Response")

            st.markdown(
                f"""
                <div class="response-box">
                {response}
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------
# Clear Button
# -----------------------------
with col2:

    if st.button("🗑️ Clear"):
        st.rerun()