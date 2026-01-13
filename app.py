import streamlit as st
from rag_core import rag_answer, load_resources


st.set_page_config(
    page_title="Alzheimer's Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .answer-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chunk-expander {
        background-color: #f0f4f8;
        border-radius: 8px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models and index...")
def init_resources():
    return load_resources()


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider(
            "Number of sources (Top-K)",
            min_value=1,
            max_value=15,
            value=6,
            help="How many relevant chunks to retrieve from the database"
        )
        
        show_context = st.checkbox(
            "Show context for LLM",
            value=False,
            help="Display the full context passed to the model"
        )
        
        st.divider()
        
        st.header("📊 Session Statistics")
        if "query_count" not in st.session_state:
            st.session_state.query_count = 0
        st.metric("Queries in session", st.session_state.query_count)
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info("""
        **RAG System** for analyzing biomedical literature 
        on Alzheimer's disease.
        
        - **Embedding**: BGE-base
        - **LLM**: Qwen2.5-7B
        - **Vector DB**: FAISS
        """)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.query_count = 0
            st.rerun()
        
        return top_k, show_context


def render_sources(sources: str, top_chunks: list):
    st.subheader("📚 Cited Sources")
    
    if sources == "No sources cited.":
        st.warning("The model did not cite any sources")
        return
    
    for line in sources.strip().split("\n"):
        if line.strip():
            st.markdown(f"""
            <div class="source-card">
                {line}
            </div>
            """, unsafe_allow_html=True)


def render_context(context: str, top_chunks: list):
    st.subheader("📄 Context (chunks from database)")
    
    for i, chunk in enumerate(top_chunks, 1):
        with st.expander(f"[{i}] {chunk.get('title', 'Untitled')[:80]}... ({chunk.get('year', 'N/A')})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Authors:** {chunk.get('authors', 'N/A')}")
                st.markdown(f"**URL:** {chunk.get('url', 'N/A')}")
            
            with col2:
                st.markdown(f"**Doc ID:** `{chunk.get('doc_id', 'N/A')}`")
                st.markdown(f"**Chunk ID:** `{chunk.get('chunk_id', 'N/A')}`")
            
            st.divider()
            st.markdown(chunk.get("chunk_text", "").strip())


def main():
 
    st.markdown('<p class="main-header">🧠 Alzheimer\'s Research Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered analysis of biomedical literature for therapeutic target identification</p>', unsafe_allow_html=True)

    resources = init_resources()
    top_k, show_context = render_sidebar()

    if "history" not in st.session_state:
        st.session_state.history = []
    
    st.subheader("🔍 Enter Your Question")
    
    example_queries = [
        "What are the main therapeutic targets for Alzheimer's disease?",
        "How does tau protein aggregation contribute to neurodegeneration?",
        "What is the role of neuroinflammation in Alzheimer's progression?",
        "Which biomarkers are used for early Alzheimer's detection?"
    ]
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_area(
            "Research question",
            height=100,
            placeholder="Enter your research question about Alzheimer's disease...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Examples:**")
        for i, eq in enumerate(example_queries[:3]):
            if st.button(f"📝 {i+1}", key=f"ex_{i}", help=eq):
                st.session_state.example_query = eq
                st.rerun()
    
    if "example_query" in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit = st.button("🚀 Get Answer", type="primary", use_container_width=True)
    
    if submit and query.strip():
        st.session_state.query_count += 1
        
        with st.spinner("🔄 Searching relevant documents and generating answer..."):
            try:
                answer, sources, context, top_chunks = rag_answer(
                    query=query.strip(),
                    top_k=top_k,
                    resources=resources
                )
                
                st.session_state.history.append({
                    "query": query,
                    "answer": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
        
        st.divider()
        
        st.subheader("💡 Answer")
        st.markdown(f"""
        <div class="answer-box">
            {answer}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            render_sources(sources, top_chunks)
        
        with col2:
            if show_context:
                render_context(context, top_chunks)
    
    elif submit:
        st.warning("⚠️ Please enter a question")
    
    if st.session_state.history:
        st.divider()
        with st.expander("📜 Query History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
                st.markdown(f"**{i}. {item['query'][:100]}...**")
                st.markdown(f"_{item['answer'][:200]}..._")
                st.divider()


if __name__ == "__main__":
    main()