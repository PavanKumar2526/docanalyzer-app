import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="RAG Book Assistant", page_icon="📚")
st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

# ── API key guard ─────────────────────────────────────────────
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    st.error("⚠️ MISTRAL_API_KEY is not set. Go to Streamlit Cloud → your app → Settings → Secrets and add it.")
    st.stop()

# ── Session state ─────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# ── File upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.success("✅ PDF uploaded!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document… this may take a minute."):
            try:
                # Load PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                if not docs:
                    st.error("❌ No text found. This PDF may be scanned/image-based.")
                    st.stop()

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)

                # Load embeddings once, reuse across reruns
                if st.session_state.embeddings is None:
                    st.session_state.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                # Build FAISS vector store in memory — no disk, no system deps
                st.session_state.vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=st.session_state.embeddings
                )

                st.success(f"✅ Vector database created from {len(chunks)} chunks!")

            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")

# ── Q&A ───────────────────────────────────────────────────────
if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader("💬 Ask Questions From the Book")

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not in the context, say: 'I could not find the answer in the document.'"""),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    query = st.text_input("Enter your question")

    if query:
        with st.spinner("Searching and generating answer..."):
            try:
                retrieved_docs = retriever.invoke(query)

                if not retrieved_docs:
                    st.warning("⚠️ No relevant content found for that question.")
                else:
                    context = "\n\n".join([d.page_content for d in retrieved_docs])
                    final_prompt = prompt.invoke({"context": context, "question": query})
                    response = llm.invoke(final_prompt)

                    st.write("### 🤖 Answer")
                    st.write(response.content)

                    with st.expander("📄 Source chunks used"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Chunk {i+1}** — Page {doc.metadata.get('page', '?')}")
                            st.caption(doc.page_content[:500])

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")

else:
    if uploaded_file:
        st.info("👆 Click 'Create Vector Database' to process your PDF first.")
    else:
        st.info("👆 Upload a PDF to get started.")
