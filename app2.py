import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Community tools
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

# HuggingFace embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# For OpenAI (optional)
# from langchain_openai import ChatOpenAI

# ---------------------------
# 1. App Title
# ---------------------------
st.title("📚 AI Q&A with LangChain + HuggingFace")

# ---------------------------
# 2. File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file:
    # Save uploaded file
    with open("uploaded.txt", "wb") as f:
        f.write(uploaded_file.getvalue())

    # ---------------------------
    # 3. Load Documents
    # ---------------------------
    loader = TextLoader("uploaded.txt")
    documents = loader.load()

    # ---------------------------
    # 4. Embeddings + Vector DB
    # ---------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    st.success("✅ Document processed & embeddings created!")

    # ---------------------------
    # 5. User Query
    # ---------------------------
    query = st.text_input("Ask a question about your document:")

    if query:
        # Search top results
        docs = vectorstore.similarity_search(query, k=3)

        # ---------------------------
        # 6. Create Prompt
        # ---------------------------
        template = """
        You are a helpful assistant. Answer the question using the context below.

        Context:
        {context}

        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = prompt.format(context=context, question=query)

        # ---------------------------
        # 7. Output (no LLM, direct context)
        # ---------------------------
        parser = StrOutputParser()
        answer = parser.parse(f"Context: {context}\n\nAnswer: Based on the document, {query}")

        st.subheader("🔍 Answer")
        st.write(answer)
