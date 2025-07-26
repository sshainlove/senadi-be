import os
import streamlit as st
import pandas as pd
from io import BytesIO
from uuid import uuid4
from dotenv import load_dotenv

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from langchain_qdrant import QdrantVectorStore

# Setup
load_dotenv()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

QDRANT_COLLECTION = "senadi-chatbot-new"

def extract_text_from_pdf(pdf_bytes: bytes, filename: str):
    documents = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                chunks = text_splitter.split_text(text)
                for j, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": filename, "page": i + 1, "chunk": j, "type": "pdf", "id": str(uuid4()), 'file_name': filename}
                    ))
    except Exception as e:
        st.warning(f"❌ Gagal parsing PDF {filename}: {e}")
    return documents

def extract_data_from_excel(excel_content, filename):
    try:
        if isinstance(excel_content, bytes):
            excel_data = pd.read_excel(BytesIO(excel_content), sheet_name=None)
        else:
            excel_data = pd.read_excel(excel_content, sheet_name=None)

        text_content = []
        for sheet_name, df in excel_data.items():
            question_cols = [col for col in df.columns if any(q in str(col).lower() for q in ['Permasalahan'])]
            answer_cols = [col for col in df.columns if any(a in str(col).lower() for a in ['Solusi'])]

            if question_cols and answer_cols:
                print(f"  Found Q&A format in sheet {sheet_name}")
                for i, row in df.iterrows():
                    for q_col in question_cols:
                        for a_col in answer_cols:
                            if pd.notna(row[q_col]) and pd.notna(row[a_col]):
                                qa_text = f"Permasalahan: {row[q_col]}\nJawaban: {row[a_col]}"
                                text_content.append(Document(
                                    page_content=qa_text,
                                    metadata={"source": f"{filename}:{sheet_name} row: {i}", "type": "qa", "id": str(uuid4()), 'file_name': filename}
                                ))
            else:
                for i, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    if row_text:
                        text_content.append(Document(
                            page_content=row_text,
                            metadata={"source": f"{filename}:{sheet_name} row: {i}", "type": "data","id": str(uuid4()), "file_name": filename}
                        ))

        return text_content
    except Exception as e:
        print(f"Error extracting data from Excel {filename}: {str(e)}")
        return []

def extract_csv(file, filename):
    documents = []
    try:
        df = pd.read_csv(file)
        for i, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            if row_text:
                documents.append(Document(
                    page_content=row_text,
                    metadata={"source": f"{filename} row: {i}", "type": "csv", "id": str(uuid4())}
                ))
    except Exception as e:
        st.warning(f"❌ Gagal parsing CSV {filename}: {e}")
    return documents

def process_files(files):
    all_docs = []
    for file in files:
        filename = file.name
        file_bytes = file.read()
        if filename.endswith('.pdf'):
            docs = extract_text_from_pdf(file_bytes, filename)
        elif filename.endswith(('.xlsx', '.xls')):
            docs = extract_data_from_excel(file_bytes, filename)
        elif filename.endswith('.csv'):
            docs = extract_csv(BytesIO(file_bytes), filename)
        else:
            st.warning(f"❌ Format file tidak didukung: {filename}")
            continue
        st.success(f"✅ {filename} menghasilkan {len(docs)} dokumen.")
        all_docs.extend(docs)
    return all_docs

def push_to_qdrant(documents, batch_size=10):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120
    )
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)}
        )

    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="metadata.file_name",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        print(f"[Qdrant] Payload index creation error (may already exist): {e}")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
        vector_name="dense"
    )

    with st.spinner("🚀 Mengunggah ke Qdrant..."):
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                st.info(f"✅ Ditambahkan batch: {len(batch)} dokumen")
            except Exception as e:
                st.error(f"❌ Gagal batch: {e}")
        st.success("🎉 Semua dokumen berhasil ditambahkan ke Qdrant.")

def main():
    st.set_page_config(page_title="Upload ke Qdrant", layout="wide")
    st.title("📚 Upload PDF, Excel, dan CSV ke Qdrant (senadi-dataset)")

    uploaded_files = st.file_uploader("📤 Pilih file", type=["pdf", "xlsx", "xls", "csv"], accept_multiple_files=True)

    if uploaded_files:
        documents = process_files(uploaded_files)
        if documents:
            if st.button("🚀 Tambahkan ke Qdrant"):
                push_to_qdrant(documents)

if __name__ == "__main__":
    main()
