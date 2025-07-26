# backend/app.py
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required, decode_token
from flask_cors import CORS
import mysql.connector
import os
import traceback
import google.generativeai as genai
import json
import bcrypt
import re
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pandas as pd
import random
# import pypdf
import base64
import requests
from io import BytesIO
import fitz
from typing import List
from uuid import uuid4

# LangChain imports
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

# Load environment variables
load_dotenv()

import sys
print("Starting app...")
print(f"Python version: {sys.version}")

# Set up Google API key for Gemini
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_AVAILABLE = False
MODEL_NAME = 'gemini-1.5-flash'  # Default model name

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Try with different model names
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            test_response = model.generate_content("Test")
            if test_response:
                print(f"Gemini model {MODEL_NAME} initialized successfully")
                GEMINI_AVAILABLE = True
        except Exception as model_error:
            print(f"Error with {MODEL_NAME}: {str(model_error)}")
            try:
                # Try alternative model name
                MODEL_NAME = 'gemini-1.5-flash'
                model = genai.GenerativeModel(MODEL_NAME)
                test_response = model.generate_content("Test")
                if test_response:
                    print(f"Gemini model {MODEL_NAME} initialized successfully")
                    GEMINI_AVAILABLE = True
            except Exception as alt_error:
                print(f"Error with {MODEL_NAME}: {str(alt_error)}")
                print("Using simple response mode for all queries.")
    except Exception as e:
        print(f"WARNING: Failed to initialize Gemini: {str(e)}")
        print("Using simple response mode for all queries.")
else:
    print("Warning: GOOGLE_API_KEY not set. Gemini integration will not work.")

# Configure uploads directory
UPLOADS_DIR = os.getenv('UPLOADS_FOLDER_PATH')
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Configure Vector store directory
VECTOR_STORE_FOLDER_PATH = os.getenv('VECTOR_STORE_FOLDER_PATH')
os.makedirs(VECTOR_STORE_FOLDER_PATH, exist_ok=True)

# Initialize Flask app
print("Flask app loaded")
app = Flask(__name__)
print("Flask app initialized")
CORS(app, 
    resources={r"/*": {
        "origins": "https://senadi.vercel.app",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }},
    supports_credentials=True
)

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['JWT_TOKEN_LOCATION'] = ['headers', 'cookies', 'query_string']
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'
jwt = JWTManager(app)

# Global vector store variable
vector_store = None
qa_chain = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# Configure database
def get_db_connection():
    # db_name = os.getenv('DB_NAME', 'chatbot.susenas')
    db_name = os.getenv('DB_NAME', os.getenv("MYSQL_DATABASE"))
    print(f"Connecting to database: {db_name}")
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', os.getenv("MYSQLHOST")),
        user=os.getenv('DB_USER', os.getenv("MYSQLUSER")), 
        password=os.getenv('DB_PASSWORD', os.getenv("MYSQLPASSWORD")),
        database=db_name,
        port=int(os.getenv('DB_PORT', os.getenv('MYSQLPORT')))
    )

def process_documents_and_add_to_qdrant(vector_store, doc_generator, batch_size):
    batch = []
    for doc in doc_generator:
        batch.append(doc)
        if len(batch) >= batch_size:
            try:
                vector_store.add_documents(batch)
                print(f"Added batch of {len(batch)} documents")
            except Exception as e:
                print(f"[Batch error] {e}")
            batch = []
    if batch:
        try:
            vector_store.add_documents(batch)
            print(f"Added final batch of {len(batch)} documents")
        except Exception as e:
            print(f"[Final batch error] {e}")


            
def extract_text_from_pdf(pdf_bytes: bytes, filename: str) -> List[Document]:
    """
    Extract text from PDF bytes and return list of Document objects with splitting.
    Each page is extracted and split into chunks using RecursiveCharacterTextSplitter.
    """
    documents = []
    # batch_size = 50
    start_page = 0
    max_pages = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        end_page = total_pages if max_pages is None else min(start_page + max_pages, total_pages)

        print(f"Extracting PDF '{filename}' from page {start_page} to {end_page}...")

        for i in range(start_page, end_page):
            try:
                page = doc[i]
                extracted = page.get_text()
                if extracted and extracted.strip():
                    chunks = text_splitter.split_text(extracted.strip())
                    for j, chunk in enumerate(chunks):
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": filename,
                                    "page": i + 1,
                                    "chunk": j,
                                    "type": "pdf",
                                    "id": str(uuid4()),
                                    'file_name': filename
                                }
                            )
                        )
                else:
                    print(f"[Warning] Page {i + 1} is empty or not extractable.")
            except Exception as e:
                print(f"[Error] Failed to extract page {i + 1}: {e}")
                continue

        return documents
    except Exception as e:
        print(f"[Fatal Error] Failed to open or parse PDF: {e}")
        return []

def extract_data_from_excel(excel_content, filename="unknown.xlsx"):
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
                            metadata={"source": f"{filename}:{sheet_name} row: {i}", "type": "data", "file_name": filename}
                        ))

        return text_content
    except Exception as e:
        print(f"Error extracting data from Excel {filename}: {str(e)}")
        return []


def process_documents_from_uploads(deleted_filename = None):
    """Process all documents in the uploads directory and convert to Document objects"""
    uploads_dir = os.getenv('UPLOADS_FOLDER_PATH')
    documents = []
    basic_info = """
    Badan Pusat Statistik (BPS) merupakan lembaga pemerintah non kementerian yang bertanggung jawab langsung kepada presiden. Berdasarkan Undang-Undang Republik Indonesia No 16 Tahun 1997, salah satu peranan BPS adalah menyediakan kebutuhan data bagi pemerintah dan masyarakat. BPS mengumpulkan data dari berbagai sumber dengan cara sensus, survei, kompilasi produk administrasi, dan cara lain sesuai perkembangan ilmu pengetahuan dan teknologi. Survei Sosial Ekonomi Nasional (Susenas) merupakan sandaran utama Indonesia dalam hal kebutuhan data untuk mengimplementasikan pembangunan yang sejalan dengan Tujuan Pembangunan Berkelanjutan/Sustainable Development Goals (TPB/SDGs). Pertanyaan-pertanyaan Susenas, baik Susenas Kor maupun Susenas Modul merupakan tulang punggung indikator SDGs, RPJMN, dan kesejahteraan bangsa.
    """
    
    # Add basic info as a document
    documents.append(Document(
        page_content=basic_info,
        metadata={"source": "basic_info", "type": "overview"}
    ))
    
    # Process each file in the uploads directory
    try:
        print(f"Scanning directory: {uploads_dir}")
        files = os.listdir(uploads_dir)
        
        files = [f for f in files if f != deleted_filename]

        # Process Excel files first for Q&A data
        excel_files = [f for f in files if f.lower().endswith(('.xlsx', '.xls'))]
        excel_files.sort(reverse=True)
        
        for filename in excel_files:
            filepath = os.path.join(uploads_dir, filename)
            if not os.path.isfile(filepath):
                continue
                
            print(f"Processing Excel: {filename}")
            try:
                excel_docs = extract_data_from_excel(filepath)
                documents.extend(excel_docs)
                print(f"Added {len(excel_docs)} Q&A pairs from {filename}")
            except Exception as e:
                print(f"Error processing Excel {filename}: {str(e)}")
        
        # Then process PDF files
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        pdf_files.sort(key=lambda f: os.path.getsize(os.path.join(uploads_dir, f)))
        
        for filename in pdf_files:
            filepath = os.path.join(uploads_dir, filename)
            if not os.path.isfile(filepath):
                continue
                
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
            max_pages = None  # Hilangkan batasan jumlah halaman, proses semua halaman
            
            print(f"Processing PDF: {filename} (max {max_pages} pages)")
            
            try:
                docs = extract_text_from_pdf(filepath, filename)
                documents.append(docs)
            except Exception as e:
                print(f"Error processing PDF {filename}: {str(e)}")
        
        # Process text files last
        text_files = [f for f in files if f.lower().endswith(('.txt', '.md', '.csv'))]
        for filename in text_files:
            filepath = os.path.join(uploads_dir, filename)
            if not os.path.isfile(filepath):
                continue
                
            print(f"Processing text file: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    chunks = text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={"source": filename, "chunk": i, "type": "text"}
                        ))
            except Exception as read_error:
                print(f"Failed to read text file {filename}: {str(read_error)}")
        
        print(f"Total documents processed: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        print(traceback.format_exc())
        return [Document(page_content=basic_info, metadata={"source": "basic_info", "type": "overview"})]

def process_documents_from_uploads_github(deleted_filename = None):
    """Process all documents in the uploads directory and convert to Document objects"""
    documents = []
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    github_path = os.getenv("GITHUB_FOLDER_PATH")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }
    
    basic_info = """
    Badan Pusat Statistik (BPS) merupakan lembaga pemerintah non kementerian yang bertanggung jawab langsung kepada presiden. Berdasarkan Undang-Undang Republik Indonesia No 16 Tahun 1997, salah satu peranan BPS adalah menyediakan kebutuhan data bagi pemerintah dan masyarakat. BPS mengumpulkan data dari berbagai sumber dengan cara sensus, survei, kompilasi produk administrasi, dan cara lain sesuai perkembangan ilmu pengetahuan dan teknologi. Survei Sosial Ekonomi Nasional (Susenas) merupakan sandaran utama Indonesia dalam hal kebutuhan data untuk mengimplementasikan pembangunan yang sejalan dengan Tujuan Pembangunan Berkelanjutan/Sustainable Development Goals (TPB/SDGs). Pertanyaan-pertanyaan Susenas, baik Susenas Kor maupun Susenas Modul merupakan tulang punggung indikator SDGs, RPJMN, dan kesejahteraan bangsa.
    """
    
    # Add basic info as a document
    # documents.append(Document(
    #     page_content=basic_info,
    #     metadata={"source": "basic_info", "type": "overview"}
    # ))
    
    yield Document(
        page_content=basic_info,
        metadata={"source": "basic_info", "type": "overview"}
    )
    
    # Process each file in the uploads directory
    try:
        url = f"https://api.github.com/repos/{repo}/contents/{github_path}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch list: {response.text}")
            return documents
        
        for file_info in response.json():
            name = file_info["name"]
            
            # Skip deleted file
            if deleted_filename and name == deleted_filename:
                print(f"Skipping deleted file: {name}")
                continue
            
            local_uploads_dir = os.getenv("UPLOADS_FOLDER_PATH")
            local_file_path = os.path.join(local_uploads_dir, name)
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"Deleted local file: {local_file_path}")
            else:
                print(f"Local file not found: {local_file_path}")
            
            file_url = file_info.get("download_url")
            file_content = requests.get(file_url)
            if file_content.status_code != 200:
                print(f"Error fetching {name}")
                continue

            print(f"Processing {name}")
            if name.lower().endswith(('.xlsx', '.xls')):
                docs = extract_data_from_excel(file_content.content, filename=name)
                # documents.extend(docs)
            elif name.lower().endswith('.pdf'):
                docs = extract_text_from_pdf(file_content.content, name)
                # documents.extend(docs)
            elif name.lower().endswith(('.txt', '.csv', '.md')):
                text = file_content.text
                chunks = text_splitter.split_text(text)
                # for i, chunk in enumerate(chunks):
                #     documents.append(Document(page_content=chunk, metadata={"source": name, "chunk": i}))
                docs = [Document(page_content=chunk, metadata={"source": name, "chunk": i}) for i, chunk in enumerate(chunks)]
            else:
                docs = []

            for doc in docs:
                yield doc
                
        # print(f"Processed {len(docs)} documents.")
        print(f"Processed documents from github is finished.")
        # return documents

    except Exception as e:
        print(traceback.format_exc())
        return documents

def initialize_vector_store():
    """Initialize or load the vector store with documents from uploads"""
    global vector_store, qa_chain
    
    try:
        # Try to load existing vector store
        if os.path.exists(os.path.join(VECTOR_STORE_FOLDER_PATH, "index.faiss")):
            print("Loading existing FAISS index...")
            try:
                # Use GoogleGenerativeAIEmbeddings as default - more reliable
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
                vector_store = FAISS.load_local(VECTOR_STORE_FOLDER_PATH, embeddings, allow_dangerous_deserialization=True)
                print("Successfully loaded FAISS index with GoogleGenerativeAIEmbeddings")
            except Exception as load_error:
                print(f"Error loading FAISS index: {str(load_error)}")
                vector_store = None

        # If vector store couldn't be loaded or doesn't exist, create a new one
        if vector_store is None:
            print("Creating new vector store...")
            # documents = process_documents_from_uploads()
            documents = process_documents_from_uploads_github()
            
            if not documents:
                print("No documents found to process")
                return False
                
            print(f"Creating embeddings for {len(documents)} documents")
            
            # Use GoogleGenerativeAIEmbeddings as default
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
                vector_store = FAISS.from_documents(documents, embeddings)
                print("Created FAISS index with GoogleGenerativeAIEmbeddings")
                
                # Save the index
                vector_store.save_local(VECTOR_STORE_FOLDER_PATH)
                print(f"Saved FAISS index to {VECTOR_STORE_FOLDER_PATH}")
            except Exception as embed_error:
                print(f"Error creating embeddings: {str(embed_error)}")
                traceback.print_exc()
                return False
        
        # Create the QA chain
        if GEMINI_AVAILABLE and vector_store:
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            
            # Template for Gemini
            template = """
            Anda adalah asisten virtual khusus untuk menangani permasalahan terkait konsep, definisi, dan kasus batas Survei Sosial Ekonomi Nasional (Susenas) yang dilaksanakan oleh Badan Pusat Statistik (BPS). Bantu pengguna dengan informasi yang akurat dan detail tentang Susenas berdasarkan konteks yang diberikan.

            Jangan hanya mencari jawaban yang persis sama dengan pertanyaan pengguna. Pelajari dan parafrase dokumen PDF dan Excel. Pahami bahwa kalimat dapat memiliki arti yang sama meskipun diparafrase. Gunakan pemahaman semantik untuk menemukan jawaban berdasarkan makna, bukan hanya kemiripan kata secara literal.

            Jika ditemukan beberapa jawaban dari dataset atau dokumen yang berbeda, utamakan jawaban yang berasal dari **dokumen atau file terbaru** (yang memiliki waktu unggah paling baru). Tunjukkan pemahaman yang tepat terhadap konteks saat ini.

            Berikan jawaban yang relevan, ringkas, dan hanya berdasarkan dokumen yang tersedia. Jangan menjawab berdasarkan asumsi atau di luar konteks.

            Jika informasi tidak tersedia dalam konteks, katakan secara formal:
            **"Terima kasih atas pertanyaan Anda. Saat ini informasi yang Anda cari sedang dalam proses peninjauan dan akan segera dijawab oleh instruktur. Kami menghargai kesabaran Anda dan akan memastikan bahwa pertanyaan Anda akan segera mendapatkan jawaban yang akurat."**

            JANGAN pernah mengarang jawaban. Jangan gunakan tanda bintang (*) atau tanda lain yang tidak formal.

            Gunakan Bahasa Indonesia yang baik dan benar. Pastikan jawaban bersifat informatif, jelas, dan tepat sasaran.

            Konteks:
            {context}
            
            Pertanyaan: {question}
            
            Jawaban yang informatif, lengkap, dan presisi:
            """
            
            PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"],
            )
            
            llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.2)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print("Successfully created QA chain with Gemini")
            return True
        else:
            print("Could not create QA chain - Gemini not available or vector store failed")
            return False
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        print(traceback.format_exc())
        return False

@app.before_request
def initialize_vector_store_qdrant():
    # global vector_store, qa_chain
    if app.config.get("qa_chain", False):
        return  # ✅ Sudah siap, skip semua
    print("Initializing vector store with Qdrant")
    try:
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,
            timeout=60
        )
        collection_name = os.getenv("QDRANT_COLLECTION")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )

        # Cek apakah collection sudah ada
        if not qdrant_client.collection_exists(collection_name):
            print(f"[Qdrant] Collection '{collection_name}' not found. Creating...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense" : VectorParams(size=768, distance=Distance.COSINE)},
            )

            try:
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.file_name",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                print(f"[Qdrant] Payload index error: {e}")
            
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
                vector_name="dense",
            )
            
            print("Memproses dokumen dari Github")
            documents = process_documents_from_uploads_github()
            # vector_store.add_documents(documents)
            process_documents_and_add_to_qdrant(vector_store, documents, batch_size=3)
            print("Berhasil menyimpan dokumen Github ke Qdrant")

        else:
            print(f"[Qdrant] Collection {collection_name} already exist. Loading collection")
            
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
                vector_name="dense",
            )

        # QA Chain
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        PROMPT = PromptTemplate(
            template="""
            Anda adalah asisten virtual khusus untuk menangani permasalahan terkait konsep, definisi, dan kasus batas Survei Sosial Ekonomi Nasional (Susenas) yang dilaksanakan oleh Badan Pusat Statistik (BPS). Bantu pengguna dengan informasi yang akurat dan detail tentang Susenas berdasarkan konteks yang diberikan.

            Jangan hanya mencari jawaban yang persis sama dengan pertanyaan pengguna. Pelajari dan parafrase dokumen PDF dan Excel. Pahami bahwa kalimat dapat memiliki arti yang sama meskipun diparafrase. Gunakan pemahaman semantik untuk menemukan jawaban berdasarkan makna, bukan hanya kemiripan kata secara literal.

            Jika ditemukan beberapa jawaban dari dataset atau dokumen yang berbeda, utamakan jawaban yang berasal dari **dokumen atau file terbaru** (yang memiliki waktu unggah paling baru). Tunjukkan pemahaman yang tepat terhadap konteks saat ini.

            Berikan jawaban yang relevan, ringkas, dan hanya berdasarkan dokumen yang tersedia. Jangan menjawab berdasarkan asumsi atau di luar konteks.

            Jika informasi tidak tersedia dalam konteks, katakan secara formal:
            **"Terima kasih atas pertanyaan Anda. Saat ini informasi yang Anda cari sedang dalam proses peninjauan dan akan segera dijawab oleh instruktur. Kami menghargai kesabaran Anda dan akan memastikan bahwa pertanyaan Anda akan segera mendapatkan jawaban yang akurat."**

            JANGAN pernah mengarang jawaban. Jangan gunakan tanda bintang (*) atau tanda lain yang tidak formal.

            Gunakan Bahasa Indonesia yang baik dan benar. Pastikan jawaban bersifat informatif, jelas, dan tepat sasaran.
            
            Pertanyaan: {question}
            
            Konteks: {context}
            """,
            input_variables=["context", "question"]
        )

        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.2)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        app.config['qa_chain'] = qa_chain
        app.config['vector_store'] = vector_store

        print("QA chain created using Qdrant vector store.")
        return 

    except Exception as e:
        print(f"Failed to initialize Qdrant vector store: {str(e)}")
        traceback.print_exc()
        return 

# Initialize the vector store on startup
# print("Initializing vector store...")
# vector_store_initialized = initialize_vector_store_qdrant()

# Enhanced conversation handlers
def get_greeting_response(message):
    """Return a natural greeting response in Indonesian"""
    greetings = {
        'hai': [
            'Hai! Ada yang bisa saya bantu tentang Susenas?',
            'Halo! Senang berbicara dengan Anda. Ada yang ingin ditanyakan tentang Susenas?'
        ],
        'halo': [
            'Halo! Apa kabar? Ada yang bisa saya bantu?',
            'Halo! Senang bertemu dengan Anda. Tanyakan saja apa yang ingin Anda ketahui tentang Susenas.'
        ],
        'selamat pagi': [
            'Selamat pagi! Semoga hari Anda menyenangkan. Ada yang bisa saya bantu terkait Susenas?',
            'Pagi yang cerah! Saya siap membantu Anda dengan informasi Susenas hari ini.'
        ],
        'selamat siang': [
            'Selamat siang! Ada yang bisa saya bantu terkait Susenas?',
            'Siang yang cerah! Saya siap membantu Anda dengan informasi Susenas hari ini.'
        ],
        'selamat sore': [
            'Selamat sore! Ada yang bisa saya bantu terkait Susenas?',
            'Sore yang menyenangkan! Saya siap membantu Anda dengan informasi Susenas hari ini.'
        ],
        'selamat malam': [
            'Selamat malam! Ada yang bisa saya bantu terkait Susenas?',
            'Malam yang tenang! Saya siap membantu Anda dengan informasi Susenas.'
        ],
        'apa kabar': [
            'Saya baik-baik saja, terima kasih telah bertanya! Bagaimana dengan Anda? Ada yang ingin ditanyakan tentang Susenas?',
            'Sebagai chatbot, saya selalu siap membantu! Bagaimana kabar Anda? Ada yang ingin ditanyakan tentang Susenas?'
        ],
        'assalamualaikum': [
            'Waalaikumsalam Warahmatullahi Wabarakatuh. Ada yang bisa saya bantu terkait Susenas?',
            'Waalaikumsalam. Semoga hari Anda menyenangkan. Ada pertanyaan tentang Susenas?'
        ],
        'terimakasih': [
            'Sama-sama! Senang bisa membantu. Ada hal lain yang ingin ditanyakan?',
            'Dengan senang hati! Jangan ragu untuk bertanya lagi jika ada yang ingin Anda ketahui tentang Susenas.'
        ],
        'terima kasih': [
            'Sama-sama! Senang bisa membantu. Ada hal lain yang ingin ditanyakan?',
            'Dengan senang hati! Jangan ragu untuk bertanya lagi jika ada yang ingin Anda ketahui tentang Susenas.'
        ],
        'salam kenal': [
            'Salam kenal juga! Saya Chatbot Susenas, asisten yang akan membantu Anda dengan informasi tentang Survei Sosial Ekonomi Nasional.',
            'Salam kenal! Saya siap membantu menjawab pertanyaan Anda tentang Susenas. Apa yang ingin Anda ketahui?'
        ],
        'siapa kamu': [
            'Saya adalah Chatbot Susenas, asisten virtual yang dirancang untuk membantu Anda dengan informasi seputar Survei Sosial Ekonomi Nasional (Susenas).',
            'Saya Chatbot Susenas, asisten yang akan membantu Anda memahami dan mengakses informasi tentang Survei Sosial Ekonomi Nasional yang dilakukan oleh BPS.'
        ],
        'kamu siapa': [
            'Saya adalah Chatbot Susenas, asisten virtual yang dirancang untuk membantu Anda dengan informasi seputar Survei Sosial Ekonomi Nasional (Susenas).',
            'Saya Chatbot Susenas, asisten yang akan membantu Anda memahami dan mengakses informasi tentang Survei Sosial Ekonomi Nasional yang dilakukan oleh BPS.'
        ]
    }
    
    # Check if message contains any greeting key
    message_lower = message.lower()
    for key, responses in greetings.items():
        if key in message_lower:
            return random.choice(responses)
    
    return None

def is_about_bot(message):
    """Check if the message is asking about the bot itself"""
    bot_indicators = ['kamu', 'bot', 'asisten', 'dirimu', 'namamu', 'siapa kamu', 'bot susenas', 'susenas bot']
    question_indicators = ['siapa', 'apa', 'kenapa', 'mengapa', 'bagaimana', 'kapan', 'untuk apa', 'fungsi', 'kegunaan']
    
    message_lower = message.lower()
    has_bot_word = any(word in message_lower for word in bot_indicators)
    has_question = any(word in message_lower for word in question_indicators)
    
    return has_bot_word and has_question

def get_bot_info_response():
    """Return information about the bot itself"""
    responses = [
        "Saya adalah Chatbot Susenas, asisten virtual yang dirancang untuk membantu Anda dengan informasi seputar Survei Sosial Ekonomi Nasional (Susenas). Saya dapat menjawab pertanyaan tentang metodologi survei, konsep dan definisi, dan kasus batas yang sering terjadi saat Susenas."
    ]
    return random.choice(responses)

def is_about_susenas(message):
    """Check if the message is asking about Susenas in general"""
    susenas_indicators = ['susenas', 'survei', 'survey', 'sosial ekonomi']
    general_question = ['apa', 'itu', 'jelaskan', 'ceritakan', 'definisi', 'maksud', 'pengertian', 'adalah']
    
    message_lower = message.lower()
    has_susenas_word = any(word in message_lower for word in susenas_indicators)
    has_general_q = any(word in message_lower for word in general_question)
    
    return has_susenas_word and has_general_q

def get_susenas_info_response():
    """Return general information about Susenas"""
    responses = [
        
        "Survei Sosial Ekonomi Nasional (Susenas) merupakan salah satu sandaran utama Indonesia dalam hal kebutuhan data untuk mengimplementasikan pembangunan yang sejalan dengan Tujuan Pembangunan Berkelanjutan/Sustainable Development Goals (TPB/SDGs). Pertanyaan-pertanyaan Susenas, baik Susenas Kor maupun Susenas Modul merupakan tulang punggung indikator SDGs, RPJMN, dan kesejahteraan bangsa."
    ]
    return random.choice(responses)

def is_about_bps(message):
    """Check if the message is asking about Susenas in general"""
    susenas_indicators = ['bps', 'Badan Pusat Statistik']
    general_question = ['apa', 'itu', 'jelaskan', 'ceritakan', 'definisi', 'maksud', 'pengertian', 'adalah']
    
    message_lower = message.lower()
    has_susenas_word = any(word in message_lower for word in susenas_indicators)
    has_general_q = any(word in message_lower for word in general_question)
    
    return has_susenas_word and has_general_q

def get_bps_info_response():
    """Return general information about BPS"""
    responses = [

        "Badan Pusat Statistik (BPS) merupakan lembaga pemerintah non kementerian yang bertanggung jawab langsung kepada presiden. Berdasarkan Undang-Undang Republik Indonesia No 16 Tahun 1997, salah satu peranan BPS adalah menyediakan kebutuhan data bagi pemerintah dan masyarakat. BPS mengumpulkan data dari berbagai sumber dengan cara sensus, survei, kompilasi produk administrasi, dan cara lain sesuai perkembangan ilmu pengetahuan dan teknologi."
    ]
    return random.choice(responses)

@app.route('/')
def index():
    try:
        # Test database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()

        # Check authentication
        jwt_config = {
            'secret_key': app.config['JWT_SECRET_KEY'][:5] + '...',
            'token_location': app.config['JWT_TOKEN_LOCATION'],
            'header_name': app.config['JWT_HEADER_NAME'],
            'header_type': app.config['JWT_HEADER_TYPE'],
            'expiration': str(app.config['JWT_ACCESS_TOKEN_EXPIRES'])
        }

        # Get list of routes
        routes = [str(rule) for rule in app.url_map.iter_rules()]

        return jsonify({
            "status": "success",
            "message": "Chatbot API is running",
            "database": "connected",
            "gemini": "available" if GEMINI_AVAILABLE else "unavailable",
            "jwt_config": jwt_config,
            "endpoints": routes,
            "cors": "configured"
        })
    except Exception as e:
        print(f"Health check error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e),
            "details": traceback.format_exc()
        }), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        hash_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') if password else None
        
        # Validate required fields
        if not username or not email or not hash_password:
            return jsonify({"error": "Username, email and password are required"}), 400
            
        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({"error": "Invalid email format"}), 400
            
        # Validate password strength
        if len(password) < 4:
            return jsonify({"error": "Password must be at least 4 characters"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({"error": "Email already registered"}), 409
        
        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({"error": "Username already taken"}), 409
            
        # Store password as plain text to match existing database format
        # Note: In production, passwords should be hashed with bcrypt
        
        # Generate user ID
        # user_id = str(uuid.uuid4())
        
        # Insert user into database (default role = 'user')
        
        print("Registering: ", username, email, hash_password)
        cursor.execute(
            "INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
            (username, email, hash_password, 'user',)
        )
        conn.commit()
        user_id = cursor.lastrowid
        # Generate JWT token
        token = create_access_token(identity=str(user_id))
        
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "token": token,
            "user": {
                "id": user_id,
                "username": username,
                "email": email,
                "is_admin": False
            }
        }), 201
        
    except Exception as e:
        print(f"Registration error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except:
            pass

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get user by email
        cursor.execute("SELECT id, username, email, password, role FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401
        
        # Compare hashed password using bcrypt
        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({"error": "Invalid email or password"}), 401
        
        # Generate JWT token
        # token = create_access_token(identity=user['id'])
        token = create_access_token(identity=str(user['id']))
        
        # Check if role is admin
        is_admin = user['role'] == 'admin'
        
        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "is_admin": is_admin
            }
        }), 200

    except Exception as e:
        print(f"Login error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except:
            pass

@app.route('/chat/new', methods=['POST'])
@jwt_required()
def create_chat():
    try:
        user_id = get_jwt_identity()
        print(f"Creating chat for user: {user_id}")
        print("Data sent: ", request.get_json())
        # Check request content type
        # content_type = request.headers.get('Content-Type', '')
        # print(f"Request Content-Type: {content_type}")
        
        # Handle different content types or empty requests
        # if 'application/json' in content_type and request.data:
        try:
            data.pop("subject", None) 
            if request.is_json:
                data = request.get_json(silent=True) or {}
            else:
                data = {}
            # title = data.get('title', 'New Chat')
            title = data.get('title')
        except Exception as json_error:
            print(f"JSON parsing error: {str(json_error)}")
            title = 'New Chat'
        # elif 'application/x-www-form-urlencoded' in content_type:
        #     title = request.form.get('title', 'New Chat')
        # else:
            # Default title for empty requests
            # title = 'New Chat'
            
        print(f"Chat title: {title}")
        
        # First verify the user exists
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            print(f"User not found: {user_id}")
            return jsonify({"error": "User not found"}), 404
        
        print(f"User found: {user['username']}")
        
        # Generate chat ID
        # chat_id = str(uuid.uuid4())
        # print(f"Generated chat ID: {chat_id}")
        
        # Fetch table structure to determine available columns
        cursor.execute("DESCRIBE chats")
        columns = cursor.fetchall()
        column_names = [col['Field'] for col in columns]
        print(f"Available columns in chats table: {column_names}")
        
        # Prepare insert query based on available columns
        # insert_columns = ['id', 'user_id']
        insert_columns = ['user_id']
        # insert_values = [chat_id, user_id]
        insert_values = [user_id]
        
        if 'title' in column_names:
            insert_columns.append('title')
            insert_values.append(title)
        
        if 'created_at' in column_names and 'DEFAULT' not in [col.get('Extra', '').upper() for col in columns if col['Field'] == 'created_at']:
            insert_columns.append('created_at')
            insert_values.append(datetime.now())
        
        # Construct the query dynamically
        placeholders = ', '.join(['%s'] * len(insert_values))
        insert_query = f"INSERT INTO chats ({', '.join(insert_columns)}) VALUES ({placeholders})"
        print(f"Insert query: {insert_query}")
        print(f"Insert values: {insert_values}")
        
        # Execute the insert
        cursor.execute(insert_query, insert_values)
        conn.commit()
        
        chat_id = cursor.lastrowid
        
        print(f"Chat created successfully: {chat_id}")
        
        return jsonify({
            "success": True,
            "chat_id": chat_id,
            "title": title
        }), 201
        
    except Exception as e:
        print(f"Create chat error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to create chat: {str(e)}"}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing database connection: {str(close_error)}")

@app.route('/chat/<chat_id>', methods=['POST'])
@jwt_required()
def chat(chat_id):
    # global qa_chain
    qa_chain = app.config.get('qa_chain')
    try:
        user_id = get_jwt_identity()
        print(f"Processing chat message for user {user_id} in chat {chat_id}")
        
        # Handle different content types
        content_type = request.headers.get('Content-Type', '')
        print(f"Request Content-Type: {content_type}")
        
        # Extract message from different content types
        content = None
        if 'application/json' in content_type and request.data:
            try:
                data = request.get_json(silent=True) or {}
                content = data.get('message')
            except Exception as json_error:
                print(f"JSON parsing error: {str(json_error)}")
        elif 'application/x-www-form-urlencoded' in content_type:
            content = request.form.get('message')
        else:
            # Try to get from any source as fallback
            content = (request.get_json(silent=True) or {}).get('message') or request.form.get('message')
        
        if not content:
            return jsonify({"error": "Message is required"}), 400
        
        print(f"Received message: {content[:50]}...")
        
        # Verify chat exists and belongs to user
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM chats WHERE id = %s AND user_id = %s", (chat_id, user_id))
    
        chat = cursor.fetchone()
        if not chat:
            cursor.close()
            conn.close()
            return jsonify({"error": "Chat not found or access denied"}), 404
        
        # Save user message - Using AUTO_INCREMENT for id
        try:
            cursor.execute(
                "INSERT INTO messages (chat_id, message, sender) VALUES (%s, %s, %s)",
                (chat_id, content, 'user')
            )
            conn.commit()
        except Exception as e:
            print(f"Error saving user message: {str(e)}")
            # Continue even if saving fails - we'll still try to generate a response
        
        # Get last 5 messages for context
        cursor.execute(
            "SELECT message, sender FROM messages WHERE chat_id = %s ORDER BY created_at DESC LIMIT 5",
            (chat_id,)
        )
        
        previous_messages = cursor.fetchall()
        previous_messages.reverse()  # Reverse to get chronological order
        
        # Build conversation context
        conversation_history = ""
        for msg in previous_messages:
            role = "Human" if msg['sender'] == 'user' else "Assistant"
            conversation_history += f"{role}: {msg['message']}\n"
        
        bot_response = ""
        is_from_correction = False
        
        # Cek apakah ada jawaban terkoreksi yang sesuai dengan pertanyaan ini
        try:
            # Cari pertanyaan serupa yang pernah dikoreksi (simple exact match)
            cursor.execute("""
                SELECT m_corr.message AS corrected_answer
                FROM messages m_user
                JOIN messages m_bot ON m_bot.chat_id = m_user.chat_id 
                JOIN messages m_corr ON m_corr.corrected_message_id = m_bot.id
                WHERE m_user.message = %s 
                    AND m_user.sender = 'user'
                    AND m_bot.sender = 'bot'
                    AND m_bot.is_corrected = FALSE
                    AND m_corr.is_correction = FALSE
                ORDER BY m_corr.created_at DESC
                LIMIT 1
            """, (content,))
            
            corrected_result = cursor.fetchone()
            
            if corrected_result:
                bot_response = corrected_result['corrected_answer']
                is_from_correction = True
                print(f"Using corrected answer from previous admin verification for '{content}'")
            
        except Exception as corr_err:
            print(f"Error checking for corrected answers: {str(corr_err)}")
            # Lanjutkan proses normal jika terjadi error
        
        # Jika tidak ada jawaban terkoreksi, gunakan flow normal
        if not bot_response:
            # Check for greeting or simple queries first
            greeting_response = get_greeting_response(content)
            if greeting_response:
                bot_response = greeting_response
            elif is_about_bot(content):
                bot_response = get_bot_info_response()
            elif is_about_susenas(content):
                bot_response = get_susenas_info_response()
            elif is_about_bps(content):
                bot_response = get_bps_info_response()
            else:
                # Use LangChain QA chain for more complex queries
                try:
                    # if qa_chain and vector_store_initialized:
                    if qa_chain :
                        print("Using LangChain QA chain for response generation")
                        try:
                            # Include conversation history in the query for context
                            result = qa_chain({"query": content})
                            
                            # Extract the answer and sources
                            if result and "result" in result:
                                bot_response = result["result"]
                                
                                # Log source documents used
                                if "source_documents" in result:
                                    sources = [doc.metadata.get('source', 'unknown') for doc in result["source_documents"]]
                                    print(f"Sources used: {sources}")
                            else:
                                print("No result from QA chain")
                                bot_response = "Maaf, saya mengalami kesulitan memproses pertanyaan Anda. Mohon coba dengan kalimat yang berbeda."
                                
                        except Exception as qa_error:
                            print(f"Error in QA chain: {str(qa_error)}")
                            traceback.print_exc()
                            
                            # Fall back to Gemini direct call if QA chain fails
                            if GEMINI_AVAILABLE:
                                print("Falling back to direct Gemini call")
                                model = genai.GenerativeModel(MODEL_NAME)
                                prompt = f"""
                                Anda adalah asisten virtual khusus untuk menangani permasalahan terkait konsep, definisi, dan kasus batas Survei Sosial Ekonomi Nasional (Susenas) yang dilaksanakan oleh Badan Pusat Statistik (BPS). Bantu pengguna dengan informasi yang akurat dan detail tentang Susenas berdasarkan konteks yang diberikan.

                                Jangan hanya mencari jawaban yang persis sama dengan pertanyaan pengguna. Pelajari dan parafrase dokumen PDF dan Excel. Pahami bahwa kalimat dapat memiliki arti yang sama meskipun diparafrase. Gunakan pemahaman semantik untuk menemukan jawaban berdasarkan makna, bukan hanya kemiripan kata secara literal.
                
                                Jika ditemukan beberapa jawaban dari dataset atau dokumen yang berbeda, utamakan jawaban yang berasal dari **dokumen atau file terbaru** (yang memiliki waktu unggah paling baru). Tunjukkan pemahaman yang tepat terhadap konteks saat ini.
                
                                Berikan jawaban yang relevan, ringkas, dan hanya berdasarkan dokumen yang tersedia. Jangan menjawab berdasarkan asumsi atau di luar konteks.
                
                                Jika informasi tidak tersedia dalam konteks, katakan secara formal:
                                **"Terima kasih atas pertanyaan Anda. Saat ini informasi yang Anda cari sedang dalam proses peninjauan dan akan segera dijawab oleh instruktur. Kami menghargai kesabaran Anda dan akan memastikan bahwa pertanyaan Anda akan segera mendapatkan jawaban yang akurat."**
                
                                JANGAN pernah mengarang jawaban. Jangan gunakan tanda bintang (*) atau tanda lain yang tidak formal.
                
                                Gunakan Bahasa Indonesia yang baik dan benar. Pastikan jawaban bersifat informatif, jelas, dan tepat sasaran.
                                
                                Pertanyaan: {content}
                                
                                Jawaban:
                                """
                                
                                response = model.generate_content(prompt)
                                if response and hasattr(response, 'text'):
                                    bot_response = response.text.strip()
                                else:
                                    bot_response = "Maaf, saya mengalami kesulitan memproses permintaan Anda."
                            else:
                                bot_response = "Maaf, saya mengalami kesulitan dengan sistem pengetahuan saya. Silakan coba lagi nanti."
                    elif GEMINI_AVAILABLE:
                        # Direct call to Gemini if QA chain isn't available
                        print("QA chain not available, using direct Gemini call")
                        model = genai.GenerativeModel(MODEL_NAME)
                        prompt = f"""
                        Anda adalah asisten virtual khusus untuk menangani permasalahan terkait konsep, definisi, dan kasus batas Survei Sosial Ekonomi Nasional (Susenas) yang dilaksanakan oleh Badan Pusat Statistik (BPS). Bantu pengguna dengan informasi yang akurat dan detail tentang Susenas berdasarkan konteks yang diberikan.

                        Jangan hanya mencari jawaban yang persis sama dengan pertanyaan pengguna. Pelajari dan parafrase dokumen PDF dan Excel. Pahami bahwa kalimat dapat memiliki arti yang sama meskipun diparafrase. Gunakan pemahaman semantik untuk menemukan jawaban berdasarkan makna, bukan hanya kemiripan kata secara literal.
        
                        Jika ditemukan beberapa jawaban dari dataset atau dokumen yang berbeda, utamakan jawaban yang berasal dari **dokumen atau file terbaru** (yang memiliki waktu unggah paling baru). Tunjukkan pemahaman yang tepat terhadap konteks saat ini.
        
                        Berikan jawaban yang relevan, ringkas, dan hanya berdasarkan dokumen yang tersedia. Jangan menjawab berdasarkan asumsi atau di luar konteks.
        
                        Jika informasi tidak tersedia dalam konteks, katakan secara formal:
                        **"Terima kasih atas pertanyaan Anda. Saat ini informasi yang Anda cari sedang dalam proses peninjauan dan akan segera dijawab oleh instruktur. Kami menghargai kesabaran Anda dan akan memastikan bahwa pertanyaan Anda akan segera mendapatkan jawaban yang akurat."**
        
                        JANGAN pernah mengarang jawaban. Jangan gunakan tanda bintang (*) atau tanda lain yang tidak formal.
        
                        Gunakan Bahasa Indonesia yang baik dan benar. Pastikan jawaban bersifat informatif, jelas, dan tepat sasaran.
                        
                        Pertanyaan: {content}
                        
                        Jawaban:
                        """
                        
                        response = model.generate_content(prompt)
                        if response and hasattr(response, 'text'):
                            bot_response = response.text.strip()
                        else:
                            bot_response = "Maaf, saya mengalami kesulitan memproses permintaan Anda."
                    else:
                        bot_response = "Maaf, sistem pengetahuan saya sedang tidak tersedia saat ini. Silakan coba lagi nanti."
                except Exception as e:
                    print(f"Knowledge base error: {str(e)}")
                    traceback.print_exc()
                    bot_response = f"Maaf, terjadi kesalahan dalam mengakses basis pengetahuan: {str(e)}"
        
        # Save bot response - Using AUTO_INCREMENT for id
        verified = False
        try:
            cursor.execute(
                "INSERT INTO messages (chat_id, message, sender, is_from_corrected) VALUES (%s, %s, %s, %s)",
                (chat_id, bot_response, 'bot', is_from_correction)
            )
            conn.commit()
            
            # Jika jawaban berasal dari koreksi admin, tandai chat ini sebagai verified
            if is_from_correction:
                verified = True
                cursor.execute(
                    "UPDATE chats SET verified = TRUE WHERE id = %s",
                    (chat_id,)
                )
                conn.commit()
        except Exception as e:
            print(f"Error saving bot response: {str(e)}")
            # Continue even if saving fails
        
        print(f"Responded with: {bot_response[:100]}...")
        
        # Return response
        return jsonify({
            "success": True,
            "message": bot_response,
            "verified": verified,
            "is_from_correction": is_from_correction
        }), 200
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing database connection: {str(close_error)}")
    
@app.route('/verify/<chat_id>', methods=['POST'])
@jwt_required()
def verify_chat(chat_id):
    try:
        user_id = get_jwt_identity()
        
        # Check if user is admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user or user.get('role') != 'admin':
            cursor.close()
            conn.close()
            return jsonify({"error": "Admin access required"}), 403
            
        # Get the chat and its messages
        cursor.execute("""
            SELECT c.id, c.user_id, c.created_at, c.verified, c.verified_at,
                   m.id as message_id, m.message, m.sender, m.created_at as message_time
            FROM chats c
            LEFT JOIN messages m ON c.id = m.chat_id
            WHERE c.id = %s
            ORDER BY m.created_at ASC
        """, (chat_id,))
        
        results = cursor.fetchall()
        
        if not results:
            cursor.close()
            conn.close()
            return jsonify({"error": "Chat not found"}), 404
        
        # Update chat verification status
        current_time = datetime.now()
        cursor.execute(
            "UPDATE chats SET verified = TRUE, verified_at = %s, verified_by = %s WHERE id = %s",
            (current_time, user_id, chat_id)
        )
        conn.commit()
        
        # Format the response
        chat = {
            "id": results[0]['id'],
            "user_id": results[0]['user_id'],
            "created_at": results[0]['created_at'].isoformat() if results[0]['created_at'] else None,
            "verified": True,
            "verified_at": current_time.isoformat(),
            "verified_by": user_id
        }
        
        messages = []
        if results[0]['message_id'] is not None:  # Only add messages if they exist
            for row in results:
                messages.append({
                    "id": row['message_id'],
                    "message": row['message'],
                    "sender": row['sender'],
                    "created_at": row['message_time'].isoformat() if row['message_time'] else None
                })
        
        return jsonify({
            "success": True,
            "chat": chat,
            "messages": messages
        }), 200
        
    except Exception as e:
        print(f"Verify chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing database connection: {str(close_error)}")

# cek udah diverif apa belum (3 juni)
@app.route('/chat/<chat_id>/status', methods=['GET'])
@jwt_required()
def chat_status(chat_id):
    user_id = get_jwt_identity()
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT verified FROM chats WHERE id = %s AND user_id = %s", (chat_id, user_id))
    chat = cursor.fetchone()
    cursor.close()
    conn.close()

    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    return jsonify({
        "success": True,
        "verified": chat.get('verified', False)
    }), 200
# last chat cek udah diverif apa belum (3 juni)

@app.route('/admin/chats', methods=['GET'])
@jwt_required()
def admin_chats():
    try:
        current_user = get_jwt_identity()
        
        # Check if user is admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT role FROM users WHERE id = %s", (current_user,))
        user = cursor.fetchone()
        
        if not user or user.get('role') != 'admin':
            cursor.close()
            conn.close()
            return jsonify({"error": "Admin access required"}), 403
        
        # Get all chats with message counts - Fix the GROUP BY issue
        cursor.execute("""
            SELECT c.id, c.user_id, c.created_at, c.verified, u.username,
                  CAST((SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS UNSIGNED) as message_count,
                  (SELECT MAX(m.created_at) FROM messages m WHERE m.chat_id = c.id) as last_message_time,
                  (SELECT message FROM messages m WHERE m.chat_id = c.id AND m.sender = 'bot' ORDER BY m.created_at DESC LIMIT 1) as last_bot_message,
                  (SELECT message FROM messages m WHERE m.chat_id = c.id AND m.sender = 'user' ORDER BY m.created_at DESC LIMIT 1) as last_user_message
            FROM chats c 
            JOIN users u ON c.user_id = u.id
            ORDER BY COALESCE(last_message_time, c.created_at) DESC
        """)
        
        chats = cursor.fetchall()
        
        # Print counts for debugging
        for chat in chats:
            print(f"Chat {chat['id']}: {chat['message_count']} messages, type: {type(chat['message_count'])}")
            # Ensure message_count is always an integer
            if chat['message_count'] is not None:
                chat['message_count'] = int(chat['message_count'])
            else:
                chat['message_count'] = 0
            
            # Truncate long messages for preview
            if chat['last_bot_message'] and len(chat['last_bot_message']) > 100:
                chat['last_bot_message'] = chat['last_bot_message'][:100] + '...'
                
            if chat['last_user_message'] and len(chat['last_user_message']) > 100:
                chat['last_user_message'] = chat['last_user_message'][:100] + '...'
            
        # Convert datetime objects to string
        for chat in chats:
            chat['created_at'] = chat['created_at'].isoformat() if chat['created_at'] else None
            if 'last_message_time' in chat and chat['last_message_time']:
                chat['last_message_time'] = chat['last_message_time'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "chats": chats
        }), 200
        
    except Exception as e:
        print(f"Admin chats error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing database connection: {str(close_error)}")

@app.route('/admin/get_file', methods= ["GET"])
@jwt_required()
def admin_get_file():
    try:
        current_user = get_jwt_identity()
        
        # Check if user is admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT role FROM users WHERE id = %s", (current_user,))
        user = cursor.fetchone()
        
        if not user or user.get('role') != 'admin':
            cursor.close()
            conn.close()
            return jsonify({"error": "Admin access required"}), 403
        
        cursor.execute("""
        SELECT * FROM knowledge_files
        """)

        files = cursor.fetchall()
        
        # if not files:
            # cursor.close()
            # conn.close()
            # return jsonify({"error": "files not found"}), 404
        if not files:
            cursor.close()
            conn.close()
            return jsonify({
                "success": True,
                "files": [],
                "message": "Tidak ada file yang sudah dimasukkan"
            }), 200
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "files": files,
        }), 200
    except Exception as e:
        print(e)

@app.route('/admin/delete/<filename>', methods = ['GET'])
@jwt_required()
def admin_delete_file(filename):
    global qa_chain
    try:
        current_user = get_jwt_identity()
        
        # Check if user is admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT role FROM users WHERE id = %s", (current_user,))
        user = cursor.fetchone()
        
        if not user or user.get('role') != 'admin':
            cursor.close()
            conn.close()
            return jsonify({"error": "Admin access required"}), 403
        
        cursor.execute("""
            DELETE FROM knowledge_files 
            WHERE filename = %s
        """, (filename,))

        conn.commit()

        if cursor.rowcount == 0:  # Check if the deletion was successful
            cursor.close()
            conn.close()
            return jsonify({"error": "File not found"}), 404

        uploads_dir = os.getenv('GITHUB_UPLOADS_PATH', 'uploads')
        file_path = os.path.join(uploads_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)  # Delete the file from local storage
            print(f"File {filename} deleted from local storage.")
        else:
            print(f"File {filename} not found in local storage.")

        # Proses dokumen dan simpan FAISS index ulang
        # Tentukan direktori tempat FAISS index disimpan
        VECTOR_STORE_FOLDER_PATH = os.getenv('VECTOR_STORE_FOLDER_PATH', 'vector_store')

        # Tentukan nama file FAISS index yang ingin dihapus (index.faiss)
        faiss_index_file = os.path.join(VECTOR_STORE_FOLDER_PATH, 'index.faiss')

        # Periksa apakah file FAISS ada, lalu hapus
        if os.path.exists(faiss_index_file):
            os.remove(faiss_index_file)
            print(f"FAISS index file '{faiss_index_file}' telah dihapus.")
        else:
            print("FAISS index file tidak ditemukan.")
            
        documents = process_documents_from_uploads(deleted_filename = filename)

        if documents:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(VECTOR_STORE_FOLDER_PATH)
            print("Vector store updated and saved after upload")
            
            # Perbarui QA chain agar pencarian bisa pakai vector store baru
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            
            PROMPT = PromptTemplate(
                template="""
                Anda adalah asisten virtual khusus untuk menangani permasalahan terkait konsep, definisi, dan kasus batas Survei Sosial Ekonomi Nasional (Susenas) yang dilaksanakan oleh Badan Pusat Statistik (BPS). Bantu pengguna dengan informasi yang akurat dan detail tentang Susenas berdasarkan konteks yang diberikan.

                Jangan hanya mencari jawaban yang persis sama dengan pertanyaan pengguna. Pelajari dan parafrase dokumen PDF dan Excel. Pahami bahwa kalimat dapat memiliki arti yang sama meskipun diparafrase. Gunakan pemahaman semantik untuk menemukan jawaban berdasarkan makna, bukan hanya kemiripan kata secara literal.

                Jika ditemukan beberapa jawaban dari dataset atau dokumen yang berbeda, utamakan jawaban yang berasal dari **dokumen atau file terbaru** (yang memiliki waktu unggah paling baru). Tunjukkan pemahaman yang tepat terhadap konteks saat ini.

                Berikan jawaban yang relevan, ringkas, dan hanya berdasarkan dokumen yang tersedia. Jangan menjawab berdasarkan asumsi atau di luar konteks.

                Jika informasi tidak tersedia dalam konteks, katakan secara formal:
                **"Terima kasih atas pertanyaan Anda. Saat ini informasi yang Anda cari sedang dalam proses peninjauan dan akan segera dijawab oleh instruktur. Kami menghargai kesabaran Anda dan akan memastikan bahwa pertanyaan Anda akan segera mendapatkan jawaban yang akurat."**

                JANGAN pernah mengarang jawaban. Jangan gunakan tanda bintang (*) atau tanda lain yang tidak formal.

                Gunakan Bahasa Indonesia yang baik dan benar. Pastikan jawaban bersifat informatif, jelas, dan tepat sasaran.

                {context}

                Pertanyaan: {question}

                Jawaban yang akurat dan relevan:
                """,
                input_variables=["context", "question"],
            )

            llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.2)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

        else:
            print("No documents processed during upload.")

        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
        }), 200
    
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/admin/delete_github/<path:filename>', methods=['GET'])
@jwt_required()
def admin_delete_file_github(filename):
    global qa_chain, vector_store
    qa_chain = app.config.get('qa_chain')
    vector_store = app.config.get('vector_store')

    try:
        current_user = get_jwt_identity()

        # 1. Cek apakah user admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT role FROM users WHERE id = %s", (current_user,))
        user = cursor.fetchone()

        if not user or user.get('role') != 'admin':
            cursor.close()
            conn.close()
            return jsonify({"error": "Admin access required"}), 403

        # 2. Hapus dari Qdrant berdasarkan metadata source
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout = 60
        )
        collection_name = os.getenv("QDRANT_COLLECTION")

        delete_filter = FilterSelector(
        filter = Filter(
            must=[
                    FieldCondition(
                        key="metadata.file_name",
                        match=MatchValue(value=filename)
                    )
                ]
            )
        )

        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=delete_filter,
            wait=True
        )
        print(f"[Qdrant] Deleted vectors where source == '{filename}'")

        # 3. Hapus dari database
        cursor.execute("DELETE FROM knowledge_files WHERE filename = %s", (filename,))
        conn.commit()
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({"error": "File not found in database"}), 404

        # 4. Hapus dari GitHub
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        github_path = os.getenv("GITHUB_FOLDER_PATH")
        github_api_url = f"https://api.github.com/repos/{repo}/contents/{github_path}/{filename}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }

        get_response = requests.get(github_api_url, headers=headers)
        if get_response.status_code == 200:
            sha = get_response.json().get("sha")
            delete_response = requests.delete(github_api_url, headers=headers, json={
                "message": f"Delete {filename}",
                "sha": sha,
            })
            if delete_response.status_code not in [200, 204]:
                print(f"[GitHub] Failed to delete: {delete_response.text}")
        else:
            print(f"[GitHub] File not found: {get_response.text}")

        cursor.close()
        conn.close()

        # 5. (Opsional) Reset QA Chain
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        PROMPT = PromptTemplate(
                template="""Anda adalah asisten virtual khusus untuk menangani permasalahan terkait konsep, definisi, dan kasus batas Survei Sosial Ekonomi Nasional (Susenas) yang dilaksanakan oleh Badan Pusat Statistik (BPS). Bantu pengguna dengan informasi yang akurat dan detail tentang Susenas berdasarkan konteks yang diberikan.

               Jangan hanya mencari jawaban yang persis sama dengan pertanyaan pengguna. Pelajari dan parafrase dokumen PDF dan Excel. Pahami bahwa kalimat dapat memiliki arti yang sama meskipun diparafrase. Gunakan pemahaman semantik untuk menemukan jawaban berdasarkan makna, bukan hanya kemiripan kata secara literal.

                Jika ditemukan beberapa jawaban dari dataset atau dokumen yang berbeda, utamakan jawaban yang berasal dari **dokumen atau file terbaru** (yang memiliki waktu unggah paling baru). Tunjukkan pemahaman yang tepat terhadap konteks saat ini.

                Berikan jawaban yang relevan, ringkas, dan hanya berdasarkan dokumen yang tersedia. Jangan menjawab berdasarkan asumsi atau di luar konteks.

                Jika informasi tidak tersedia dalam konteks, katakan secara formal:
                **"Terima kasih atas pertanyaan Anda. Saat ini informasi yang Anda cari sedang dalam proses peninjauan dan akan segera dijawab oleh instruktur. Kami menghargai kesabaran Anda dan akan memastikan bahwa pertanyaan Anda akan segera mendapatkan jawaban yang akurat."**

                JANGAN pernah mengarang jawaban. Jangan gunakan tanda bintang (*) atau tanda lain yang tidak formal.

                Gunakan Bahasa Indonesia yang baik dan benar. Pastikan jawaban bersifat informatif, jelas, dan tepat sasaran.

                {context}

                Pertanyaan: {question}

                Jawaban yang akurat dan relevan:""",
                input_variables=["context", "question"]
            )
        llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        app.config['qa_chain'] = qa_chain

        return jsonify({"success": True}), 200

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/chat/<chat_id>', methods=['GET'])
@jwt_required()
def admin_get_chat(chat_id):
    try:
        current_user = get_jwt_identity()
        
        # Check if user is admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT role FROM users WHERE id = %s", (current_user,))
        user = cursor.fetchone()
        
        if not user or user.get('role') != 'admin':
            cursor.close()
            conn.close()
            return jsonify({"error": "Admin access required"}), 403
        
        # Get chat details
        cursor.execute("""
            SELECT c.id, c.user_id, c.created_at, c.verified, u.username
            FROM chats c
            JOIN users u ON c.user_id = u.id
            WHERE c.id = %s
        """, (chat_id,))
        
        chat = cursor.fetchone()
        
        if not chat:
            cursor.close()
            conn.close()
            return jsonify({"error": "Chat not found"}), 404
        
        # Get messages with message IDs
        cursor.execute("""
            SELECT id, message, sender, created_at, is_corrected, is_correction
            FROM messages
            WHERE chat_id = %s
            ORDER BY created_at ASC
        """, (chat_id,))
        
        messages = cursor.fetchall()
        print(f"Retrieved {len(messages)} messages for chat {chat_id}")
        
        # Convert datetime objects to string
        chat['created_at'] = chat['created_at'].isoformat() if chat['created_at'] else None
        for message in messages:
            message['created_at'] = message['created_at'].isoformat() if message['created_at'] else None
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "chat": chat,
            "messages": messages
        }), 200

    except Exception as e:
        print(f"Admin get chat error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing database connection: {str(close_error)}")

# Add a new route for the plural form that was being incorrectly called
@app.route('/admin/chats/<chat_id>', methods=['GET'])
@jwt_required()
def admin_get_chat_plural(chat_id):
    """Alias for admin_get_chat to handle the plural form of the URL"""
    return admin_get_chat(chat_id)

@app.route('/upload_github', methods=['POST'])
@jwt_required()
def upload_file_github():
    # global qa_chain, vector_store
    qa_chain = app.config.get('qa_chain')
    vector_store = app.config.get('vector_store')
    try:
        # 1. Ambil user
        user_id = get_jwt_identity()
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT username, role FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user or user["role"] != "admin":
            return jsonify({"error": "Admin access required"}), 403
        user_name = user["username"]

        # 2. Ambil file
        file = request.files.get('files')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        filename = secure_filename(file.filename)
        file_bytes = file.read()

        # 3. Tentukan tipe file
        ext = filename.lower()
        if ext.endswith('.pdf'):
            file_type = 'pdf'
        elif ext.endswith(('.xlsx', '.xls')):
            file_type = 'excel'
        elif ext.endswith('.csv'):
            file_type = 'csv'
        elif ext.endswith('.txt'):
            file_type = 'text'
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # 4. Ekstraksi konten
        try:
            if file_type == 'pdf':
                content = extract_text_from_pdf(file_bytes, filename)
            elif file_type == 'excel':
                content = extract_data_from_excel(BytesIO(file_bytes), filename)  # list of Documents
            elif file_type in ['csv', 'text']:
                content = file_bytes.decode('utf-8')
            else:
                content = ""
        except Exception as e:
            return jsonify({"error": f"Error extracting content: {str(e)}"}), 500

        # 5. Proses FAISS indexing
        try:
            new_documents = []

            if file_type == 'pdf':
                new_documents = content
            elif file_type == 'excel':
                new_documents = content  # sudah berupa list of Document
            elif file_type in ['csv', 'text']:
                chunks = text_splitter.split_text(content)
                new_documents = [
                    Document(page_content=chunk, metadata={"source": filename, "chunk": i, "type": file_type})
                    for i, chunk in enumerate(chunks)
                ]

            if new_documents:
                # Inisialisasi embedding
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=GOOGLE_API_KEY
                )

                # Inisialisasi Qdrant client
                qdrant_client = QdrantClient(
                    url=os.getenv("QDRANT_URL"),  # contoh: "http://localhost:6333"
                    api_key=os.getenv("QDRANT_API_KEY")  # jika tidak pakai key, bisa None
                )

                # Nama collection di Qdrant
                collection_name = os.getenv("QDRANT_COLLECTION")

                # vector_store = QdrantVectorStore.from_existing_collection(
                #     url=os.getenv("QDRANT_URL"),
                #     collection_name=collection_name,
                #     embedding=embeddings,
                #     api_key=os.getenv("QDRANT_API_KEY"),
                #     prefer_grpc=False,
                #     timeout=30.0
                # )

                if vector_store:
                    # Buat index untuk field 'source' agar bisa digunakan sebagai filter
                    try:
                        qdrant_client.create_payload_index(
                            collection_name=collection_name,
                            field_name="metadata.file_name",
                            field_schema=PayloadSchemaType.KEYWORD
                        )
                    except Exception as e:
                        print(f"[Qdrant] Payload index creation error (may already exist): {e}")

                    # Buat vector store dari Qdrant
                    vector_store.add_documents(new_documents)
                    print("Sukses menambahkan dokumen di Qdrant")
                
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

                PROMPT = PromptTemplate(
                    template="""
                        Anda adalah asisten virtual khusus untuk menangani permasalahan terkait konsep, definisi, dan kasus batas Survei Sosial Ekonomi Nasional (Susenas) yang dilaksanakan oleh Badan Pusat Statistik (BPS). Bantu pengguna dengan informasi yang akurat dan detail tentang Susenas berdasarkan konteks yang diberikan.

                        Jangan hanya mencari jawaban yang persis sama dengan pertanyaan pengguna. Pelajari dan parafrase dokumen PDF dan Excel. Pahami bahwa kalimat dapat memiliki arti yang sama meskipun diparafrase. Gunakan pemahaman semantik untuk menemukan jawaban berdasarkan makna, bukan hanya kemiripan kata secara literal.

                        Jika ditemukan beberapa jawaban dari dataset atau dokumen yang berbeda, utamakan jawaban yang berasal dari **dokumen atau file terbaru** (yang memiliki waktu unggah paling baru). Tunjukkan pemahaman yang tepat terhadap konteks saat ini.

                        Berikan jawaban yang relevan, ringkas, dan hanya berdasarkan dokumen yang tersedia. Jangan menjawab berdasarkan asumsi atau di luar konteks.

                        Jika informasi tidak tersedia dalam konteks, katakan secara formal:
                        **"Terima kasih atas pertanyaan Anda. Saat ini informasi yang Anda cari sedang dalam proses peninjauan dan akan segera dijawab oleh instruktur. Kami menghargai kesabaran Anda dan akan memastikan bahwa pertanyaan Anda akan segera mendapatkan jawaban yang akurat."**

                        JANGAN pernah mengarang jawaban. Jangan gunakan tanda bintang (*) atau tanda lain yang tidak formal.

                        Gunakan Bahasa Indonesia yang baik dan benar. Pastikan jawaban bersifat informatif, jelas, dan tepat sasaran.

                        Konteks:
                        {context}
                        
                        Pertanyaan: {question}
                        
                        Jawaban yang informatif, lengkap, dan presisi:
                    """,
                    input_variables=["context", "question"])

                llm = ChatGoogleGenerativeAI(
                    model=MODEL_NAME,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                app.config['qa_chain'] = qa_chain
                app.config['vector_store'] = vector_store

        except Exception as indexing_err:
            return jsonify({"error": f"Qdrant indexing error: {str(indexing_err)}"}), 500


        # 6. Upload ke GitHub
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        github_path = os.getenv("GITHUB_FOLDER_PATH")
        api_url = f"https://api.github.com/repos/{repo}/contents/{github_path}/{filename}"

        encoded_content = base64.b64encode(file_bytes).decode("utf-8")
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }
        data = {
            "message": f"Upload {filename}",
            "content": encoded_content,
        }

        response = requests.put(api_url, headers=headers, json=data)
        if response.status_code not in [200, 201]:
            return jsonify({"error": f"GitHub upload failed: {response.text}"}), 500

        # 7. Simpan metadata ke database
        try:
            cursor.execute("DESCRIBE knowledge_files")
            columns = cursor.fetchall()
            # column_names = [col['Field'] for col in columns]

            # if 'content' in column_names:
            #     cursor.execute(
            #         "INSERT INTO knowledge_files (filename, file_type, uploaded_by) VALUES (%s, %s, %s)",
            #         (filename, file_type, user_id,)
            #     )
            # else:
            #     cursor.execute(
            #         "INSERT INTO knowledge_files (filename, file_type, uploaded_by) VALUES (%s, %s, %s)",
            #         (filename, file_type, user_id,)
            #     )
            cursor.execute(
                "INSERT INTO knowledge_files (filename, file_type, uploaded_by, created_at) VALUES (%s, %s, %s, %s)",
                (filename, file_type, user_name, datetime.now(), )
            )
            conn.commit()
        except Exception as db_err:
            return jsonify({"error": f"DB Error: {str(db_err)}"}), 500

        # 9. Selesai
        cursor.close()
        conn.close()
        print("File uploaded to github, saved in DB, and FAISS updated")

        return jsonify({
            "success": True,
            "message": "File uploaded to GitHub, saved in DB, and FAISS updated.",
            "file": {
                "name": filename,
                "type": file_type
            }
        }), 200

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_file():
    global qa_chain
    try:
        user_id = get_jwt_identity()

        # Cek admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user or user.get('role') != 'admin':
            cursor.close()
            conn.close()
            return jsonify({"error": "Admin access required for uploading"}), 403

        # Cek file yang diunggah
        file = request.files.get('file')
        if not file:
            cursor.close()
            conn.close()
            return jsonify({"error": "No file uploaded"}), 400

        # Simpan file di direktori uploads
        uploads_dir = os.getenv('UPLOADS_FOLDER_PATH')
        os.makedirs(uploads_dir, exist_ok=True)
        secure_name = secure_filename(file.filename)
        file_path = os.path.join(uploads_dir, secure_name)
        file.save(file_path)

        # Tentukan tipe file
        file_type = 'unknown'
        if secure_name.lower().endswith('.pdf'):
            file_type = 'pdf'
        elif secure_name.lower().endswith(('.xlsx', '.xls')):
            file_type = 'excel'
        elif secure_name.lower().endswith('.csv'):
            file_type = 'csv'
        elif secure_name.lower().endswith('.txt'):
            file_type = 'text'

        # Proses file untuk mengekstrak konten berdasarkan tipe file
        content = None
        try:
            if file_type == 'pdf':
                content = extract_text_from_pdf(file_path, secure_name)
            elif file_type == 'excel':
                excel_data = extract_data_from_excel(file_path)
                content = json.dumps(excel_data, ensure_ascii=False, default=str)
            elif file_type in ['csv', 'text']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
        except Exception as e:
            content = f"Error processing file: {str(e)}"

        # Simpan file di database
        try:
            cursor.execute("DESCRIBE knowledge_files")
            columns = cursor.fetchall()
            column_names = [col['Field'] for col in columns]

            if 'content' in column_names:
                cursor.execute(
                    "INSERT INTO knowledge_files (filename, file_type, content, uploaded_by) VALUES (%s, %s, %s, %s)",
                    (secure_name, file_type, content, user_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO knowledge_files (filename, file_type, uploaded_by) VALUES (%s, %s, %s)",
                    (secure_name, file_type, user_id)
                )

            conn.commit()

        except Exception as e:
            cursor.close()
            conn.close()
            return jsonify({"error": f"Database error: {str(e)}"}), 500

        # Proses dokumen dan simpan FAISS index ulang
        documents = process_documents_from_uploads()

        if documents:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(VECTOR_STORE_FOLDER_PATH)
            print("Vector store updated and saved after upload")
            
            # Perbarui QA chain agar pencarian bisa pakai vector store baru
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            
            PROMPT = PromptTemplate(
                template="""
                Anda adalah asisten virtual khusus untuk menangani permasalahan terkait konsep, definisi, dan kasus batas Survei Sosial Ekonomi Nasional (Susenas) yang dilaksanakan oleh Badan Pusat Statistik (BPS). Bantu pengguna dengan informasi yang akurat dan detail tentang Susenas berdasarkan konteks yang diberikan.

                Jangan hanya mencari jawaban yang persis sama dengan pertanyaan pengguna. Pelajari dan parafrase dokumen PDF dan Excel. Pahami bahwa kalimat dapat memiliki arti yang sama meskipun diparafrase. Gunakan pemahaman semantik untuk menemukan jawaban berdasarkan makna, bukan hanya kemiripan kata secara literal.

                Jika ditemukan beberapa jawaban dari dataset atau dokumen yang berbeda, utamakan jawaban yang berasal dari **dokumen atau file terbaru** (yang memiliki waktu unggah paling baru). Tunjukkan pemahaman yang tepat terhadap konteks saat ini.

                Berikan jawaban yang relevan, ringkas, dan hanya berdasarkan dokumen yang tersedia. Jangan menjawab berdasarkan asumsi atau di luar konteks.

                Jika informasi tidak tersedia dalam konteks, katakan secara formal:
                **"Terima kasih atas pertanyaan Anda. Saat ini informasi yang Anda cari sedang dalam proses peninjauan dan akan segera dijawab oleh instruktur. Kami menghargai kesabaran Anda dan akan memastikan bahwa pertanyaan Anda akan segera mendapatkan jawaban yang akurat."**

                JANGAN pernah mengarang jawaban. Jangan gunakan tanda bintang (*) atau tanda lain yang tidak formal.

                Gunakan Bahasa Indonesia yang baik dan benar. Pastikan jawaban bersifat informatif, jelas, dan tepat sasaran.

                {context}

                Pertanyaan: {question}

                Jawaban yang akurat dan relevan:
                """,
                input_variables=["context", "question"],
            )

            llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.2)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

        else:
            print("No documents processed during upload.")

        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "message": "File uploaded, saved in database, and RAG generated successfully.",
            "file": {
                "name": secure_name,
                "type": file_type
            }
        }), 200

    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Alias route for admin/upload
@app.route('/admin/upload', methods=['POST'])
@jwt_required()
def admin_upload_file():
    """Alias for upload_file"""
    return upload_file()

@app.route('/test', methods=['GET', 'POST'])
def test():
    """Simple test endpoint that doesn't require authentication"""
    return jsonify({
        "success": True,
        "message": "Test endpoint working",
        "method": request.method
    })

@app.route('/correct/<message_id>', methods=['POST'])
@jwt_required()
def correct_message(message_id):
    try:
        user_id = get_jwt_identity()
        # Terima parameter 'correction' atau 'corrected_message' dari frontend
        correction = request.json.get('correction') or request.json.get('corrected_message')
        
        if not correction:
            return jsonify({"error": "Correction text is required"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user or user.get('role') != 'admin':
            cursor.close()
            conn.close()
            return jsonify({"error": "Admin access required"}), 403
        
        # Get the original message
        cursor.execute(
            "SELECT chat_id, message, sender FROM messages WHERE id = %s", 
            (message_id,)
        )
        
        message = cursor.fetchone()
        if not message:
            cursor.close()
            conn.close()
            return jsonify({"error": "Message not found"}), 404
            
        if message['sender'] != 'bot':
            cursor.close()
            conn.close()
            return jsonify({"error": "Only bot messages can be corrected"}), 400
        
        # Mark original message as corrected
        cursor.execute(
            "UPDATE messages SET is_corrected = TRUE WHERE id = %s",
            (message_id,)
        )
        
        # Sederhanakan proses dengan INSERT tanpa menggunakan custom UUID - gunakan AUTO_INCREMENT
        # yang sudah ada di tabel messages
        cursor.execute(
            """INSERT INTO messages 
                (chat_id, message, sender, is_correction, corrected_message_id, corrected_by) 
                VALUES (%s, %s, 'bot', TRUE, %s, %s)""",
            (message['chat_id'], correction, message_id, user_id)
        )
        
        # Get the auto-generated ID
        cursor.execute("SELECT LAST_INSERT_ID()")
        result = cursor.fetchone()
        correction_id = result['LAST_INSERT_ID()'] if result else None
        
        conn.commit()
        
        # Also update the chat to mark it as verified
        try:
            cursor.execute(
                "UPDATE chats SET verified = TRUE, verified_at = %s, verified_by = %s WHERE id = %s",
                (datetime.now(), user_id, message['chat_id'])
            )
            conn.commit()
        except Exception as verify_error:
            print(f"Warning: Could not update chat verification status: {str(verify_error)}")
        
        return jsonify({
            "success": True,
            "message": {
                "corrected_message": correction,
                "id": correction_id
            },
            "correction_id": correction_id
        }), 200
        
    except Exception as e:
        print(f"Correction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing database connection: {str(close_error)}")
    
@app.route('/chats', methods=['GET'])
@jwt_required()
def get_chats():
    try:
        user_id = get_jwt_identity()
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get all user's chats
        cursor.execute(
            """
            SELECT c.id, c.title, c.created_at, c.verified, c.verified_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) as message_count,
                   (SELECT message FROM messages m WHERE m.chat_id = c.id ORDER BY m.created_at DESC LIMIT 1) as last_message
            FROM chats c
            WHERE c.user_id = %s
            ORDER BY c.created_at DESC
            """,
            (user_id,)
        )
        
        chats = cursor.fetchall()
        
        # Format the response
        formatted_chats = []
        for chat in chats:
            formatted_chat = {
                "id": chat['id'],
                "title": chat['title'],
                "created_at": chat['created_at'].isoformat() if chat['created_at'] else None,
                "verified": bool(chat['verified']),
                "verified_at": chat['verified_at'].isoformat() if chat['verified_at'] else None,
                "message_count": chat['message_count'],
                "last_message": chat['last_message']
            }
            formatted_chats.append(formatted_chat)
        
        return jsonify({
            "success": True,
            "chats": formatted_chats
        }), 200
        
    except Exception as e:
        print(f"Get chats error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/chat/<chat_id>/messages', methods=['GET'])
@jwt_required()
def get_chat_messages(chat_id):
    try:
        user_id = get_jwt_identity()
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Verify chat belongs to user
        cursor.execute(
            "SELECT id FROM chats WHERE id = %s AND user_id = %s",
            (chat_id, user_id)
        )
        
        chat = cursor.fetchone()
        if not chat:
            cursor.close()
            conn.close()
            return jsonify({"error": "Chat not found or access denied"}), 404
        
        # Get messages
        cursor.execute(
            """SELECT id, message, sender, created_at, is_corrected, is_correction,
                  corrected_message_id 
               FROM messages 
               WHERE chat_id = %s 
               ORDER BY created_at ASC""",
            (chat_id,)
        )
        
        messages = cursor.fetchall()
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted_message = {
                "id": msg['id'],
                "message": msg['message'],
                "sender": msg['sender'],
                "created_at": msg['created_at'].isoformat() if msg['created_at'] else None,
                "is_corrected": bool(msg['is_corrected']),
                "is_correction": bool(msg['is_correction']),
                "corrected_message_id": msg['corrected_message_id']
            }
            formatted_messages.append(formatted_message)
        
        return jsonify({
            "success": True,
            "chat_id": chat_id,
            "messages": formatted_messages
        }), 200
        
    except Exception as e:
        print(f"Get chat messages error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing database connection: {str(close_error)}")

@app.route('/schema/check', methods=['GET'])
def check_schema():
    """Verify and update the database schema if needed"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if messages table has auto_increment for id
        cursor.execute("SHOW CREATE TABLE messages")
        table_schema = cursor.fetchone()[1]
        
        # Check if we need to modify the schema
        if 'AUTO_INCREMENT' not in table_schema or 'id` VARCHAR(36)' in table_schema:
            print("Updating messages table schema to use AUTO_INCREMENT INT primary key")
            try:
                # Drop the existing primary key constraint
                cursor.execute("ALTER TABLE messages DROP PRIMARY KEY")
                
                # Modify the id column to be INT AUTO_INCREMENT
                cursor.execute("ALTER TABLE messages MODIFY id INT AUTO_INCREMENT PRIMARY KEY")
                conn.commit()
                print("Successfully updated messages table schema")
            except Exception as alter_error:
                print(f"Error updating messages table schema: {str(alter_error)}")
                # If the update fails, create a new table with correct schema and migrate data
                try:
                    cursor.execute("""
                    CREATE TABLE messages_new (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        chat_id VARCHAR(36) NOT NULL,
                        sender VARCHAR(10) NOT NULL,
                        message TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_corrected BOOLEAN DEFAULT FALSE,
                        is_correction BOOLEAN DEFAULT FALSE,
                        corrected_message_id INT,
                        corrected_by VARCHAR(36),
                        FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
                        FOREIGN KEY (corrected_by) REFERENCES users(id) ON DELETE SET NULL
                    )
                    """)
                    
                    # Copy existing data if any
                    cursor.execute("""
                    INSERT INTO messages_new 
                    (chat_id, sender, message, created_at, is_corrected, is_correction, corrected_message_id, corrected_by)
                    SELECT chat_id, sender, message, created_at, is_corrected, is_correction, corrected_message_id, corrected_by
                    FROM messages
                    """)
                    
                    # Replace old table with new one
                    cursor.execute("DROP TABLE messages")
                    cursor.execute("RENAME TABLE messages_new TO messages")
                    conn.commit()
                    print("Created new messages table with correct schema and migrated data")
                except Exception as migration_error:
                    print(f"Error migrating messages data: {str(migration_error)}")
                    conn.rollback()
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error checking schema: {str(e)}")
        print(traceback.format_exc())
        return False

@app.route('/test/message', methods=['POST'])
def test_message():
    """Simple test endpoint for messaging without JWT requirement"""
    try:
        data = request.json
        message = data.get('message', 'No message provided')
        
        return jsonify({
            "success": True,
            "message": "Test message received",
            "your_message": message,
            "response": f"You said: {message}",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/debug/request', methods=['GET', 'POST', 'OPTIONS'])
def debug_request():
    """Print and return request details for debugging"""
    try:
        # Get request details
        method = request.method
        headers = dict(request.headers)
        args = dict(request.args)
        
        # Safe extraction of body content
        body = None
        if request.data:
            try:
                if 'application/json' in request.headers.get('Content-Type', ''):
                    body = request.get_json(silent=True)
                else:
                    body = request.data.decode('utf-8')
            except:
                body = "Could not decode request body"
        
        # Get form data if present
        form_data = dict(request.form) if request.form else None
        
        # Construct response
        response_data = {
            "success": True,
            "request": {
                "method": method,
                "path": request.path,
                "headers": headers,
                "query_args": args,
                "body": body,
                "form_data": form_data
            }
        }
        
        # Log the details
        print(f"DEBUG REQUEST: {method} {request.path}")
        print(f"Headers: {headers}")
        print(f"Body: {body}")
        print(f"Form: {form_data}")
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 500

@app.route('/chat/quick', methods=['GET'])
@jwt_required()
def quick_create_chat():
    """Create a new chat using GET request, no body needed"""
    try:
        user_id = get_jwt_identity()
        print(f"Creating quick chat for user: {user_id}")
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Generate chat ID
        # chat_id = str(uuid.uuid4())
        # print(f"Generated quick chat ID: {chat_id}")
        
        # Insert chat with bare minimum fields
        try:
            cursor.execute(
                "INSERT INTO chats (user_id, created_at) VALUES (%s, %s)",
                (user_id, datetime.now(),)
            )
            conn.commit()
            
            chat_id = cursor.lastrow
            print(f"Quick chat created successfully: {chat_id}")
            
            # Return success with CORS headers explicitly set
            response = jsonify({
                "success": True,
                "chat_id": chat_id,
                "message": "Chat created successfully"
            })
            return response, 201
            
        except Exception as db_error:
            print(f"Database error creating chat: {str(db_error)}")
            return jsonify({"error": f"Failed to create chat: {str(db_error)}"}), 500
            
    except Exception as e:
        print(f"Quick chat creation error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to create quick chat: {str(e)}"}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing database connection: {str(close_error)}")

@app.route('/debug/frontend-issue', methods=['GET', 'POST'])
def frontend_issue_tracker():
    """Endpoint to track frontend issues with chat creation"""
    try:
        # Log as much info as possible about the request
        data = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "path": request.path,
            "headers": dict(request.headers),
            "remote_addr": request.remote_addr,
            "user_agent": request.user_agent.string,
            "is_xhr": request.is_xhr,
            "url": request.url,
            "query_string": request.query_string.decode('utf-8')
        }
        
        # Check for JSON body
        if 'application/json' in request.headers.get('Content-Type', ''):
            try:
                data["json_body"] = request.get_json(silent=True)
            except:
                data["json_body"] = "Error parsing JSON"
                
        # Check for form data
        if request.form:
            data["form_data"] = dict(request.form)
            
        # Check for query params
        if request.args:
            data["query_params"] = dict(request.args)
            
        # Try to extract auth token
        auth_header = request.headers.get('Authorization', '')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            data["auth_token_present"] = True
            data["auth_token_length"] = len(token)
            try:
                # from flask_jwt_extended import decode_token
                decoded = decode_token(token)
                data["token_identity"] = decoded['sub']
                data["token_valid"] = True
            except Exception as token_error:
                data["token_error"] = str(token_error)
                data["token_valid"] = False
        else:
            data["auth_token_present"] = False
            
        # Log the data for debugging
        print("FRONTEND ISSUE DATA:")
        print(json.dumps(data, indent=2))
        
        # Return useful information
        return jsonify({
            "success": True,
            "message": "Issue data logged successfully",
            "issue_type": request.args.get('type', 'unknown'),
            "data": data
        })
        
    except Exception as e:
        print(f"Error tracking frontend issue: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# @app.after_request
# def after_request(response):
#     """Add CORS headers to every response"""
#     origin = request.headers.get('Origin', '*')
#     response.headers.add('Access-Control-Allow-Origin', origin)
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

@app.after_request
def after_request(response):
    """Tambahkan header CORS ke setiap respons"""
    response.headers["Access-Control-Allow-Origin"] = "https://senadi.vercel.app"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response
    
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_preflight(path):
    """Handle CORS preflight requests for any route"""
    response = app.make_default_options_response()
    return response

@app.route('/chat/simpleNew', methods=['GET'])
def simple_create_chat():
    """Create a new chat with token in query parameters, no JWT required"""
    try:
        # Get token from query parameter
        token = request.args.get('token')
        
        if not token:
            print("No token provided")
            return jsonify({"error": "No token provided"}), 401
            
        # Manually decode JWT token
        try:
            # from flask_jwt_extended import decode_token
            decoded = decode_token(token)
            user_id = decoded['sub']  # 'sub' contains the identity in JWT
            print(f"Decoded token for user_id: {user_id}")
        except Exception as e:
            print(f"Token decoding error: {str(e)}")
            return jsonify({"error": "Invalid token"}), 401
            
        print(f"Creating simple chat for user: {user_id}")
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Verify user exists
        cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return jsonify({"error": "User not found"}), 404
            
        # Generate chat ID
        # chat_id = str(uuid.uuid4())
        # print(f"Generated chat ID: {chat_id}")
        
        # Create chat with minimal fields, without 'title' column
        try:
            cursor.execute(
                "INSERT INTO chats (user_id, created_at) VALUES (%s, %s)",
                (user_id, datetime.now(),)
            )
            conn.commit()
            
            chat_id = cursor.lastrow
            print(f"Simple chat created successfully: {chat_id}")
            
            return jsonify({
                "success": True,
                "chat_id": chat_id,
                "message": "Chat created successfully"
            }), 201
            
        except Exception as db_error:
            print(f"Database error creating chat: {str(db_error)}")
            return jsonify({"error": f"Failed to create chat: {str(db_error)}"}), 500
            
    except Exception as e:
        print(f"Simple chat creation error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except Exception as close_error:
            print(f"Error closing connection: {str(close_error)}")

# Endpoint to rebuild vector store (for admin use)
@app.route('/rebuild-knowledge', methods=['POST'])
@jwt_required()
def rebuild_knowledge():
    try:
        user_id = get_jwt_identity()
        
        # Check if user is admin
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user or user['role'] != 'admin':
            return jsonify({"error": "Admin access required"}), 403
        
        # First try to delete the existing vector store files
        try:
            for file in os.listdir(VECTOR_STORE_FOLDER_PATH):
                file_path = os.path.join(VECTOR_STORE_FOLDER_PATH, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
        except Exception as del_error:
            print(f"Error cleaning vector store directory: {str(del_error)}")
        
        # Reinitialize the vector store
        success = initialize_vector_store()
        
        if success:
            return jsonify({
                "success": True,
                "message": "Knowledge base rebuilt successfully"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Failed to rebuild knowledge base"
            }), 500
    
    except Exception as e:
        print(f"Error rebuilding knowledge base: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
        except:
            pass

@app.route("/")
def hello():
    return "Halo railway!"

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
    # app.run(debug=True)
    # pass
