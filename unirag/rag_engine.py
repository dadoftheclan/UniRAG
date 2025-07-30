# unirag/rag_engine.py
"""
🎓 EDUCATIONAL RAG ENGINE
=========================

This module demonstrates core concepts in Retrieval-Augmented Generation (RAG):
1. Document loading and preprocessing
2. Text chunking strategies
3. Vector embedding generation
4. Vector database operations
5. Incremental updates and change detection
6. Retrieval-based question answering

Key Learning Concepts:
- How text becomes mathematical vectors (embeddings)
- Why chunking matters for retrieval quality
- How vector databases enable semantic search
- Efficient update strategies for production systems
"""

import os
import logging
import hashlib
import json

# LangChain imports - the framework that connects all these pieces together
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# Configure logging to show what's happening under the hood
# This helps learners see each step of the RAG process
logging.basicConfig(level=logging.INFO, format="🔍 %(message)s")

# 📁 CONFIGURATION CONSTANTS
# =========================
# These paths define where everything lives - customize for your project!
DB_DIR = "unirag/chroma_db"              # Vector database storage location
DOCS_DIR = "unirag/docs"                 # Where to look for source documents  
MANIFEST_PATH = "unirag/vector_manifest.json"  # Tracks document changes

def load_documents(path=DOCS_DIR):
    """
    📚 DOCUMENT LOADING FUNCTION
    ============================
    
    This function demonstrates how RAG systems ingest various document types.
    It's designed to be modular - you can easily add new document types here.
    
    Learning Points:
    - Different file types need different loaders
    - Glob patterns help filter specific file types
    - Error handling prevents crashes when documents are malformed
    
    Args:
        path (str): Directory to scan for documents
        
    Returns:
        list: Loaded document objects with content and metadata
        
    🔧 CUSTOMIZATION IDEAS:
    - Add more file types (docx, html, csv)
    - Filter by file size or date
    - Add custom metadata extraction
    """
    logging.info(f"Loading documents from: {path}")
    docs = []

    try:
        # 📄 Create loaders for different file types
        # Each loader understands how to extract text from its format
        txt_loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
        md_loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader)
        pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)

        # 🔄 Process each loader type
        for loader in [txt_loader, md_loader, pdf_loader]:
            loaded = loader.load()
            logging.info(f"  ↳ {len(loaded)} documents loaded.")
            docs.extend(loaded)

    except Exception as e:
        logging.error(f"❌ Failed to load documents: {e}")
        raise

    return docs

def get_manifest():
    """
    📋 MANIFEST MANAGEMENT - CHANGE DETECTION SYSTEM
    ===============================================
    
    The manifest tracks which documents we've already processed using MD5 hashes.
    This enables incremental updates - only reprocess changed documents!
    
    Learning Points:
    - Hashing allows efficient change detection
    - JSON provides human-readable storage
    - Graceful handling when manifest doesn't exist yet
    
    Returns:
        dict: Document hash -> metadata mapping
    """
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(manifest):
    """
    💾 SAVE MANIFEST TO DISK
    =======================
    
    Persists the document tracking information for future runs.
    The indent=2 makes it human-readable for debugging/learning.
    
    Args:
        manifest (dict): Document tracking data to save
    """
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def hash_chunk(doc):
    """
    🔐 DOCUMENT FINGERPRINTING
    =========================
    
    Creates a unique identifier for each document chunk using MD5 hashing.
    This is how we detect if content has changed between runs.
    
    Learning Points:
    - MD5 creates a unique "fingerprint" for text content
    - Small changes in text = completely different hash
    - Hashes are much smaller than storing full text for comparison
    
    Args:
        doc: Document object with page_content attribute
        
    Returns:
        str: MD5 hash of the document content
        
    💡 EXPERIMENT: Try changing a single character in a document and see 
       how the hash changes completely!
    """
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

def build_vector_store(documents):
    """
    🧠 VECTOR STORE CONSTRUCTION - THE HEART OF RAG
    ==============================================
    
    This function demonstrates the core RAG workflow:
    1. Split documents into chunks (why? for better retrieval precision)
    2. Generate embeddings (convert text -> mathematical vectors)
    3. Store in vector database (enables fast similarity search)
    4. Create manifest (track what we've processed)
    
    Learning Points:
    - Why chunking: Balance between context and precision
    - Embeddings: How AI understands text mathematically
    - Vector databases: Enable semantic (meaning-based) search
    
    Args:
        documents (list): Loaded document objects
        
    Returns:
        Chroma: Vector store ready for querying
        
    🔧 EXPERIMENT WITH:
    - chunk_size: Try 500, 1500, 2000 - how does it affect answers?
    - chunk_overlap: Try 0, 100, 400 - what's the sweet spot?
    - model: Try different embedding models in Ollama
    """
    logging.info("Splitting documents into chunks...")
    
    # 📝 TEXT CHUNKING STRATEGY
    # RecursiveCharacterTextSplitter tries to keep related content together
    # by splitting on paragraphs first, then sentences, then words
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 🔧 TUNE THIS: Larger = more context, slower retrieval
        chunk_overlap=200     # 🔧 TUNE THIS: Prevents information loss at boundaries
    )
    chunks = splitter.split_documents(documents)
    logging.info(f"  ↳ Total chunks: {len(chunks)}")

    # 🧮 EMBEDDING GENERATION
    # This is where the magic happens - text becomes mathematical vectors!
    # The "llama3" model converts each chunk into a list of numbers that 
    # capture its semantic meaning
    logging.info("Generating embeddings and building vector store...")
    embeddings = OllamaEmbeddings(model="llama3")  # 🔧 TRY: mistral, codellama
    
    # 🗄️ VECTOR DATABASE CREATION
    # Chroma stores our embeddings and enables fast similarity search
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR  # Saves to disk for future use
    )

    # 📋 CREATE MANIFEST FOR CHANGE TRACKING
    # Maps each chunk's hash to its source file
    manifest = {hash_chunk(c): {"source": c.metadata.get("source", "")} for c in chunks}
    save_manifest(manifest)

    logging.info("Vector store built.")
    return vectorstore

def update_vector_store_with_new_or_modified_docs():
    """
    🔄 INCREMENTAL UPDATE SYSTEM - PRODUCTION-READY EFFICIENCY
    ========================================================
    
    This function demonstrates how production RAG systems handle updates efficiently.
    Instead of rebuilding everything, we only process changed documents!
    
    Learning Points:
    - Change detection using file hashing
    - Selective deletion and re-insertion
    - Maintaining data consistency
    - Production-grade update strategies
    
    Algorithm:
    1. Load current documents and create chunks
    2. Compare hashes against manifest (what changed?)
    3. Remove old versions of changed documents
    4. Add new/updated chunks
    5. Update manifest
    
    🏗️ PRODUCTION CONCEPT: This pattern is used by enterprise systems
       processing millions of documents where full rebuilds are impossible.
    """
    # 📚 Load and chunk current documents
    docs = load_documents()
    if not docs:
        logging.warning("No documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    if not chunks:
        logging.warning("No document chunks to process.")
        return

    # 🧮 Initialize embeddings and connect to existing vector store
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # 📋 CHANGE DETECTION ALGORITHM
    manifest = get_manifest()
    current_hashes = {hash_chunk(chunk): chunk for chunk in chunks}
    existing_hashes = set(manifest.keys())

    # 🆕 Find what's new or changed
    new_hashes = set(current_hashes.keys()) - existing_hashes
    modified_chunks = [current_hashes[h] for h in new_hashes]

    if not modified_chunks:
        logging.info("🔁 No new or changed documents to embed.")
        return

    # 🗑️ SMART DELETION: Remove old versions of changed documents
    # This prevents duplicate/outdated information in the vector store
    changed_sources = set(chunk.metadata.get("source", "") for chunk in modified_chunks)
    for h, entry in list(manifest.items()):
        if entry["source"] in changed_sources:
            vectorstore.delete(ids=[h])  # Remove from vector DB
            manifest.pop(h, None)        # Remove from manifest

    # ➕ ADD NEW/UPDATED CONTENT
    vectorstore.add_documents(modified_chunks)
    for chunk in modified_chunks:
        h = hash_chunk(chunk)
        manifest[h] = {"source": chunk.metadata.get("source", "")}

    save_manifest(manifest)
    logging.info(f"✅ Updated vector store with {len(modified_chunks)} new/modified chunks.")

def get_qa_chain():
    """
    🤖 QA CHAIN ASSEMBLY - BRINGING IT ALL TOGETHER
    ==============================================
    
    This function creates the complete RAG pipeline that can answer questions.
    It combines retrieval (finding relevant chunks) with generation (LLM answering).
    
    Learning Points:
    - RAG = Retrieval + Augmented + Generation
    - Vector search finds relevant context
    - LLM generates final answer using that context
    - Chain abstraction simplifies complex workflows
    
    Returns:
        RetrievalQA: Complete question-answering system
        
    🧠 THE RAG PROCESS:
    1. User asks question
    2. Question is embedded into vector space
    3. Vector database finds similar chunks (retrieval)
    4. Chunks + question sent to LLM (augmented generation)
    5. LLM generates contextual answer
    
    🔧 TUNING PARAMETERS:
    - k=5: Number of chunks to retrieve (try 3-10)
    - return_source_documents: Set True to see what was retrieved
    """
    # 🏗️ BUILD OR UPDATE VECTOR STORE
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        logging.info("Vector DB not found. Building from scratch...")
        docs = load_documents()
        if not docs:
            raise ValueError("❌ No documents found in the docs/ directory.")
        build_vector_store(docs)
    else:
        # 🔄 Smart incremental updates for efficiency
        update_vector_store_with_new_or_modified_docs()

    # 🗄️ CONNECT TO VECTOR DATABASE
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=OllamaEmbeddings(model="llama3")
    )
    
    # 🎯 CONFIGURE RETRIEVAL
    # This determines how many relevant chunks to find for each question
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # 🔧 EXPERIMENT: Try 3, 8, 10 - how does it change answers?
    )
    
    # 🧠 INITIALIZE LANGUAGE MODEL
    # This is what actually generates the final answers
    llm = OllamaLLM(model="llama3")  # 🔧 TRY: mistral, codellama, or other models

    # ⛓️ CREATE THE COMPLETE RAG CHAIN
    # RetrievalQA automatically handles the retrieve -> augment -> generate workflow
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False  # 🔧 SET TO TRUE: See which chunks were used!
    )

# 🎓 LEARNING EXERCISES FOR STUDENTS:
# ==================================
# 
# 1. BEGINNER: Change chunk_size to 500 and 2000. How do answers change?
# 
# 2. INTERMEDIATE: Set return_source_documents=True and examine what gets retrieved
# 
# 3. ADVANCED: Add a new document loader for a different file type
# 
# 4. EXPERT: Implement a custom similarity function or add metadata filtering
# 
# 5. RESEARCH: Try different embedding models and compare retrieval quality