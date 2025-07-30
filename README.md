# 🦄 UniRAG - Educational Vector Database & RAG System

A hands-on educational project demonstrating **Retrieval-Augmented Generation (RAG)** and **Vector Database** concepts using unicorn-themed content. Perfect for learning how modern AI systems store, search, and retrieve knowledge from large document collections.

## 🎓 Educational Goals

This project teaches you:

- **Vector Embeddings**: How text is converted into mathematical representations
- **Semantic Search**: Finding relevant information based on meaning, not just keywords
- **RAG Architecture**: Combining retrieval systems with language models
- **Vector Databases**: Efficient storage and querying of high-dimensional data
- **Incremental Updates**: Smart document change detection and processing
- **Real-world AI Workflows**: From document ingestion to conversational AI

## 🧠 What You'll Learn About Vector Databases

### Core Concepts

**Vector Embeddings**: Text documents are converted into numerical vectors (arrays of numbers) that capture semantic meaning. Similar concepts cluster together in high-dimensional space.

**Similarity Search**: Instead of exact text matching, vector databases use mathematical distance (cosine similarity, euclidean distance) to find semantically related content.

**Chunking Strategy**: Large documents are split into smaller, overlapping pieces to balance context preservation with retrieval precision.

### Why This Matters

Modern AI systems like ChatGPT, Claude, and others use similar architectures to:
- Answer questions about specific document collections
- Provide factual information from knowledge bases  
- Create domain-specific chatbots and assistants
- Power search engines and recommendation systems

## ✨ Technical Features

- **Multi-format Document Processing**: Handles `.txt`, `.md`, and `.pdf` files
- **Smart Incremental Updates**: MD5-based change detection - only reprocesses modified content
- **Persistent Vector Storage**: Uses Chroma database for efficient embedding storage
- **Conversational Memory**: Maintains context across multiple questions
- **Local AI Models**: Runs entirely offline using Ollama's Llama3
- **Educational Logging**: Detailed console output showing each step of the process

## 🔬 How Vector Databases Work (Illustrated)

```
1. Document Ingestion
   📄 "Unicorns are magical creatures..." 
   ↓
2. Text Chunking
   📝 Split into 1000-character overlapping pieces
   ↓  
3. Embedding Generation
   🔢 Convert to vectors: [0.1, -0.3, 0.7, ...]
   ↓
4. Vector Storage
   🗄️ Store in Chroma database with metadata
   ↓
5. Query Processing
   ❓ "What are unicorns?" → [0.2, -0.1, 0.8, ...]
   ↓
6. Similarity Search
   🎯 Find closest vectors using cosine similarity
   ↓
7. Context Retrieval + Generation
   🤖 Feed relevant chunks to LLM for final answer
```

## 🚀 Getting Started

### Prerequisites

Understanding this project requires basic knowledge of:
- Python programming
- Windows Command Prompt or PowerShell
- Basic AI/ML concepts (helpful but not required)

**Software Requirements:**
- Windows 10/11
- Python 3.8+ (from [python.org](https://www.python.org/downloads/))
- [Ollama for Windows](https://ollama.ai/download/windows) for local AI models

### Installation

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - ✅ **Important**: Check "Add Python to PATH" during installation

2. **Setup Ollama and Models**:
   - Download [Ollama for Windows](https://ollama.ai/download/windows)
   - Run the installer (requires restart)
   - Open Command Prompt or PowerShell and run:
   ```cmd
   ollama pull llama3
   ```
   *(This downloads the AI model - will take 5-10 minutes depending on internet speed)*

3. **Clone and Setup Project**:
   ```cmd
   git clone https://github.com/yourusername/unirag.git
   cd unirag
   pip install -r requirements.txt
   ```
   
   *If you don't have Git installed, you can download the project as a ZIP file from GitHub instead.*

4. **Create Document Folder**:
   ```cmd
   mkdir unirag\docs
   ```
   Add your unicorn-themed documents here for testing. Try different file types: `.txt`, `.md`, `.pdf`

### Running the Demo

```cmd
python unirag.py
```

**Troubleshooting Windows Setup:**
- If `python` command not found: Try `py` instead of `python`
- If `pip` command not found: Try `py -m pip install -r requirements.txt`
- If Ollama not found: Restart your Command Prompt after installation

Watch the console output to see:
- Document loading and processing
- Embedding generation progress
- Vector database construction
- Query processing in real-time

## 📚 Learning Exercises

### Beginner Exercises

1. **Document Comparison**: Add similar documents and see how the system groups related content
2. **Query Experiments**: Try different question styles and observe retrieval differences
3. **Chunk Analysis**: Modify chunk sizes and see how it affects answer quality

### Intermediate Exercises

1. **Embedding Exploration**: Examine the vector manifest to understand document fingerprinting
2. **Retrieval Tuning**: Adjust the `k` parameter to retrieve more/fewer chunks
3. **Model Comparison**: Try different Ollama models and compare performance

### Advanced Exercises

1. **Custom Similarity Metrics**: Implement different distance calculations
2. **Metadata Filtering**: Add document categories and filtered search
3. **Evaluation Metrics**: Build test questions and measure retrieval accuracy

## 🔧 Configuration & Experimentation

### Chunking Parameters
```python
# In rag_engine.py - experiment with these values
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Try: 500, 1500, 2000
    chunk_overlap=200   # Try: 100, 300, 400
)
```

### Retrieval Settings
```python
# Number of chunks to retrieve per query
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Try: 3, 8, 10
```

### Different Models
```python
# Experiment with different Ollama models
embeddings = OllamaEmbeddings(model="llama3")     # Try: mistral, codellama
llm = OllamaLLM(model="llama3")
```

## 📁 Project Architecture

```
unirag/
├── unirag.py              # 🎮 Interactive demo interface
├── unirag/
│   ├── rag_engine.py      # 🧠 Core RAG implementation
│   ├── docs/              # 📚 Your educational documents
│   ├── chroma_db/         # 🗄️ Vector database (auto-created)
│   └── vector_manifest.json # 📋 Change tracking system
└── requirements.txt       # 📦 Python dependencies
```

## 🎯 Real-World Applications

After mastering these concepts, you'll understand how to build:

- **Enterprise Knowledge Bases**: Company documentation systems
- **Customer Support Bots**: FAQ and help desk automation  
- **Research Assistants**: Academic paper analysis tools
- **Content Recommendation**: Personalized content discovery
- **Code Documentation**: Developer tool integration

## 🤝 Contributing to Education

This is an educational project! Contributions that improve learning are especially welcome:

- **Better Examples**: More diverse unicorn content for testing
- **Learning Exercises**: Additional hands-on activities
- **Documentation**: Clearer explanations of vector database concepts
- **Visualizations**: Charts or graphs showing embedding relationships
- **Performance Metrics**: Tools to measure and understand system behavior

## 📖 Further Reading

- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [RAG Architecture Deep Dive](https://python.langchain.com/docs/use_cases/question_answering/)
- [Embedding Models Comparison](https://huggingface.co/blog/mteb)
- [Chunking Strategies](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

## 📝 License

MIT License - Use this code for learning, teaching, and building your own educational projects!

---

**Ready to explore the magical world of vector databases?** 🦄✨

*This project demonstrates production-grade RAG concepts in an approachable, educational format. Perfect for students, educators, and anyone curious about how modern AI systems work under the hood.*
