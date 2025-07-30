# unirag.py
"""
🦄 UNIRAG - INTERACTIVE RAG DEMONSTRATION
========================================

This script provides a user-friendly interface to interact with the RAG system.
It demonstrates key concepts in conversational AI and user experience design.

Key Learning Concepts:
1. Conversational interfaces and chat loops
2. Context management across multiple turns
3. User experience in AI applications
4. Threading for responsive interfaces
5. Error handling in production systems

🎓 EDUCATIONAL GOALS:
- See how RAG systems work in real-time
- Understand conversation flow and context
- Learn UI/UX patterns for AI applications
- Practice Python threading and interface design
"""

import threading
import time
from unirag.rag_engine import get_qa_chain

# 🌟 GLOBAL STATE MANAGEMENT
# =========================
# In larger applications, you'd use more sophisticated state management
# But this simple global variable demonstrates the concept
spinner_running = False

def loading_spinner():
    """
    ⏳ VISUAL FEEDBACK SYSTEM - USER EXPERIENCE DESIGN
    ================================================
    
    This function provides visual feedback while the AI is thinking.
    It's a simple but important UX pattern - users need to know something is happening!
    
    Learning Points:
    - User experience matters in AI applications
    - Threading allows UI to remain responsive
    - Simple animations can greatly improve perceived performance
    - Global state coordination between threads
    
    🎨 UX PRINCIPLES DEMONSTRATED:
    - Progressive disclosure (show process step-by-step)
    - Feedback (visual indication of system state)
    - Perceived performance (makes waiting feel shorter)
    
    🔧 CUSTOMIZATION IDEAS:
    - Add progress percentages
    - Show which step is currently executing
    - Use colored output for different stages
    - Add estimated time remaining
    """
    # 🎭 Animation characters - simple but effective visual feedback
    spinner_chars = ['|', '/', '-', '\\']
    idx = 0
    
    # 🔄 Animation loop - runs until spinner_running becomes False
    while spinner_running:
        print(f"\r🧠 Thinking... {spinner_chars[idx % len(spinner_chars)]}", end='', flush=True)
        time.sleep(0.1)  # 🔧 ADJUST: Faster/slower animation speed
        idx += 1

def main():
    """
    🎮 MAIN CONVERSATION LOOP - THE HEART OF THE INTERFACE
    ====================================================
    
    This function demonstrates how to build conversational AI interfaces
    that maintain context and provide good user experience.
    
    Key Concepts Demonstrated:
    1. Chat loop pattern (infinite loop with exit conditions)
    2. Context accumulation (building conversation history)
    3. Error handling (graceful failure recovery)
    4. Threading coordination (responsive UI during processing)
    5. State management (tracking conversation across turns)
    
    🧠 CONVERSATION FLOW:
    1. Welcome user and explain interface
    2. Initialize RAG system (load documents, build vectors)
    3. Enter chat loop:
       - Get user input
       - Check for exit conditions
       - Build context from history
       - Process query with visual feedback
       - Display result and update history
    4. Graceful exit
    """
    
    # 🎭 WELCOME MESSAGE - SETTING EXPECTATIONS
    # ========================================
    # Good AI interfaces explain what they can do and how to use them
    print("🦄 Welcome to UniRAG chat! Ask me anything about unicorns.")
    print("💬 Type 'exit' to quit.\n")
    
    # 📚 CONVERSATION HISTORY MANAGEMENT
    # =================================
    # This list stores the entire conversation for context building
    # In production systems, you'd add limits and possibly persistence
    history = []  # Format: [(question1, answer1), (question2, answer2), ...]
    
    # 🏗️ SYSTEM INITIALIZATION
    # ========================
    # Get the QA chain (this triggers document loading and vector building)
    # We do this once at startup for efficiency
    print("🔧 Initializing RAG system...")
    try:
        chain = get_qa_chain()
        print("✅ System ready!\n")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("🔍 Check that you have documents in the unirag/docs/ folder")
        return
    
    # 🔄 MAIN CONVERSATION LOOP
    # ========================
    # This is the core pattern for conversational AI applications
    while True:
        # 📝 GET USER INPUT
        # ================
        # Simple input with visual prompt - strip() removes extra whitespace
        user_input = input("🗨️  You: ").strip()
        
        # 🚪 EXIT CONDITION HANDLING
        # =========================
        # Multiple exit phrases for better UX
        if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
            print("👋 Bye!")
            break
        
        # ⚠️ EMPTY INPUT HANDLING
        # ======================
        # Skip empty inputs gracefully
        if not user_input:
            continue
            
        # 🧠 CONTEXT BUILDING - THE KEY TO CONVERSATIONAL AI
        # =================================================
        # This is where we build context from conversation history
        # The format mimics natural conversation flow
        
        context_prompt = ""
        
        # 📜 Add conversation history for context
        # Each previous Q&A pair provides context for the current question
        for i, (q, a) in enumerate(history):
            context_prompt += f"Q{i+1}: {q}\nA{i+1}: {a}\n"
        
        # ➕ Add current question
        context_prompt += f"Q{len(history)+1}: {user_input}\nA{len(history)+1}:"
        
        # 🎯 VISUAL FEEDBACK COORDINATION
        # ==============================
        # Start the loading spinner in a separate thread
        # This keeps the interface responsive while AI processes
        global spinner_running
        spinner_running = True
        spinner_thread = threading.Thread(target=loading_spinner)
        spinner_thread.daemon = True  # Dies when main thread dies
        spinner_thread.start()
        
        # 🤖 AI PROCESSING WITH ERROR HANDLING
        # ===================================
        try:
            # This is where the magic happens - RAG system processes the query
            result = chain.invoke({"query": context_prompt})
            answer = result.get("result", "").strip()
            
            # 🧹 RESPONSE CLEANING
            # ===================
            # Sometimes AI responses have extra formatting - clean it up
            if not answer:
                answer = "I couldn't generate a response. Try rephrasing your question."
                
        except Exception as e:
            # 🛡️ GRACEFUL ERROR HANDLING
            # =========================
            # Never let errors crash the conversation
            print(f"\n⚠️ Error details: {e}")  # For debugging/learning
            answer = "I encountered an error processing your question. Please try again."
            
        finally:
            # 🛑 ALWAYS STOP THE SPINNER
            # =========================
            # This runs whether we succeed or fail
            spinner_running = False
            spinner_thread.join()  # Wait for spinner thread to finish
            print("\r" + " " * 50 + "\r", end="")  # Clear spinner line
        
        # 🎨 RESPONSE PRESENTATION
        # =======================
        # Visual formatting makes responses easier to read
        print("\n" + "="*50)
        print("🦄 UNIRAG RESPONSE")
        print("="*50)
        print(f"{answer}\n")
        
        # 💾 CONVERSATION HISTORY UPDATE
        # =============================
        # Save this exchange for future context
        history.append((user_input, answer))
        
        # 📊 OPTIONAL: CONVERSATION STATISTICS
        # ==================================
        # Uncomment to show conversation stats (useful for learning/debugging)
        # print(f"📈 Conversation length: {len(history)} exchanges")
        
        # 🔧 MEMORY MANAGEMENT (ADVANCED)
        # ==============================
        # In production, you'd limit history length to prevent context overflow
        # Uncomment to limit to last 10 exchanges:
        # if len(history) > 10:
        #     history = history[-10:]

# 🎯 PYTHON ENTRY POINT PATTERN
# =============================
# This ensures the script only runs when executed directly, not when imported
if __name__ == "__main__":
    # 🛡️ TOP-LEVEL ERROR HANDLING
    # ==========================
    # Catch any startup errors gracefully
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("🔍 This might help with debugging:")
        import traceback
        traceback.print_exc()

# 🎓 LEARNING EXERCISES FOR STUDENTS:
# ==================================
# 
# 1. BEGINNER: Add a command to show conversation history (type "history")
# 
# 2. INTERMEDIATE: Implement conversation saving/loading to files
# 
# 3. ADVANCED: Add commands like "clear" to reset history, "help" for instructions
# 
# 4. EXPERT: Implement conversation branching (save/restore different conversation paths)
# 
# 5. UX RESEARCH: Try different spinner animations, response formatting, or color coding
# 
# 6. PRODUCTION: Add conversation limits, user authentication, or web interface
# 
# 🔧 CUSTOMIZATION IDEAS:
# ======================
# 
# - Change the theme from unicorns to your domain (cooking, history, science)
# - Add conversation statistics and analytics
# - Implement different response modes (brief, detailed, creative)
# - Add voice input/output capabilities
# - Create a web interface using Flask or Streamlit
# - Add conversation export functionality (PDF, text, JSON)
# - Implement user profiles and personalization