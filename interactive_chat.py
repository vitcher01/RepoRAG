import os
import sys
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#same model used that was used during indexing
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code" 
VECTOR_DB_PATH = "faiss_index"

print("="*60)
print(" PHASE 2: SPRING BOOT RAG ASSISTANT")
print("="*60 + "\n")

print("Initializing...")

# 1. Load Embeddings
try:
    print(f"Loading model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    print(f"Error loading embedding model: {e}")
    sys.exit(1)

# 2. Load Vector Store
try:
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"ERROR: Index folder '{VECTOR_DB_PATH}' not found!")
        print("Run 'python indexer.py' first.")
        sys.exit(1)
        
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    print("Vector database loaded successfully.")
    
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    sys.exit(1)

# 3. Setup Claude
try:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not found in .env")
        sys.exit(1)
    
    llm = ChatAnthropic(
        model="claude-opus-4-5-20251101", #model name taken from claude dashboard
        temperature=0,
        api_key=anthropic_key
    )
    print("Claude initialized.")
except Exception as e:
    print(f"Error initializing Claude: {e}")
    sys.exit(1)

# 4. Define Prompt
system_prompt = """You are a Senior Java Spring Boot Architect. 
You are answering questions about a specific local codebase based on the provided context chunks.

Guidelines:
- The context includes file paths, package names, and class definitions. Use them.
- If the answer involves a Controller, mention the specific API endpoints (@PostMapping, etc).
- If the answer involves a Service, explain the business logic.
- Always cite the file name when explaining logic (e.g. "In CreditCardController.java...").
- If the context does not contain the answer, say "I don't see that in the indexed code."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

def format_docs(docs):
    """Format retrieved docs for the LLM"""
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content
        formatted_chunks.append(f"--- SOURCE: {source} ---\n{content}\n")
    return "\n".join(formatted_chunks)

# 5. Build Chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nReady! Type 'exit' to quit.\n")
print("-" * 60)

# 6. Interactive Loop
while True:
    try:
        query = input("You: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\nSearching codebase & Thinking...")
        
        # Get source documents first
        source_docs = retriever.invoke(query)
        
        # Get answer
        answer = chain.invoke(query)
        
        print("\nAssistant:")
        print(answer)
        
        # Show Sources
        print(f"\n📚 Reference Sources ({len(source_docs)} chunks):")
        unique_files = set()
        for doc in source_docs:
            f_name = doc.metadata.get('source', 'unknown')
            if f_name not in unique_files:
                print(f"  • {f_name}")
                unique_files.add(f_name)
                
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error processing query: {e}")
    
    print("-" * 60)
