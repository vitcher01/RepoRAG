import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# Supported file extensions to index
SUPPORTED_EXTENSIONS = {
    '.java': 'java', 
    '.yaml': 'yaml', 
    '.yml': 'yaml', 
    '.json': 'json', 
    '.sql': 'sql', 
    '.md': 'markdown',
    '.properties': 'properties'
}

# Embedding Model Config
# Using Jina V2 Base Code ( supports 8192 tokens)
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
# Fallback model if Jina fails
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"

def extract_java_metadata(content):
    """
    Extracts Java-specific metadata to give the LLM context.
    Finds the package name and the primary class/interface name.
    """
    metadata = {"package": "unknown", "class": "unknown"}
    
    pkg_match = re.search(r'^\s*package\s+([\w\.]+);', content, re.MULTILINE)
    if pkg_match:
        metadata["package"] = pkg_match.group(1)
        
    # Extract Class/Interface Name
    class_match = re.search(r'public\s+(class|interface|enum|@interface)\s+(\w+)', content)
    if class_match:
        metadata["class"] = class_match.group(2)
        
    return metadata

def main():
    print("=" * 60)
    print("PHASE 1: SPRING BOOT RAG INDEXER")
    print("=" * 60)

    # 1. Load Repo Paths
    local_repo_paths_str = os.getenv('LOCAL_REPO_PATHS', '')
    if not local_repo_paths_str:
        print("WARNING: LOCAL_REPO_PATHS not set in .env")
        print("Using current directory as default.")
        repo_paths = ["."] 
    else:
        repo_paths = [p.strip() for p in local_repo_paths_str.split(';') if p.strip()]

    print(f"Target Repositories: {len(repo_paths)}")

    # 2. Load Documents
    all_documents = []
    
    for repo_path in repo_paths:
        path_obj = Path(repo_path)
        if not path_obj.exists():
            print(f"Skipping {repo_path} (Not Found)")
            continue
            
        print(f"Scanning {repo_path}...")
        file_count = 0
        
        # Walk through all files recursively
        for filepath in path_obj.rglob('*'):
            if filepath.is_file() and filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    # Read file content
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    if not content.strip():
                        continue

                    # Base Metadata
                    rel_path = str(filepath.relative_to(path_obj))
                    metadata = {
                        'source': rel_path,
                        'repo': str(repo_path),
                        'filename': filepath.name,
                        'extension': filepath.suffix.lower()
                    }

                    # Java-Specific Processing
                    if filepath.suffix.lower() == '.java':
                        java_meta = extract_java_metadata(content)
                        metadata.update(java_meta)
                        
                        # Prepend context header for the LLM
                        # This ensures even small chunks know which file they belong to
                        context_header = (
                            f"// File: {rel_path}\n"
                            f"// Package: {metadata['package']}\n"
                            f"// Class: {metadata['class']}\n"
                        )
                        final_content = context_header + content
                    else:
                        # Generic header for other files
                        final_content = f"## File: {rel_path}\n" + content

                    all_documents.append(Document(page_content=final_content, metadata=metadata))
                    file_count += 1
                    
                except Exception as e:
                    print(f"  Error reading {filepath.name}: {e}")

        print(f"  -> Loaded {file_count} files")

    if not all_documents:
        print("No documents found. Exiting.")
        sys.exit(1)

    # 3. Split Documents 
    print("\nStep 3: Splitting Code...")
    
    spring_separators = [
        "\nclass ", 
        "\ninterface ", 
        "\nenum ", 
        "\n@",           # Split before annotations
        "\n\t@",         # Indented annotations
        "\n    @",       
        "\npublic ", 
        "\nprotected ", 
        "\nprivate ",
        "\n\n",
        "\n",
        ";"
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,       # Large chunks for full methods
        chunk_overlap=200,     # Context overlap
        separators=spring_separators,
        keep_separator=True,   # Keeps the separator (e.g. '@') in the chunk
        is_separator_regex=False
    )

    split_docs = text_splitter.split_documents(all_documents)
    print(f"Created {len(split_docs)} chunks from {len(all_documents)} files.")

    # 4. Generate Embeddings
    print(f"\nStep 4: generating Embeddings ({EMBEDDING_MODEL_NAME})...")
    print("This may take a moment...")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"Warning: Failed to load {EMBEDDING_MODEL_NAME}. Error: {e}")
        print(f"Falling back to {FALLBACK_MODEL_NAME}...")
        embeddings = HuggingFaceEmbeddings(model_name=FALLBACK_MODEL_NAME)

    # 5. Create Vector Store (FAISS)
    print("\nStep 5: Building Vector Index...")
    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # Save to disk
        output_dir = "faiss_index"
        vectorstore.save_local(output_dir)
        print(f"Success! Index saved to './{output_dir}'")
        
    except Exception as e:
        print(f"Critical Error building FAISS index: {e}")
        sys.exit(1)

    print("-" * 60)
    print("READY. You can now run your query script.")
    print("-" * 60)

if __name__ == "__main__":
    main()
