import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# --- 1. Load Environment Variables ---
# Make sure to create a .env file with your GITHUB_ACCESS_TOKEN
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

# --- 2. Define Constants ---
# The GitHub repository to analyze
# The vector database path
# The LLM and embedding models to use
# Make sure to have the models available in your Ollama server
# You can change these to your preferred models
REPO_URL = "https://github.com/woliveiras/reader-agent"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"

def main():
    """
    Main function to set up and run the GitHub QA agent.
    """
    print("Starting the AI agent for GitHub repository analysis...")

    # --- 3. Initialize Models ---
    llm = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")

    # --- 4. Load or Create the Vector Database ---
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print(f"Loading existing vector database from {CHROMA_PERSIST_DIRECTORY}...")
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        print(f"Creating new vector database for {REPO_URL}...")
        
        # Load the repository files
        print("Loading files from the GitHub repository...")
        loader = GithubFileLoader(
            repo="woliveiras/reader-agent",
            branch="main",
            access_token=GITHUB_TOKEN,
            github_api_url="https://api.github.com",
            file_filter=lambda file_path: file_path.endswith((".py", ".md")),
        )
        docs = loader.load()
        
        # Filter for relevant source code files
        print(f"Loaded {len(docs)} documents from the repository.")
        source_code_docs = docs
        print(f"Filtered to {len(source_code_docs)} source code documents.")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(source_code_docs)
        print(f"Documents split into {len(splits)} chunks.")

        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        print("Vector database created and persisted.")

    # --- 5. Create the RAG Chain ---
    retriever = vector_store.as_retriever()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer user questions only based on the context below, which is extracted from the repository. "
            "Never use prior model knowledge or external information. If the answer is not in the context,"
            "explicitly state that it is not possible to answer based only on the repository.\n\n{context}"
        )),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # --- 6. Start Interactive Chat Loop ---
    print("\nAgent is ready. You can now ask questions.")
    while True:
        try:
            question = input("Type your question (or 'exit' to quit): ")
            if question.strip().lower() == 'exit':
                print("Exiting agent...")
                break
            
            print("Querying the agent...")
            result = retrieval_chain.invoke({"input": question})
            
            print("\n--- Agent Response ---")
            print(result["answer"])
            print("--------------------------\n")

        except KeyboardInterrupt:
            print("\nExiting agent...")
            break

if __name__ == "__main__":
    main()