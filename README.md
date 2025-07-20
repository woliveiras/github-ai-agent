# github-ai-agent

This repository contains an example of an AI agent for automation and integration with GitHub, using Python, LangChain, Local LLM with Ollama, nomic-embed-text for embedding, and ChromaDB for vector database. The agent can be expanded for different automated workflows.

## Requirements

- Python 3.8+
- Ollama server running with the required models available
- A GitHub personal access token with read access to the repository
- Dependencies listed in `requirements.txt`

## Setup

1. **Clone this repository**

```sh
git clone git@github.com:woliveiras/github-ai-agent.git
cd github-ai-agent
```

2. **Create and activate a virtual environment**

```sh
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux. On Windows, use `venv\Scripts\activate`
```

3. **Install dependencies**

```sh
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root of the project directory and add the following environment variables:

```env
GITHUB_ACCESS_TOKEN=your_github_access_token
```

Replace `your_github_access_token` with your actual GitHub personal access token.

## Running the Agent

To run the agent and test its functionality, execute:

```sh
python agent.py
```

## File Structure

- `agent.py`: Main script that initializes the agent and manages operations.
- `requirements.txt`: List of required Python dependencies.
- `chroma_db/`: Directory for the local database used by the agent.

## References

[How to Chat with Your GitHub Repository: A Guide to Local RAG with Ollama and LangChain](https://woliveiras.github.io/posts/how-to-chat-with-github-repository-a-guide-to-local-rag-with-ollama-and-langchain/)