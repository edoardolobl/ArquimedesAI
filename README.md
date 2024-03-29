
# ArquimedesAI: An Advanced LLM-Powered Chatbot Solution

<p align="center">
  <img src="assets/arquimedesai.jpg" alt="ArquimedesAI Logo" title="ArquimedesAI" width="35%" height="35%">
</p>

Welcome to ArquimedesAI V0.2! This version marks a significant upgrade, transforming ArquimedesAI into a state-of-the-art chatbot powered by a Large Language Model (LLM) and integrated with Discord. Leveraging advanced NLP technologies, ArquimedesAI offers rich, context-aware interactions and can be run locally on a user's PC.


> ## :warning: Disclaimer
>
> **Development Status:** Please note that ArquimedesAI is currently in active development. As such, the software is in a continuous state of evolution. Users should be aware that frequent updates and significant changes are expected as we work towards refining features and enhancing performance.
>
> **Use with Caution:** Given its developmental nature, we advise users to exercise caution when integrating ArquimedesAI into critical applications. While we strive for reliability, the current version may still contain bugs or inconsistencies that could affect its operation.
>
> **Feedback and Contributions:** We welcome feedback and contributions from the community. Your input is invaluable in helping us shape the future of ArquimedesAI. If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
>
> Thank you for your understanding and support as we continue to develop ArquimedesAI into a robust and reliable tool.


## Features
- **LangChain Framework Integration**: Incorporates the powerful LangChain framework for sophisticated language understanding and generation.
- **RAG-Ready**: Utilizes Retrieval Augmented Generation (RAG) for enhanced response accuracy and relevance.
- **Advanced Embeddings**: Employs a sentence-transformers model to convert text into embedding vectors, stored efficiently in a FAISS database.
- **Powered by Mistral 7b LLM**: Leverages the capabilities of Mistral 7b, a cutting-edge Large Language Model, for generating nuanced and contextually relevant responses.
- **Modular Design and Scalable Architecture**: Structured for easy maintenance and scalability, facilitating future enhancements.
- **Local Run Capability**: Designed to run locally on a user's PC, ensuring privacy and ease of access.


## Changelog

### [0.2.0.1] - 2024-01-14
#### Changed
- Code refactoring for better performance and maintainability.
- Added Contextual Compression Retriever: ColBERT (v2) for improved document retrieval.
- Integration of Sentence Transformers token splitter for enhanced language processing.
- Implementation of LangChain Expression Language (LCEL) for advanced query handling.

### [0.2.0] - 2024-01-07
#### Added
- Complete code overhaul for enhanced functionality and efficiency.
- Replaced Haystack with Langchain Framework for sophisticated language understanding and generation.
- Implemented RAG (Retrieval Augmented Generation) to chat with documents.
- Integration of Sentence Transformers for efficient text embedding.
- Added FAISS vector database for storing embedding vectors.
- Implementation of Mistral 7b Large Language Model for generating nuanced responses.

### [0.1.0] - 2023-08-11
#### Added
- Initial implementation of a Q&A system using Haystack framework.
- Q&A query-answer pair retrieval using BERT (NLP) and SQLite database.
- Basic Discord interface for user interactions.


## Getting Started

### Prerequisites

Ensure these libraries are installed:
- Python 3.x
- discord.py
- LangChain
- sentence-transformers
- FAISS
- Other dependencies (listed in `requirements.txt`)

### Setting Up

1. **Clone the repository**:
```bash
git clone https://github.com/edoardolobl/ArquimedesAI
```

2. **Navigate to the ArquimedesAI directory**:
```bash
cd ArquimedesAI
```

3. **Install the required dependencies:**:
```bash
pip install -r requirements.txt
```

4. **Replace the Discord token**:
Open the `Main.py` file and replace `YOUR_DISCORD_BOT_TOKEN` with your unique Discord bot token.

### Running ArquimedesAI

Invoke the main script:
```bash
python Main.py
```

## Usage

1. **Engage with ArquimedesAI on Discord**:
Simply mention the bot in your text and await its response.

## Behind the Scenes
- **LLM Integration**: The `RetrievalGeneration` class handles document retrieval and response generation using the Mistral 7b LLM.
- **Discord Integration**: The `DiscordChatbot` class manages interactions with Discord, receiving user messages and sending back LLM-generated responses.

## Contributing

Your contributions can shape ArquimedesAI's future! Dive into the contribution guidelines and join the mission.

## License

ArquimedesAI is protected under the Apache 2.0. Delve into the `LICENSE` file for intricate details.

## Acknowledgments
