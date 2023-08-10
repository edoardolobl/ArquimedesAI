
# ArquimedesAI: An Advanced Chatbot Solution

Welcome to ArquimedesAI V0.1 (alpha)! This project embodies a comprehensive chatbot solution, integrated with Discord, leveraging the strengths of BERT embeddings tailored for Portuguese. Integrated seamlessly with Discord, the bot delivers responses founded on a meticulously curated training dataset.

## Features

1. **BERT-based Semantic Understanding**: Employs the power of BERT embeddings, specifically fine-tuned for Portuguese, ensuring nuanced understanding and relevant responses.
2. **Discord Integration**: Designed to function as a Discord bot, it actively listens and responds to users when mentioned.
3. **SQLite Database Connectivity**: Utilizes an SQLite database to systematically manage and fetch training data.
4. **Feedback System (Under Development)**: Future versions aim to incorporate a feedback mechanism, enabling users to communicate their satisfaction level with the bot's responses.

## Getting Started

### Prerequisites

Ensure these libraries are installed:
- `discord.py`
- `transformers`
- `torch`
- `faiss`
- `numpy`
- `sqlite3`

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

1. **BERT's Semantic Genius**: The `Chatbot` class in `Chatbot.py` initializes with a BERT model and tokenizer for Portuguese. By tapping into an SQLite3 database, it loads training data and pre-computes BERT embeddings. Utilizing FAISS, it indexes these embeddings to ascertain the most analogous questions to any query swiftly.

2. **SQLite Database Mastery**: The `DBHandler` class in `DBHandler.py` meticulously orchestrates interactions with the SQLite3 database, ensuring swift fetching of responses for queries.

3. **Discord Integration Prowess**: The `DiscordBot` class in `DiscordBot.py` forms the bridge between the chatbot and Discord. Alert and active, it responds whenever the bot is mentioned, delivering the apt response from ArquimedesAI.

## Contributing

Your contributions can shape ArquimedesAI's future! Dive into the contribution guidelines and join the mission.

## License

ArquimedesAI is protected under the MIT License. Delve into the `LICENSE` file for intricate details.

## Acknowledgments

- A huge shoutout to OpenAI's GPT-4 for guidance throughout the journey.
- Immense gratitude towards the creators of the BERT model, especially the model for Brazilian Portuguese. Please refer to the research: 
  ```
  @inproceedings{souza2020bertimbau,
    author    = {Fábio Souza and Rodrigo Nogueira and Roberto Lotufo},
    title     = {BERTimbau: pretrained BERT models for Brazilian Portuguese},
    booktitle = {9th Brazilian Conference on Intelligent Systems, BRACIS, Rio Grande do Sul, Brazil, October 20-23},
    year      = {2020}
  }
  ```
- Heartfelt thanks to the team behind FAISS for their groundbreaking work:
  ```
  @article{johnson2019billion,
    title={Billion-scale similarity search with GPUs},
    author={Johnson, Jeff and Douze, Matthijs and Jégou, Hervé},
    journal={IEEE Transactions on Big Data},
    volume={7},
    number={3},
    pages={535--547},
    year={2019},
    publisher={IEEE}
  }
  ```
