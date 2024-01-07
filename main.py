from indexing import Indexing
from retrieval_generation import RetrievalGeneration
from discord_chatbot import DiscordChatbot  # Import the DiscordChatbot class

# Initialize the Indexing class
# It will handle loading, splitting, and storing of documents
indexing = Indexing("https://en.wikipedia.org/wiki/James_Webb_Space_Telescope", 
                    "distiluse-base-multilingual-cased-v1", 
                    "./embeddings_cache/")

# Load documents from the specified URL
docs = indexing.load_documents()

# Split the loaded documents into smaller, manageable segments
documents = indexing.split_documents(docs)

# Store and index these documents in a FAISS vector database
db = indexing.store_documents(documents)

# Initialize the RetrievalGeneration class
# It will handle retrieval of document segments and generation of responses
retrieval_gen = RetrievalGeneration("mistral:7b-instruct-v0.2-q5_K_M", 
                                    """
                                    Answer the following question based only on the provided context:

                                    <context>
                                    {context}
                                    </context>

                                    Question: {input}
                                    """)

# Create a document chain for combining document information
document_chain = retrieval_gen.create_document_chain()

# Set up a retrieval chain using the FAISS database and document chain
retriever = db.as_retriever(search_type="mmr")
retrieval_chain = retrieval_gen.create_retrieval_chain(retriever, document_chain)

# Initialize the DiscordChatbot class with the retrieval chain
discord_bot = DiscordChatbot("YOUR_DISCORD_BOT_TOKEN", retrieval_chain)

# Start the Discord bot
discord_bot.run()
