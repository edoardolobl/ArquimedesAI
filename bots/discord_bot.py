"""
Discord bot interface for ArquimedesAI.

Provides async Discord integration with RAG chain for
answering questions based on indexed documents.
"""

import logging

import discord
from discord.ext import commands

logger = logging.getLogger(__name__)


class DiscordChatbot:
    """
    Discord bot for ArquimedesAI RAG system.
    
    Listens for mentions and responds with answers from the RAG chain.
    Uses async pattern with edit-message to avoid timeouts.
    
    Attributes:
        token: Discord bot authentication token
        rag_chain: RAG chain instance for question answering
        bot: Discord bot instance
    """
    
    def __init__(self, token: str, rag_chain):
        """
        Initialize Discord chatbot.
        
        Args:
            token: Discord bot token
            rag_chain: Initialized RAG chain instance
        """
        self.token = token
        self.rag_chain = rag_chain
        self.bot = self._setup_bot()
        
        logger.info("Discord bot initialized")
    
    def _setup_bot(self) -> commands.Bot:
        """
        Configure Discord bot with intents and event handlers.
        
        Returns:
            Configured Discord bot instance
        """
        # Configure intents (need message_content for reading messages)
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        
        bot = commands.Bot(command_prefix="!", intents=intents)
        
        @bot.event
        async def on_ready():
            """Called when bot successfully connects."""
            logger.info(f"Bot logged in as {bot.user.name} (ID: {bot.user.id})")
            logger.info(f"Connected to {len(bot.guilds)} guild(s)")
            print(f"\n✓ Bot is ready: {bot.user.name}")
            print("Listening for mentions...\n")
        
        @bot.event
        async def on_message(message: discord.Message):
            """
            Handle incoming messages.
            
            Responds to messages that mention the bot.
            """
            # Ignore bot's own messages
            if message.author.bot:
                return
            
            # Only respond to mentions
            if bot.user not in message.mentions:
                return
            
            await self._process_message(message)
        
        return bot
    
    async def _process_message(self, message: discord.Message):
        """
        Process a message and generate response.
        
        Uses edit pattern: send "processing" message, then edit with answer.
        
        Args:
            message: Discord message to process
        """
        try:
            # Send initial "processing" message
            processing_msg = await message.channel.send(
                "🤔 Processing your question..."
            )
            
            # Extract query (remove bot mention)
            query = message.content
            for mention in message.mentions:
                query = query.replace(f"<@{mention.id}>", "").strip()
            
            if not query:
                await processing_msg.edit(
                    content="Please ask a question after mentioning me!"
                )
                return
            
            logger.info(f"Query from {message.author}: {query}")
            
            # Generate response using RAG chain
            result = await self.rag_chain.ainvoke(query)
            answer = result.get("answer", "I couldn't generate an answer.")
            
            # Format response with context info
            num_sources = len(result.get("context", []))
            formatted_response = f"{answer}\n\n*({num_sources} source(s) used)*"
            
            # Edit message with actual answer
            await processing_msg.edit(content=formatted_response)
            
            logger.info("✓ Response sent")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            
            # Try to send error message
            try:
                await processing_msg.edit(
                    content="⚠️ Sorry, I encountered an error processing your question."
                )
            except:
                pass
    
    def run(self):
        """
        Start the Discord bot.
        
        Blocks until bot is stopped (Ctrl+C).
        """
        logger.info("Starting Discord bot...")
        try:
            self.bot.run(self.token)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            raise
