import discord
from discord.ext import commands

class DiscordChatbot:
    """
    DiscordChatbot class for interfacing with the chat system via Discord.
    """

    def __init__(self, token, retrieval_chain, prefix='!'):
        """
        Initialize the DiscordChatbot.

        :param token: Discord bot token.
        :param retrieval_chain: The retrieval chain object for generating responses.
        :param prefix: Command prefix for the bot.
        """
        self.token = token
        self.retrieval_chain = retrieval_chain
        self.prefix = prefix
        self.bot = self.setup_bot()

    def setup_bot(self):
        """
        Set up the bot with necessary events and intents.

        :return: Configured Discord bot object.
        """
        intents = discord.Intents.all()
        intents.message_content = True
        bot = commands.Bot(command_prefix=self.prefix, intents=intents)

        @bot.event
        async def on_ready():
            print(f"We have logged in as {bot.user.name}")

        @bot.event
        async def on_message(message):
            if not message.author.bot and bot.user in message.mentions:
                await self.process_message(message)

        return bot

    async def process_message(self, message):
        """
        Process the incoming message, generate and send a response.

        :param message: The incoming Discord message.
        """
        # Send an initial message indicating that the bot is processing the request
        processing_message = await message.channel.send("I'm working on your answer, please wait a moment...")

        # Generate the response
        response = await self.generate_response(message.content)

        # Edit the initial message with the actual response
        await processing_message.edit(content=response)

    async def generate_response(self, message_content):
        """
        Generate a response using the retrieval chain.

        :param message_content: The content of the Discord message.
        :return: Response string.
        """
        input_data = {"input": message_content}
        response = await self.retrieval_chain.ainvoke(input_data)
        return response["answer"]

    def run(self):
        """
        Start the Discord bot.
        """
        self.bot.run(self.token)
