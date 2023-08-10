import discord
from discord.ext import commands
from Chatbot import Chatbot

class DiscordBot:
    def __init__(self, token, prefix='!'):
        self.token = token
        self.prefix = prefix
        self.bot = self.setup_bot()
        self.chatbot = Chatbot()

    def setup_bot(self):
        intents = discord.Intents.all()
        intents.message_content = True
        bot = commands.Bot(command_prefix=self.prefix, intents=intents)

        @bot.event
        async def on_ready():
            print(f"We have logged in as {bot.user.name}")

        @bot.event
        async def on_message(message):
            if not message.author.bot and bot.user in message.mentions:
                response = self.chatbot.get_response(message.content)
                await message.channel.send(response + "\n\n🚀 Gostou da resposta? Deixe seu feedback com um 👍 ou 👎! Ajude-nos a melhorar! 🚀")
        
        @bot.event
        async def on_reaction_add(reaction, user):
            # Exclude reactions made by the bot itself
            if user == bot.user:
                return
            
            # Define feedback emojis
            positive_feedback_emoji = ["👍"]
            negative_feedback_emoji = ["👎"]
            
            if str(reaction.emoji) in positive_feedback_emoji:
                print(f"Received positive feedback from {user.name} using {reaction.emoji}")

            elif str(reaction.emoji) in negative_feedback_emoji:
                print(f"Received negative feedback from {user.name} using {reaction.emoji}")


        return bot

    def run(self):
        self.bot.run(self.token)
