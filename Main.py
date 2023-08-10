from DiscordBot import DiscordBot

if __name__ == "__main__":
    # Replace 'YOUR_DISCORD_BOT_TOKEN' with your actual bot token
    TOKEN = 'YOUR_DISCORD_BOT_TOKEN'
    
    # Initialize and run the Discord bot
    discord_bot = DiscordBot(token=TOKEN)
    discord_bot.run()
