from text_visual import VideoSymbols
import discord


class MegaBot(discord.Client):
    def __init__(self, video_path, channel_id, *args, **kwargs):
        self.animation = VideoSymbols(video_path, (40, 20))
        self.channel_id = channel_id
        self.__token = 'NzMwNzI5MzE2MjY2MDE2NzY4.YlmDVQ.EETz6Uij4a1K6yMloSdlCKxsgC8'
        super().__init__(*args, **kwargs)
    

    async def on_ready(self):
        print(f'{self.user} has joined!')

        channel = self.get_channel(self.channel_id)
        msg = await channel.send('Starting...')
        await self.animation.play(msg)
         
        
    def run_bot(self):
        try:
            self.run(self.__token)
        except:
            print('Possibly wrong token!')


if __name__ == '__main__':
    bot = MegaBot('content/vid.mp4', 785210033931026452)
    bot.run_bot()