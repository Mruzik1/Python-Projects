import cv2 as cv
import discord, time


class VideoSymbols(discord.Client):
    def __init__(self, video_path, size, channel_id, *args, **kwargs):
        self.__symbols_list = '█▓▒░'
        self.video = cv.VideoCapture(video_path)
        self.size = size
        self.channel_id = channel_id
        self.__token = ''
        super().__init__(*args, **kwargs)


    def _text_image(self, img):
        result = ''

        for y in img:
            for x in y:
                idx = round((0.2126*x[0]+0.7152*x[1]+0.0722*x[2])/255*(len(self.__symbols_list)-1))
                result += self.__symbols_list[idx]
            result += '\n'

        return result


    async def play(self):
        channel = self.get_channel(self.channel_id)
        msg = await channel.send('Starting...')

        while True:
            frames_grabbed, tmp_frame = self.video.read()

            if not frames_grabbed:
                break

            frame = cv.resize(tmp_frame, self.size, interpolation = cv.INTER_AREA)
            await channel.send(self._text_image(frame))
            print(self._text_image(frame))
            time.sleep(1)

    
    async def on_ready(self):
        print(f'{self.user} has joined!')

        await self.play()
         
        
    def run_bot(self):
        try:
            self.run(self.__token)
        except:
            print('Possibly wrong token!')


if __name__ == '__main__':
    try:
        bot = VideoSymbols(0, (50, 25), 785210033931026452)
        bot.run_bot()
    except KeyboardInterrupt:
        bot.video.release()