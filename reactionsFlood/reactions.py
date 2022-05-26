import time
import os
import discord
import threading


class EmojiFlooder(discord.Client):
    def __init__(self, *args, **kwargs):
        self.channels = []
        self.reaction = '🤓'
        self.delay = 2
        self.people = []
        self.__pause = False
        super().__init__(*args, **kwargs)


    def print_preset(self):
        try:
            print(f'Channels: {[self.get_channel(e).name for e in self.channels]}')
        except: print('You have wrong channel ID somewhere!')

        try:
            print(f'People: {[self.get_user(e).name for e in self.people]}')
        except: print('You have wrong user ID somewhere!')

        print(f'Delay: {self.delay}s')
        print(f'Reaction: {self.reaction}')

    
    def delay_counter(self):
        self.__pause = True
        time.sleep(self.delay)
        self.__pause = False

    
    async def on_message(self, msg):
        if not self.__pause:
            threading.Thread(target = self.delay_counter).start()

            if msg.author.id in self.people and msg.channel.id in self.channels:
                try:
                    await msg.add_reaction(self.reaction)
                    await msg.remove_reaction(self.reaction, self.user)
                except: print("\nCan't add/remove reaction!\n>>> ", end='')


    async def on_ready(self):
        print(f'\n{self.user} has joined!\n\nDefault preset:')
        self.print_preset()

        threading.Thread(target = self.execute_command).start()

    
    def execute_command(self):
        while not self.is_closed():
            cmd = input('>>> ').split()
            if not len(cmd): continue

            # stop bot
            if cmd[0].lower() == 'stop':
                print('Bye!')
                os._exit(0)
            
            # change reaction
            elif cmd[0].lower() == 'reaction':
                if len(cmd) < 2:
                    print('Wrong command usage.\nREACTION [EMOJI/EMOJI_ID]')
                else:
                    self.reaction = cmd[1]
                    print(f'New emoji to react: {cmd[1]}')
            
            # check bot's setup
            elif cmd[0].lower() == 'bot':
                self.print_preset()

            # change reactions delay
            elif cmd[0].lower() == 'delay':
                if len(cmd) < 2:
                    print('Wrong command usage.\nDELAY [SECONDS, COULD BE 0]')
                else:
                    try:
                        if int(cmd[1]) >= 0:
                            print(f'New delay: {cmd[1]}s')
                            self.delay = int(cmd[1])
                            continue
                    except: pass
                    print('Wrong time format, use positive integers!')

            # adding ids
            elif cmd[0].lower() == 'add':
                if len(cmd) < 3:
                    print('Wrong command usage.\nADD [CHAT/USER] [ID]')
                elif cmd[1].lower() == 'chat':
                    try:
                        chat = self.get_channel(int(cmd[2]))
                        if (chat != None):
                            print(f'New chat-room: {chat}')
                            self.channels.append(int(cmd[2]))
                            continue
                    except: pass
                    print('Wrong chat-room ID!')
                elif cmd[1].lower() == 'user':
                    try:
                        user = self.get_user(int(cmd[2]))
                        if (user != None):
                            print(f'New user: {user}')
                            self.people.append(int(cmd[2]))
                            continue
                    except: pass
                    print('Wrong user ID!')
                else: print('Use [CHAT/USER] as the second argument!')

            # removing ids
            elif cmd[0].lower() == 'remove':
                try:
                    if len(cmd) < 3:
                        print('Wrong command usage.\nREMOVE [CHAT/USER] [ID]')
                    elif cmd[1].lower() == 'chat':
                        print(f'{self.get_channel(int(cmd[2]))} has been removed')
                        self.channels.remove(int(cmd[2]))
                    elif cmd[1].lower() == 'user':
                        print(f'{self.get_user(int(cmd[2]))} has been removed')
                        self.people.remove(int(cmd[2]))
                    else:
                        print('Use [CHAT/USER] as the second argument!')
                except: print('Invalid ID!')

            # print users and channels with their ids
            elif cmd[0].lower() == 'ids':
                print(f'Channels ids: {self.channels}')
                print(f'Users ids: {self.people}')

            # all commands list
            elif cmd[0].lower() == 'help':
                print('Commands list:')
                print('STOP  -  exit the program.')
                print('HELP  -  check all commands.')
                print("BOT  -  check bot's setup.")
                print("IDS  -  check chat-rooms and users IDs.")
                print('REACTION [EMOJI/EMOJI_ID]  -  change the reaction emoji.')
                print('DELAY [SECONDS, COULD BE 0]  -  change the delay.')
                print('ADD [CHAT/USER] [ID]  -  add chat-room/user ID to the list.')
                print('REMOVE [CHAT/USER] [ID]  -  remove chat-room/user ID from the list.')
            
            # wrong command
            else: print('Wrong command! Try using [HELP]')