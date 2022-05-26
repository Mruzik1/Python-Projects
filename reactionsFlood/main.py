import reactions


bot = reactions.EmojiFlooder() 

 
def main():
    token = input('Token:\n')
    try:
        bot.run(token)
    except:
        print('Possibly wrong token!')


if __name__ == '__main__':
    main()