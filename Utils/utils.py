



class Logger:
    @staticmethod
    def log(message):
        f=open('log.txt','a+')
        f.write(message)
        f.close()
        print(message)