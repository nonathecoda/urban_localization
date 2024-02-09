from icecream import ic

class Localizer():
    def __init__(self):
        self.__location = "Japan"

    def run(self):
        ic(self.__location)


if __name__ == "__main__":
    #initialise stuff
    loc = Localizer()
    loc.run()