from Agent import Agent


class Human(Agent):
    def act(self, state, showNQ=False):
        return input()