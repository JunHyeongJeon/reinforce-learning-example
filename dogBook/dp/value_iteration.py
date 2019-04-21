from value_iteration_environment import GraphicDisplay, Env

class ValueIteration:
    def __init__(self, env):
        # init env object
        self.env = env
        # init value function as 2D array
        self.value_table = [[0.00] * env.width for __ in range(env.height)]
        # discount factor
        self.discount_factor = 0.9

    # value iteration
    # find next value by using bellman optimality equation
    def value_iteration(self):
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]
        for state in self.env.get_all_states():
            if state == [2,2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            # for value function
            value_list = []

            # calculate every action
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((
                    reward + self.discount_factor * next_value
                ))
            # set maxium value as next value function
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
        self.value_table = next_value_table

    # return action from current value function
    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2,2] :
            return []
        
        # calculate every action's Q function
        # return max Q function's action
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)
        
        return action_list

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]])

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()

                

