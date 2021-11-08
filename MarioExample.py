from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros, data
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 0 = no action, 1 = right, 2 = right + A, 3 = right + B, 4 = right + A + B, 5 = A, 6 = left

# Input into play game the number of the run and match it to an array in data.py for mutation
class MarioExample():

    def playGame(self, x):
        lastX = 0
        counter = 0
        counter2 = 0
        i = 0

        env.reset()

        while i < 3000:
            # Comment out to remove human-viewable game screen for faster performance
            env.render()


            # Attempt to split Actions Per Ssecond by 4 from 60 -> 15
            for r in range(4):
                state, reward, done, info = env.step(int(x[i]))

                if info['life'] == 1 or counter == 100 or i == 3000 or done:
                
                    env.reset()

                    if data.furthest < i:
                        data.furthest = i
                # Want to maximize distance, maximize time
                    return 0 - info['x_pos'], 0 - info['time']
            
            i = i + 1
            #print(info['life'])

            # Checks if mario is stuck (NOTE NEEDS WORK TO STORE OLD DATA AND REPLACE WITH NEW DATA)
            if lastX == info['x_pos']:
                counter = counter + 1
                counter2 = counter2 + 1

            # If mario dies or is stuck
            # ISSUE: When mario dies, the life doesnt update til he respawns I believe,
            # which causes the distance and time to be returned as the deafault 40 and 400
            if info['life'] == 1 or counter == 100 or i == 3000:
                
                #env.reset()

                if data.furthest < i:
                    data.furthest = i
                # Want to maximize distance, maximize time
                return 0 - info['x_pos'], 0 - info['time']

            lastX = info['x_pos']

        env.close()


