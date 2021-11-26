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
        i = 0

        env.reset()

        while i < 3000:
            # Comment out to remove human-viewable game screen for faster performance
            env.render()


            # Splits Actions Per Ssecond by 4 from 60 -> 15
            for r in range(4):
                state, reward, done, info = env.step(int(x[i]))

                # If mario dies, is stuck, or the game ends
                if info['life'] == 1 or counter == 100 or i == 3000 or done:
                
                    env.reset()

                    if data.furthest < i:
                        data.furthest = i
                    # if data.ids[id] < i:
                    #     data.ids[id] = i

                # Want to maximize distance, maximize time
                    return 0 - info['x_pos'], 0 - info['time'], i
            
            i = i + 1

            # Mario stuck counter (NOTE NEEDS WORK TO STORE OLD DATA AND REPLACE WITH NEW DATA)
            if lastX == info['x_pos']:
                counter = counter + 1


            lastX = info['x_pos']

        env.close()


