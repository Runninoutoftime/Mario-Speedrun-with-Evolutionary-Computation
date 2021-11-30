import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros, data
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import retro

from time import sleep

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
#env = retro.make('SuperMarioBros-1-1-v0', record='.')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#env = gym.wrappers.Monitor(env, "recording", force=True)

# 0 = no action, 1 = right, 2 = right + A, 3 = right + B, 4 = right + A + B, 5 = A, 6 = left

# Input into play game the number of the run and match it to an array in data.py for mutation
class MarioExample():

    def playGame(self, x):
        import time
        
        lastX = 0
        counter = 0
        i = 0

        env.reset()

        while i < data.size - 1:
            # Comment out to remove human-viewable game screen for faster performance
            env.render()
            # Renders game in 60 fps
            time.sleep(.0166)


            # Splits Actions Per Ssecond by 4 from 60 -> 15
            for r in range(4):
                state, reward, done, info = env.step(int(x[i]))

                # If mario dies, is stuck, or the game ends
                if info['life'] == 1 or counter == 15 or i == data.size - 2 or done:
                    
                    dist = info['x_pos']
                    time = info['time']
                    gene = i

                    #env.reset()
                    #print("here")
                    if gene > data.furthest:
                        data.furthest = gene

                    #print("furthest", data.furthest)
                    #print(0 - info['x_pos'], 0 - info['time'], i)
                # Want to maximize distance, maximize time
                    return 0 - dist, 0 - time, gene
            
            i = i + 1

            # Mario stuck counter
            #print("Last: ", lastX)
            #print("Current: ", info['x_pos'])
            #print("Counter: ", counter)
            if lastX == info['x_pos']:
                counter = counter + 1
            if lastX > info['x_pos']:
                counter = 0


            lastX = info['x_pos']

        #env.close()


