from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 0 = no action, 1 = right, 2 = right + A, 3 = right + B, 4 = right + A + B, 5 = A, 6 = left

class MarioExample():

    def playGame(self, x):
        lastX = 0
        counter = 0
        counter2 = 0
        i = 0

        env.reset()

        while i < 1000:
            # Comment out to remove human-viewable game screen for faster performance
            env.render()

            # Attempt to split Actions Per Ssecond by 4 from 60 -> 15
            for i in range(4):
                state, reward, done, info = env.step(int(x[i]))
            i = i + 1
            
            # Checks if mario is stuck (NOTE NEEDS WORK TO STORE OLD DATA AND REPLACE WITH NEW DATA)
            if lastX == info['x_pos']:
                counter = counter + 1
                counter2 = counter2 + 1

            # If mario dies or is stuck 15 times
            if info['life'] == 0 or counter == 50 or i == 999:
                env.reset()
                return 0 - info['x_pos']

            lastX = info['x_pos']

        env.close()


