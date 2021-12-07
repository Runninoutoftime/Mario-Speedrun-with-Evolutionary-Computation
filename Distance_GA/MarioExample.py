from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros, data
from numpy.lib.function_base import average
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# In SIMPLE_MOVEMENT: 0 = no action, 1 = right, 2 = right + A, 3 = right + B, 4 = right + A + B, 5 = A, 6 = left
# Actions in code: 0 = right, 1 = right + B, 2 = right + A + B
# Max x_pos = 3000


# Input into play game the number of the run and match it to an array in data.py for mutation
class MarioExample():

    def playGame(self, x):
        import time
        
        # Stores x_pos of mario from last step
        lastX = 0

        # Counts number of frames mario is stuck
        counter = 0

        # Iterates through individual genes
        i = 0

        # Variable for action to perform
        action = 0

        # Maximum frames mario can be stuck
        max_stuck = 20

        env.reset()

        while i < data.size - 1:

            # Comment out to remove human-viewable game screen for faster performance
            #************************************************************************
            env.render()
            #************************************************************************
            
            
            # Renders game in 60 fps for recording purposes
            time.sleep(.0166)


            # Each action takes place for 4 frames
            for r in range(4):

                # Maps values from individual to correct action in gym-super-mario-bros simple movement dataset (commented above class)
                if x[i] == 0:
                    action = 1
                elif x[i] == 1:
                    action = 3
                elif x[i] == 2:
                    action = 4

                
                state, reward, done, info = env.step(action)

                # If mario dies, is stuck, or the game ends
                if info['life'] == 1 or counter == max_stuck or i == data.size - 2 or done:
                    
                    dist = info['x_pos']
                    time = info['time']
                    gene = i

                    # Updates data.furthest to be the furthest gene that any individual made it to
                    if gene > data.furthest:
                        data.furthest = gene
                    
                    # Updates data.dist to be the furthest x_pos any individual has been
                    if dist > data.dist:
                        data.dist = dist
                    
                    # Returns negative of dist and time because we can only minimize through pymoo
                    return 0 - dist, 0 - time, gene
            

            i = i + 1

            # Checks if mario is stuck
            if lastX == info['x_pos']:
                counter = counter + 1
            if lastX > info['x_pos']:
                counter = 0

            lastX = info['x_pos']



