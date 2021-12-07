import time
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True

with open('TimeResult2.npy', 'rb') as f:
	x = np.load(f)

env.reset()
env.render()
time.sleep(.0166)
for indiv in x:
	for act in indiv:
		for r in range(4):
			if act == 0:
				action = 1
			elif act == 1:
				action = 3
			elif act == 2:
				action = 4
			state, reward, done, info = env.step(action)
			env.render()
			if done:
				env.reset()
				break
		if done:
			break

env.close()

