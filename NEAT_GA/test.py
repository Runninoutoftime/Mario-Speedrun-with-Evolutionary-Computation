import pickle


gen, config, pop, species_set, rndstate = pickle.load( open("winner.pkl", "rb"))

print(gen)