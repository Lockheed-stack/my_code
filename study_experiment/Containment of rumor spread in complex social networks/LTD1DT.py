#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#%%
class LTD1DT_model:
    def __init__(self,G:nx.Graph,W,theta,theta_R,seed_R,seed_T) -> None:
        self.final_R_receiver = {}
        self.final_T_receiver = {}
        pass

    def diffusion_simulation(self):
        pass

    def get_final_R_receiver(self):
        return self.final_R_receiver

    def get_final_T_receiver(self):
        return self.final_T_receiver