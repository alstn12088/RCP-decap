from simulator.decap_sim import decap_sim
from policy_net.Reward_AM import AttentionModel
import numpy as np
import torch

def generate_decap_data():
    
 
    probing_port = [23]


  

    zeros = np.zeros((len(probing_port)*10,100, 3))
    
    normalization = 10
    for j in range(10):
        for i in range(len(probing_port)):
            for x in range(10):
                num_restriction = 0
                for y in range(10):
    
                    zeros[i+j*100,x*10+y, 0] = x/normalization
                    zeros[i+j*100,x*10+y,1] = y/normalization
                    
                    if x*10+y ==probing_port[i]:             
                        zeros[i+j*100,x*10+y,2] = 2
                        
    
#     zeros = np.stack([zeros[1],zeros[9],zeros[21],zeros[37],zeros[43],zeros[55],zeros[59],zeros[77],zeros[81],zeros[93]])

    
    
    return zeros


model = AttentionModel(
    128,
    128,
    n_encode_layers=3,
    mask_inner=True,
    mask_logits=True, 
).cuda()


solution = torch.FloatTensor(np.load("offline_data/solution.npy")).cuda()

problem  = torch.FloatTensor(np.zeros((2,100, 3))).cuda()

for i in range(10):
    for j in range(10):
        problem[:,10*i+j,0] = i
        problem[:,10*i+j,1] = j
        if i*10 + j == 32:
            problem[:,10*i+j,2] = 2 




model(problem,reward_query = solution[:2,1],action = solution[:2,1:])