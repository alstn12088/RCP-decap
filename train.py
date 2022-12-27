import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from dataset import SequenceDataset, ScoreDataset,ZipDataset
from policy_net.Reward_AM import AttentionModel
from tqdm import tqdm
import wandb
wandb.init(project="designcon_2023", entity="alstn12088")


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
    return zeros    

def preprocess(filename, max_reward=12):

    data = np.load(filename)

    rewards = data[:,0]
    sequences = data[:,1:]
    sequences = sequences[rewards<max_reward]
    rewards = rewards[rewards<max_reward]


    return rewards, sequences

def collate(data_list):
    target_data_list, cond_data_list = zip(*data_list)
    batched_target_data = SequenceDataset.collate(target_data_list)
    batched_cond_data = ScoreDataset.collate(cond_data_list)
    
    return batched_target_data, batched_cond_data

def compute_sequence_cross_entropy(logits, batched_sequence_data):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    return loss


def evaluate(model,num_samples=100,query=0.9):
    model.eval()
    problem = torch.FloatTensor(np.zeros((num_samples,100, 3))).cuda()

    for i in range(10):
        for j in range(10):
            problem[:,10*i+j,0] = i
            problem[:,10*i+j,1] = j
            if i*10 + j == 32:
                problem[:,10*i+j,2] = 2
    query = torch.ones(num_samples,1).cuda() * query
    cost, solution = model(problem,reward_query = query,return_pi = True)
    wandb.log({'mean_reward':-cost.mean()})
    wandb.log({'max_reward':-cost.min()})
    wandb.log({'min_reward':-cost.max()})
    model.train()
    
    return cost





def train(hparams):
    rewards, sequences = preprocess(hparams.filename)
    
    
    rewards = (rewards - 10.0)/(13.5-10.0)
    
    #input_data = input[sequences,:] = 1

    #input_dataset = GraphDataset(input_data.to_list())
    target_dataset = SequenceDataset(sequences.tolist())
    score_dataset = ScoreDataset(rewards.tolist(),np.mean(rewards,dtype=np.float),np.std(rewards,dtype=np.float))
    

    train_dataset = ZipDataset(target_dataset, score_dataset)

    train_loader = DataLoader(train_dataset,
                batch_size=hparams.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=8,
                drop_last=False)

    
    
    
    scores_np = score_dataset.get_tsrs().view(-1).numpy()
    ranks = np.argsort(np.argsort(-1 * scores_np))
    weights = 1.0 / (1e-3 * len(scores_np) + ranks)
    sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(scores_np), replacement=True
            )

    loader = torch.utils.data.DataLoader(
        train_dataset, 
        sampler=sampler, 
        batch_size=hparams.batch_size, 
        collate_fn=collate,
        drop_last=True
        )


    model = AttentionModel(
    128,
    128,
    n_encode_layers=3,
    mask_inner=True,
    mask_logits=True, 
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    model.train()
    for epoch in tqdm(range(hparams.max_epochs)):
        
        if hparams.sampling =="reweighting":
            step = 0
            while step < 10:
                step += 1
                try: 
                    batched_data = next(data_iter)
                except:
                    data_iter = iter(loader)
                    batched_data = next(data_iter)
                

                batched_target_data, batched_cond_data = batched_data
                problem = torch.FloatTensor(np.zeros((batched_target_data.shape[0],100, 3))).cuda()

                for i in range(10):
                    for j in range(10):
                        problem[:,10*i+j,0] = i
                        problem[:,10*i+j,1] = j
                        if i*10 + j == 32:
                            problem[:,10*i+j,2] = 2             

                _, log_likelihood_IL= model(problem, reward_query = batched_cond_data.view(-1,1).float().cuda(), action = batched_target_data.cuda())
                
                # importance sampling
                loss = (-log_likelihood_IL).mean()
                wandb.log({'loss':loss})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.gradient_clip_val)
                optimizer.step()
            if epoch%10==0:
                cost = evaluate(model) 
        
        if hparams.sampling =="uniform":
            for _, batched_data in enumerate(train_loader):
                

                batched_target_data, batched_cond_data = batched_data
                problem = torch.FloatTensor(np.zeros((batched_target_data.shape[0],100, 3))).cuda()

                for i in range(10):
                    for j in range(10):
                        problem[:,10*i+j,0] = i
                        problem[:,10*i+j,1] = j
                        if i*10 + j == 32:
                            problem[:,10*i+j,2] = 2             

                _, log_likelihood_IL= model(problem, reward_query = batched_cond_data.view(-1,1).float().cuda(), action = batched_target_data.cuda())
                
                # importance sampling
                loss = (-log_likelihood_IL).mean()
                wandb.log({'loss':loss})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.gradient_clip_val)
                optimizer.step()
            if epoch%10==0:
                cost = evaluate(model)
        
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict()
        #     },  os.path.join('trained_models/test2/', 'epoch-{}.pt'.format(epoch)))
    

        

 
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--filename", type=str, default='offline_data/rs_data_probe23_20000.npy')
    parser.add_argument("--sampling", type=str, default='uniform')

    
    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # # sampling
    # parser.add_argument("--num_queries", type=int, default=1000)
    # parser.add_argument("--query_batch_size", type=int, default=250)
    # parser.add_argument("--num_workers", type=int, default=8)
    # parser.add_argument("--max_len", type=int, default=120)   

    hparams = parser.parse_args()
    train(hparams)