import torch
from torch import nn
from torch.nn import DataParallel
import numpy as np
import math
from typing import NamedTuple
from policy_net.graph_encoder import GraphAttentionEncoder
from simulator.decap_sim import decap_sim
from policy_net.state import StateDecap



def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class CachedLookup(object):

    def __init__(self, data):
        self.orig = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        assert not isinstance(key, slice), "CachedLookup does not support slicing, " \
                                           "you can slice the result of an index operation instead"

        assert torch.is_tensor(key)  # If tensor, idx all tensors by this tensor:

        if self.key is None:
            self.key = key
            self.current = self.orig[key]
        elif len(key) != len(self.key) or (key != self.key).any():
            self.key = key
            self.current = self.orig[key]

        return self.current



def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    
    probing = input[0][:,:,2]==2
    keep_out = input[0][:,:,2]==1    
    for i in range(iter_rep):

        _log_p, pi,_ = inner_func(input[1].view(100,input[0].shape[0],-1),None,probing,keep_out )
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input[0], pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts



class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)

class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',num_decap = 20, input_dim = 3, 
                 n_heads=8):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0


        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits


        self.n_heads = n_heads
        self.num_decap = num_decap


        
        # Problem specific context parameters (placeholder and step context dimension)

        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        node_dim = input_dim  # x, y, is_probing
        
        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        self.cond_embedding = nn.Linear(1, embedding_dim)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)




    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input,reward_query, action=None, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        self.probing_point = torch.where(input[:,:,2]==2)[1].clone().view(-1,1)
        self.probing = input[:,:,2]==2
        self.keep_out = input[:,:,2]==1
  


        embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input,reward_query, embeddings,action)

        if action==None:

                cost, mask = self.get_costs(input, pi)
                ll = self._calc_log_likelihood(_log_p, pi, mask)
            
        else:
            ll = self._calc_log_likelihood(_log_p, action, None)
            cost = 0

        # cost, mask = self.problem.get_costs(input, pi, self.raw_pdn,self.z_init_list)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        # ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, pi
        # np.save('PGRL_IL_solution_R', pi.cpu().numpy())
        # np.save('PGRL_IL_rewards_R', cost.cpu().numpy()) 
        return cost, ll

    def get_costs(self,dataset,pi):
        reward_list = []
        for i in range(pi.size(0)):


            reward = decap_sim(probe = (dataset[i,:,2]==2).nonzero().item(), solution = pi[i].cpu().numpy().tolist())
          
            reward_list.append(reward)
        reward = torch.FloatTensor(reward_list).cuda().view(-1,1)
        
        return -reward, None


    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))


    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions

        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        return self.init_embed(input)

    def make_state(*args, **kwargs):
        return StateDecap.initialize(*args, **kwargs)
    

    def _inner(self, input,reward_query, embeddings,action=None, probing=None, keep_out=None):

        # in the case that only inner is called by eval.py function
        #######################################
        if probing is not None:
            self.probing = probing
        if keep_out is not None:
            self.keep_out = keep_out 
        #######################################

        
        outputs = []
        sequences = []

        state = StateDecap.initialize(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings,reward_query)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        for i in range(self.num_decap):


            log_p, mask = self._get_log_p(fixed, state,probing = self.probing, keep_out = self.keep_out)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            if action==None:
                selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            else:
                
                selected = action[:,i]

            state = state.update(selected)



            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        # if self.decode_type == "greedy":
        #     _, selected = probs.max(1)
        #     assert not mask.gather(1, selected.unsqueeze(
        #         -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        #elif self.decode_type == "sampling":
        selected = probs.multinomial(1).squeeze(1)

        # Check if sampling went OK, can go wrong due to bug on GPU
        # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
        while mask.gather(1, selected.unsqueeze(-1)).data.any():
            print('Sampled bad values, resampling!')
            selected = probs.multinomial(1).squeeze(1)

        return selected

    def _precompute(self, embeddings,reward_query, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        
        graph_embed += self.cond_embedding(reward_query)

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)



    def _get_log_p(self, fixed, state, probing=None, keep_out=None,normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # expands like mask: [batch, 1, problem_size]
        probing = probing.view(probing.shape[0],1,-1)
        keep_out = keep_out.view(keep_out.shape[0],1,-1)

 
        mask = ~(~mask*~probing*~keep_out)


        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)


        log_p = torch.clip(torch.log_softmax(log_p / self.temp, dim=-1),min=-10)
        
        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()


        
        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if state.i.item() == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
        
                return embeddings.gather(
                    1,
                    torch.cat((self.probing_point, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
        # More than one step, assume always starting with first
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )


    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):


        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
