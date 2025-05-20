import einops
import torch
from dataset_management.dataset_manager.dataloader.codebook import Codebook, ObjectiveVAP, bin_times_to_frames
import itertools
import numpy as np


class VAPDecoder():


    def __init__(self, bin_times) -> None:
        self.bin_times = bin_times
        self.oj = ObjectiveVAP(bin_times=bin_times)

    def p_future(self, vap):
        if len(vap.shape)<3:
            vap = vap.unsqueeze(0)
        return self.oj.probs_next_speaker(vap, 2, 3).squeeze()

    def p_now(self, vap):
        if len(vap.shape)<3:
            vap = vap.unsqueeze(0)
        return self.oj.probs_next_speaker(vap, 0, 1).squeeze()

    def decode_probabilities(self, vap):

        # decode and turn into probs 
        if len(vap.shape)<3:
            vap = vap.unsqueeze(dim=0)

        # now
        p0 = self.oj.probs_next_speaker(vap, 0, 0)
        p1 = self.oj.probs_next_speaker(vap, 1, 1)
        p2 = self.oj.probs_next_speaker(vap, 2, 2)
        p3 = self.oj.probs_next_speaker(vap, 3, 3)

        p_all = torch.stack((p0,p1,p2,p3), dim=-1)

        return p_all.squeeze()

    def p_now_p_future(self, vap):

        
        # decode and turn into probs 

        # now
        p_now = self.oj.probs_next_speaker(vap, 0, 1)
        p_future = self.oj.probs_next_speaker(vap, 2, 3)

        p_all = torch.stack((p_now, p_future), dim=-1)

        return p_all
    
    def p_all(self, vap):

        if len(vap.shape)<3:
            vap = vap.unsqueeze(0)

        idx = torch.arange(self.oj.codebook.n_classes).to(vap.device)
        states = self.oj.codebook.decode(idx)

        # need to find [[x,x,1,1],[x,x,0,0]] indices 
        # and to find  [[x,x,0,0],[x,x,1,1]] indices of states 

        abp = states
        # Dot product over all states
        all_bins = torch.einsum("bnd,dck->bnck", vap, abp)
        
        return all_bins
    
    def p_bc_idxs(self, who):

        assert who in [0,1]
        
        # who speaks the bc is automatically 1
        invert = False

        if who == 0:
            
            # everything is inverted
            invert = True

        # construct states relevant to a backchannel
        combi = np.array(list(itertools.product([0, 1], repeat=3)))
        bottom = np.hstack((combi, np.ones((combi.shape[0],1))))

        # at least 1 active
        top = combi[1:, :]
        top = np.hstack((top, np.zeros((top.shape[0],1))))

        result = []
        for i in range(top.shape[0]):
            for j in range(bottom.shape[1]):
                
                top_state = top[i, :]
                bottom_state = bottom[j, :]
                
                if not invert:
                    stack = (top_state, bottom_state)
                elif invert:
                    stack = (bottom_state, top_state)

                state = np.stack(stack, axis=0)
                result.append(state)

        # states for backchannel...
        states = torch.tensor(np.stack(result, axis=0)).float()
        idxs = self.oj.codebook.encode(states)

        return idxs
    
    def p_bc(self, vap):

        idx_0 = self.p_bc_idxs(0)
        idx_1 = self.p_bc_idxs(1)

        ap = vap[..., idx_0].sum(-1)
        bp = vap[..., idx_1].sum(-1)
        
        res = torch.stack((ap, bp), dim=-1)

        return res


