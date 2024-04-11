import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.spec_network import SpecNetwork
import os
from utils.system_utils import searchForMaxIteration



class SpecModel:
    def __init__(self):
        self.specdecoder = SpecNetwork().cuda()
        self.optimizer = None


    def step(self, vector, view, rough):
        return self.spec_decoder(vector, view, rough)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.spec_decoder.parameters()),
             'lr': training_args.spec_decoder_lr,
             "name": "specdecoder"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "specdecoder/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.specdecoder.state_dict(), os.path.join(out_weights_path, 'specdecoder.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "specdecoder"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "specdecoder/iteration_{}/specdecoder.pth".format(loaded_iter))
        self.specdecoder.load_state_dict(torch.load(weights_path))


