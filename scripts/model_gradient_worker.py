"""One party's actual forward/backward on its own public-data mini-batch."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


class FiftyLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Linear(4,32)
        self.hidden = nn.ModuleList([nn.Linear(32,32) for _ in range(48)])
        self.exit = nn.Linear(32,2)

    def forward(self,x):
        x = torch.tanh(self.entry(x))
        for layer in self.hidden:
            x = x + .1 * torch.tanh(layer(x))
        return self.exit(x)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('data',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();args.output.mkdir()
    torch.set_num_threads(1);torch.manual_seed(args.seed);torch.use_deterministic_algorithms(True)
    model=FiftyLayerModel().double()
    data=np.load(args.data,allow_pickle=False)
    x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y']).long()
    start=time.perf_counter();prediction=model(x);loss=nn.functional.cross_entropy(prediction,y)
    loss.backward();seconds=time.perf_counter()-start
    gradients=torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]).numpy()
    weights=torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy()
    layers=[]
    for name,layer in model.named_modules():
        if isinstance(layer,nn.Linear):
            g=torch.cat([p.grad.detach().reshape(-1) for p in layer.parameters()])
            layers.append({'name':name,'inputs':layer.in_features,'outputs':layer.out_features,
                           'parameters':g.numel(),'gradient_l2':float(torch.linalg.vector_norm(g))})
    assert len(layers)==50 and len(gradients)==50914
    assert all(item['gradient_l2']>0 for item in layers)
    assert np.all(np.isfinite(gradients)) and np.all(np.isfinite(weights))
    np.save(args.output/'gradients.npy',gradients);np.save(args.output/'weights.npy',weights)
    (args.output/'model.json').write_text(json.dumps({'torch':torch.__version__,'seed':args.seed,
        'samples':len(x),'loss':float(loss.detach()),'forward_backward_seconds':seconds,
        'trainable_affine_layers':len(layers),'parameters':len(gradients),'layers':layers},indent=2)+'\n')


if __name__=='__main__':main()
