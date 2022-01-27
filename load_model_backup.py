from torch import nn
import torch


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, kernel_size=7, stride=2)
        self.cn2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.cn3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.relu = nn.ReLU()


path = '/Users/nshervt/Private/Research/PostDoc/MetaPlasticity/Codes/MetaPlasticity/data/models/omniglot_example'
stat = True
saving = True
if saving:
    # -- load model
    path_model = '/Users/nshervt/Private/Research/PostDoc/MetaPlasticity/Codes/mrcl-master/results/27January2022/OML_Omniglot/0/1_1/learner.model'
    old_model = torch.load(path_model, map_location='cpu')

    new_model = NewModel()

    for idx, ((old_key, old_param), (new_key, new_param)) in enumerate(zip(old_model.named_parameters(), new_model.named_parameters())):
        if idx < 12:
            new_param.data = old_param.data

    if not stat:
        torch.save(new_model, path + '/model.pth')
    else:
        torch.save(new_model.state_dict(), path + '/model_stat.pth')
else:
    if not stat:
        old_model = torch.load(path + '/model.pth')
    else:
        old_model = torch.load(path + '/model_stat.pth')
