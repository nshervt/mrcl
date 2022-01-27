from torch import nn
import torch


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.cn1 = nn.Conv2d(1, 256, kernel_size=3, stride=2)
        self.cn2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.cn3 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.cn4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.cn5 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.cn6 = nn.Conv2d(256, 256, kernel_size=3, stride=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        y1 = self.relu(self.cn1(x))
        y2 = self.relu(self.cn2(y1))
        y3 = self.relu(self.cn3(y2))
        y4 = self.relu(self.cn4(y3))
        y5 = self.relu(self.cn5(y4))
        y6 = self.relu(self.cn6(y5))

        return y6.view(y6.size(0), -1)


path = '/Users/nshervt/Library/Application Support/JetBrains/PyCharm2021.2/scratches/data/models'
stat = True
saving = False
if saving:
    # -- load model
    path_model = '/Users/nshervt/Private/Research/MetaPlasticity/Codes/mrcl/saved_models/1_7/learner.model'
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


new_model = NewModel()

for idx, (old_key, (new_key, new_param)) in enumerate(zip(old_model, new_model.named_parameters())):
    new_param.data = old_model[new_key]
