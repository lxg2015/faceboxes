import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from networks import FaceBox
from multibox_loss import MultiBoxLoss
from dataset import ListDataset

import visdom
import numpy as np

use_gpu = torch.cuda.is_available()
file_root = os.path.expanduser('~/codedata/aflw/')

learning_rate = 0.001
num_epochs = 300
batch_size = 32

net = FaceBox()
if use_gpu:
    net.cuda()

print('load model...')
# net.load_state_dict(torch.load('weight/faceboxes.pt'))

criterion = MultiBoxLoss()

# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0003)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

train_dataset = ListDataset(root=file_root,list_file='label/box_label.txt',train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=5)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))

num_iter = 0
vis = visdom.Visdom()
win = vis.line(Y=np.array([0]), X=np.array([0]))

net.train()
for epoch in range(num_epochs):
    if epoch == 190 or epoch == 250:
        learning_rate *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.

    for i,(images,loc_targets,conf_targets) in enumerate(train_loader):
        images = Variable(images)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)
        if use_gpu:
            images,loc_targets,conf_targets = images.cuda(),loc_targets.cuda(),conf_targets.cuda()
        
        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds,loc_targets,conf_preds,conf_targets)
        total_loss += loss.data[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.data[0], total_loss / (i+1)))
            num_iter = num_iter + 1
            vis.line(Y=np.array([total_loss / (i+1)]), X=np.array([num_iter]), 
                    win=win,
                    update='append')
    
    if not os.path.exists('weight/'):
        os.mkdir('weight')  
    print('saving model ...')  
    torch.save(net.state_dict(),'weight/faceboxes.pt')
    

