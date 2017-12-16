import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from networks import FaceBox
from multibox_loss import MultiBoxLoss
from dataset import ListDataset

use_gpu = torch.cuda.is_available()

file_root = '/home/lxg/codedata/aflw/'
learning_rate = 0.001
num_epochs = 50
batch_size = 10
net = FaceBox()

net.load_state_dict(torch.load('weight/faceboxes.pt'))

print(net)
print('load pre-trined model')
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = MultiBoxLoss()
if use_gpu:
    net.cuda()

net.train()
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(net.parameters(),weight_decay=1e-4)

train_dataset = ListDataset(root=file_root,list_file='wider_aflw_label.txt',train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=5)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'wa')
for epoch in range(num_epochs):
    if epoch == 10:
        learning_rate=0.001
    if epoch == 20:
        learning_rate=0.0001
    # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
    # for param_group in optimizer.param_groups:
        # param_group['lr'] = learning_rate
    
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
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.data[0], total_loss / (i+1)))
    logfile.writelines(str(epoch) + '\t' + str(total_loss / (len(train_dataset)/batch_size)) + '\n')  
    logfile.flush()      
    torch.save(net.state_dict(),'weight/faceboxes.pt')
    

