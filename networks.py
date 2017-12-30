#encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

from multibox_layer import MultiBoxLayer


class CReLU(nn.Module):
	def __init__(self):
		super(CReLU, self).__init__()
	def forward(self, input):
		return torch.cat((F.relu(input), F.relu(-input)), 1)

class Inception(nn.Module):
	def __init__(self):
		super(Inception,self).__init__()
		self.conv1 = nn.Conv2d(128,32,kernel_size=1)
		self.conv2 = nn.Conv2d(128,32,kernel_size=1)
		self.conv3 = nn.Conv2d(128,24,kernel_size=1)
		self.conv4 = nn.Conv2d(24,32,kernel_size=3,padding=1)
		self.conv5 = nn.Conv2d(128,24,kernel_size=1)
		self.conv6 = nn.Conv2d(24,32,kernel_size=3,padding=1)
		self.conv7 = nn.Conv2d(32,32,kernel_size=3,padding=1)
	def forward(self,x):
		x1 = self.conv1(x)
		
		x2 = F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
		x2 = self.conv2(x2)

		x3 = self.conv3(x)
		x3 = self.conv4(x3)
		
		x4 = self.conv5(x)
		x4 = self.conv6(x4)
		x4 = self.conv7(x4)

		output = torch.cat([x1,x2,x3,x4],1)
		return output


class FaceBox(nn.Module):
	input_size = 1024
	def __init__(self):
		super(FaceBox, self).__init__()

		#model
		self.conv1 = nn.Conv2d(3,24,kernel_size=7,stride=4,padding=3)
		self.conv2 = nn.Conv2d(48,64,kernel_size=5,stride=2,padding=2)

		self.inception1 = Inception()
		self.inception2 = Inception()
		self.inception3 = Inception()

		self.conv3_1 = nn.Conv2d(128,128,kernel_size=1)
		self.conv3_2 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
		self.conv4_1 = nn.Conv2d(256,128,kernel_size=1)
		self.conv4_2 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)

		self.multilbox = MultiBoxLayer()

	def forward(self,x):
		hs = []
		
		x = self.conv1(x)
		x = torch.cat((F.relu(x), F.relu(-x)),1)
		
		x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)
		x = self.conv2(x)
		x = torch.cat((F.relu(x), F.relu(-x)),1)
		
		x = F.max_pool2d(x,kernel_size=3,stride=2,padding=1)
		x = self.inception1(x)
		x = self.inception2(x)
		x = self.inception3(x)
		
		# print('x1', x.size())
		hs.append(x)
		x = self.conv3_1(x)
		x = self.conv3_2(x)
		# print('x2', x.size())
		hs.append(x)
		x = self.conv4_1(x)
		x = self.conv4_2(x)
		# print('x3', x.size())
		hs.append(x)
		loc_preds, conf_preds = self.multilbox(hs)

		
		return loc_preds, conf_preds

if __name__ == '__main__':
	model = FaceBox()
	data = Variable(torch.randn(1,3,512,512)) 
	loc, conf = model(data)
	print('loc', loc.size())
	print('conf', conf.size())
	
