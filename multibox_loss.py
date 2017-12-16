#encoding:utf-8
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable

class MultiBoxLoss(nn.Module):
	num_classes = 2
	def __init__(self):
		super(MultiBoxLoss,self).__init__()

	def cross_entropy_loss(self, x, y):
		x = x.detach()
		y = y.detach()
		xmax = x.data.max()
		#xmax = xmax.detach()
		log_sum_exp = torch.log(torch.sum(torch.exp(x-xmax), 1, keepdim=True)) + xmax
		return log_sum_exp - x.gather(1, y.view(-1,1))

	def hard_negative_mining(self,conf_loss,pos):
		'''
		conf_loss [N*21482,]
		pos [N,21482]
		return negative indice
		'''
		batch_size,num_boxes = pos.size()
		conf_loss[pos.view(-1,1)] = 0 #去掉正样本,the rest are neg conf_loss
		conf_loss = conf_loss.view(batch_size,-1)

		_,idx = conf_loss.sort(1,descending=True)
		_,rank = idx.sort(1)

		num_pos = pos.long().sum(1,keepdim=True)
		num_neg = torch.clamp(3*num_pos, max=num_boxes-1)

		neg = rank < num_neg.expand_as(rank)
		return neg

	def forward(self,loc_preds,loc_targets,conf_preds,conf_targets):
		'''
		loc_preds[batch,21842,4]
		loc_targets[batch,21842,4]
		conf_preds[batch,21842,2]
		conf_targets[batch,21842]
		'''
		batch_size,num_boxes, _ = loc_preds.size()
		#print(batch_size,num_boxes)
		#print('ok1')
		pos = conf_targets>0 #大于0的地方，说明匹配到了人脸框
		num_pos = pos.long().sum(1, keepdim=True)
		#print(torch.sum(pos))
		#print(conf_targets.size())
		num_matched_boxes = pos.data.long().sum()
		if num_matched_boxes == 0:
			return Variable(torch.Tensor([0]),requires_grad=True)
		#print('ok2')
		pos_mask1 = pos.unsqueeze(2).expand_as(loc_preds)
		#print(pos_mask1.size())
		# print('pos_mask1 sum {}'.format(torch.sum(pos_mask1)))
		pos_loc_preds = loc_preds[pos_mask1].view(-1,4)
		pos_loc_targets = loc_targets[pos_mask1].view(-1,4)
		# for i in range(num_matched_boxes):
			# print(i, pos_loc_preds[i,:], pos_loc_targets[i,:])

		loc_loss = F.smooth_l1_loss(pos_loc_preds,pos_loc_targets,size_average=False)
		# if loc_loss.data[0] > 10000: #是因为preds有非常大的，导致loss很大，所以是正常的。
		# 	print('preds', pos_loc_preds)
		# 	print('targets', pos_loc_targets)
		#print('ok3')
		#temp_conf_loss = Variable(requires_grad=False)
		conf_loss = self.cross_entropy_loss(conf_preds.view(-1,self.num_classes),
									conf_targets.view(-1,1))
		#print('conf_loss size {}'.format(conf_loss.size()))
		neg = self.hard_negative_mining(conf_loss, pos)
		pos_mask = pos.unsqueeze(2).expand_as(conf_preds)

		neg_mask = neg.unsqueeze(2).expand_as(conf_preds)
		# print('sum neg mask {} size {}'.format(torch.sum(neg_mask),neg_mask.size()))
		# print('sum pos mask {} size {}'.format(torch.sum(pos_mask),pos_mask.size()))
		#print(neg_mask)
		mask = (pos_mask+neg_mask).gt(0)
		# print('sum mask {} size {}'.format(torch.sum(mask),mask.size()))

		pos_and_neg = (pos+neg).gt(0)
		# print('sum neg {} size {}'.format(torch.sum(neg),neg.size()))
		# print('sum pos {}'.format(torch.sum(pos)))
		# print('sum pos_and_neg {}'.format(torch.sum(pos_and_neg)))
		# print('preds shape {}'.format(conf_preds.size()))
		preds = conf_preds[mask].view(-1,self.num_classes)
		targets = conf_targets[pos_and_neg]
		conf_loss = F.cross_entropy(preds,targets,size_average=False)

		N = num_pos.data.sum()
		loc_loss /= N
		conf_loss /= N
		print('loc_loss:%f conf_loss:%f, pos_num:%d' % (loc_loss.data[0], conf_loss.data[0], N))
		return loc_loss+conf_loss
