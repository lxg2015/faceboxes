#encoding:utf-8

import torch
import math
import itertools
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DataEncoder:
	def __init__(self):
		scale = 512
		feature_map_size = (16,8,4)
		boxes_size = [[32,64,128],[256],[512]]
		times_num = [[4,2,1],[1],[1]]
		num_layers = len(feature_map_size)
		boxes = []
		for i in range(num_layers):#所有layer层循环
			# boxshow=[]#show
			box_size = boxes_size[i]
			time_num = times_num[i]
			fmsize = feature_map_size[i]
			for j,bsize in enumerate(box_size):#一层的所有box_size与对应的密度循环
				t_num = time_num[j]
				anchor_x_start = (scale//(fmsize*2)) 
				anchor_y_start = (scale//(fmsize*2))
				if t_num > 1:
					x_start = anchor_x_start-int(bsize*(3./4))
					y_start = anchor_y_start-int(bsize*(3./4))
				else:
					x_start = anchor_x_start-int(bsize*(1./2))
					y_start = anchor_y_start-int(bsize*(1./2))
				x_start_initinal = x_start
				for k in range(fmsize): #一层的一种box_size的列循环
					y_start += (scale//fmsize)
					x_start = x_start_initinal
					#print(y_start)
					for l in range(fmsize):#一层的一种box_size的行循环
						x_start += (scale//fmsize)
						for m in range(t_num):
							for n in range(t_num):
								x1 = x_start + m* (bsize/t_num)
								y1 = y_start + n* (bsize/t_num)
								x2 = x1 + bsize
								y2 = y1 + bsize
								boxes.append((x1,y1,x2,y2))
								# boxshow.append((x1,y1,x2,y2))
			'''print(len(boxshow))
			img = np.zeros((512,512,3),np.uint8)
			for box in boxshow:
				print(box)
				cv2.rectangle(img,box[:2],box[2:],(0,255,0),1)
			plt.figure()
			plt.imshow(img,'brg')
			plt.show()'''
					

		self.default_boxes = torch.Tensor(boxes)/scale
		#print(self.default_boxes)

	def iou(self,box1,box2):
		'''
		计算两组box之间的iou
		box1 [x,y,h,w]
		box2 [x1,y1,x2,y2]
		box1 (tensor) sized[N,4]
		box2 (tensor) sized[M,4]
		output:iou (tensor) sized[N,M]
		'''
		N = box1.size(0)
		M = box2.size(0)
		lt = torch.max(
			box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
			box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
		)

		rb = torch.min(
			(box1[:,:2]+box1[:,2:]).unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
			box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
		)

		wh = rb - lt  # [N,M,2]
		wh[wh<0] = 0  # clip at 0
		inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

		#area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
		area1 = (box1[:,2]) * (box1[:,3])
		area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
		area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
		area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

		iou = inter / (area1 + area2 - inter)
		return iou

	def encode(self,boxes,classes,threshold=0.45):
		'''
		boxes:真实目标框(x,y,w,h)
		default_box (x1,y1,x2,y2)
		return:boxes: (tensor) [num_face,21824,4]
		classes:class label [obj,]
		'''
		#print(boxes,classes)
		default_boxes = self.default_boxes #[21824,4]
		num_default_boxes = default_boxes.size(0)
		num_obj=boxes.size(0)#人脸个数
		#print('num_faces {}'.format(num_obj))
		#iou = self.iou(boxes,torch.cat([default_boxes[:,:2] - default_boxes[:,2:]/2,
		#	default_boxes[:,:2] + default_boxes[:,2:]/2], 1))
		iou = self.iou(boxes,default_boxes)
		#print('iou size {}'.format(iou.size()))
		iou ,max_index= iou.max(0) #看哪一个boxes匹配default_boxes
		#print(max(iou))
		max_index.squeeze_(0)
		iou.squeeze_(0)
		#print(iou)
		boxes = boxes[max_index]
		#variances = [0.1, 0.2]
		'''cxcy = (boxes[:,:2] + boxes[:,2:])/2 - default_boxes[:,:2]#[21824,2]
		cxcy /= variances[0] * default_boxes[:,2:]
		wh = (boxes[:,2:] - boxes[:,:2]) / default_boxes[:,2:] # [21824,2]
		wh = torch.log(wh) / variances[1]'''
		cxcy = boxes[:,:2]+0.5*boxes[:,2:] - (default_boxes[:,:2]+default_boxes[:,2:])*0.5 #中心点
		cxcy /= (default_boxes[:,2:]-default_boxes[:,:2])
		wh = (boxes[:,2:])/(default_boxes[:,2:]-default_boxes[:,:2])
		wh = torch.log(wh)
		loc = torch.cat([cxcy, wh], 1) # [21824,4]

		conf = classes[max_index] #其实都是1 [21824,]
		conf[iou < threshold] = 0 #iou小的设为背景

		#print(loc,conf)
		return loc,conf

	def nms(self,bboxes,scores,threshold=0.5):
		'''
		bboxes(tensor) [N,4]
		scores(tensor) [N,]
		'''
		x1 = bboxes[:,0]
		y1 = bboxes[:,1]
		x2 = bboxes[:,2]
		y2 = bboxes[:,3]

		_,order = scores.sort(0,descending=True)
		keep = []
		while order.numel() > 0:
			i = order[0]
			keep.append(i)

			if order.numel() == 1:
				break

			xx1 = x1[order[1:]].clamp(min=x1[i])
			yy1 = y1[order[1:]].clamp(min=y1[i])
			xx2 = x2[order[1:]].clamp(max=x2[i])
			yy2 = y2[order[1:]].clamp(max=y2[i])

			w = (xx2-xx1).clamp(min=0)
			h = (yy2-yy1).clamp(min=0)
			inter = w*h

			ovr = inter / (areas[i] + areas[order[1:]] - inter)
			ids = (ovr<=threshold).nonzero().squeeze()
			if ids.numel() == 0:
				break
			order = order[ids+1]
		return torch.LongTensor(keep)

	def decode(self,loc,conf):
		'''
		將预测出的 loc/conf转换成真实的人脸框
		loc [21842,4]
		conf [21824,2]
		'''
		cxcy = loc[:,:2] * (self.default_boxes[:,2:]-self.default_boxes[:,:2])
		xy = cxcy - (self.default_boxes[:,2:]-self.default_boxes[:,:2])*0.5
		wh  = torch.exp(loc[:,2:]) * (self.default_boxes[:,2:]-self.default_boxes[:,:2])
		boxes = torch.cat([xy,xy+wh],1) #[21824,4]

		max_conf, labels = conf.max(1) #[21842,1]
		print(max_conf)
		# print labels
		ids = labels.nonzero()#.squeeze(1)
		print(ids)

		keep = self.nms(boxes[ids],max_conf[ids].squeeze(1))

		return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]

if __name__ == '__main__':
	dataencoder = DataEncoder()
	print((dataencoder.default_boxes))
	boxes = torch.Tensor([[-8,-8,24,24],[400,400,500,500]])/1024
	dataencoder.encode(boxes,torch.Tensor([1,1]))
