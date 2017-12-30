from networks import FaceBox
from encoderl import DataEncoder

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
print('opencv version', cv2.__version__)

def detect(im):
    im = cv2.resize(im, (1024,1024))
    im_tensor = torch.from_numpy(im.transpose((2,0,1)))
    im_tensor = im_tensor.float().div(255)
    print(im_tensor.shape)
    loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0), volatile=True))
    boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0),
                                                F.softmax(conf.squeeze(0)).data)
    return boxes, probs

def testVideo(file):
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print("video cann't open")

    _, im = cap.read()
    h,w,_ = im.shape

    while True:
        _,im = cap.read()
        boxes,_ = detect(im)

        print(boxes)
        for box in boxes:
            x1 = int(box[0]*w)
            x2 = int(box[2]*w)
            y1 = int(box[1]*h)
            y2 = int(box[3]*h)
            print(x1, y1, x2, y2, w, h)
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow("video", im)
        cv2.waitKey(2)

def testIm(file):
    im = cv2.imread(file)
    if im is None:
        print("can not open image:", file)
        return
    h,w,_ = im.shape
    boxes, probs = detect(im)
    print(boxes)
    for i, (box) in enumerate(boxes):
        print('i', i, 'box', box)
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        print(x1, y1, x2, y2, w, h)
        cv2.rectangle(im,(x1,y1+4),(x2,y2),(0,255,0),2)
        cv2.putText(im, str(probs[i]), (x1,y1), font, 0.4, (0,255,0))
    cv2.imshow('photo', im)
    cv2.waitKey(0)
    return im

def testImList(path, file_name):
    with open(path+file_name) as f:
        file_list = f.readlines()

    for item in file_list:
        testIm(path+item.strip()+'.jpg')

def saveFddbData(path, file_name):
    '''
    Args:
        file_name: fddb image list
    '''
    with open(path+file_name) as f:
        file_list = f.readlines()
    f_write = open(path+'predict.txt', 'w')
    
    image_num = 0
    for item in file_list:
        item = item.strip()
        if not ('/' in item):
            continue
        image_num += 1
        im = cv2.imread(path+item+'.jpg')
        if im is None:
            print('can not open image', item)
            return
        h,w,_ = im.shape
        boxes, probs = detect(im)
        f_write.write(item+'\n')
        f_write.write(str(boxes.size(0))+'\n')
        print('image_num', image_num, 'box_num', boxes.size(0))
        for i, (box) in enumerate(boxes):
            x1 = box[0]*w
            x2 = box[2]*w
            y1 = box[1]*h
            y2 = box[3]*h
            f_write.write(str(x1)+'\t'+str(y1)+'\t'+str(x2-x1)+'\t'+str(y2-y1)+'\t'+str(probs[i])+'\t'+'1\n')
    f_write.close()

def getFddbList(path, file_name):
    with open(path+file_name) as f:
        file_list = f.readlines()
    f_write = open(path+'fddblist.txt', 'w')
    for item in file_list:
        if '/' in item:
            f_write.write(item)
    f_write.close()
    print('get fddb list done')

if __name__ == '__main__':
    net = FaceBox()
    net.load_state_dict(torch.load('weight/faceboxes.pt', map_location=lambda storage, loc:storage))
    
    net.eval()
    data_encoder = DataEncoder()

    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    
    # given video path, predict and show 
    path = "/home/lxg/codedata/faceVideo/1208.mp4"
    # testVideo(path)    

    # given image path, predict and show
    root_path = "/home/lxg/codedata/widerFace/WIDER_train/images/0--Parade/"
    picture = '0_Parade_marchingband_1_495.jpg'
    # testIm(root_path + picture)

    # given image path, predict and show
    fddb_path = "/home/lxg/codedata/fddb/2002/07/19/big/"
    picture = 'img_463.jpg'
    # im = testIm(fddb_path + picture)
    # cv2.imwrite('picture/'+picture, im)

    # given image file list, predict and show
    path = '/home/lxg/codedata/fddb/'
    file_name = 'FDDB-folds/FDDB-fold-01.txt'
    # testImList(path, file_name)

    # get fddb preddict and write them to predict.txt
    path = '/home/lxg/codedata/fddb/'
    file_name = 'fddb.txt'
    saveFddbData(path, file_name)
    # getFddbList(path, file_name)