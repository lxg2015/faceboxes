path = '/home/lxg/codedata/widerFace/wider_face_split/'

with open(path+'wider_face_train_bbx_gt.txt') as f:
    lines = f.readlines()
    nums = len(lines)

output = open(path+'box_label.txt', 'w')

for i in range(nums):
    # if i == 10:
    #     break

    line = lines[i]
    if 'jpg' not in line:
        continue
    
    im_name = line.strip()
    face_num = int(lines[i+1].strip())
    im_name = im_name + ' ' + str(face_num) + ' '

    for j in range(face_num):
        line = lines[i+2+j] 
        splited = line.strip().split()
        im_name = im_name + splited[0] + ' '
        im_name = im_name + splited[1] + ' '
        im_name = im_name + splited[2] + ' '
        im_name = im_name + splited[3] + ' '
        im_name = im_name + '1' + ' '
        
    # print(i)    
    output.writelines('widerFace/WIDER_train/images/' + im_name+'\n')

# create aflw data
# with open('label.txt') as f:
#     lines = f.readlines()
# output = open('label_path.txt', 'w') 
# for line in lines:
#     output.writelines('data/all/' + line)


