## aflw
<!-- 20 epoch -->
Epoch [20/20], Iter [1055/1057] Loss: 1.8820

<!-- 40 epoch -->
loc_loss:0.984644 conf_loss:1.205544, pos_num:78
loc_loss:0.698231 conf_loss:1.439163, pos_num:163
loc_loss:0.644837 conf_loss:1.328891, pos_num:143
loc_loss:0.420935 conf_loss:1.332006, pos_num:149
loc_loss:0.246193 conf_loss:0.883853, pos_num:242
Epoch [20/20], Iter [1055/1057] Loss: 1.1300

## wider face, aflw
<!-- 100epoch -->
loc_loss:1.417856 conf_loss:2.061183, pos_num:152
loc_loss:0.655475 conf_loss:1.698712, pos_num:65
loc_loss:1.235528 conf_loss:2.012040, pos_num:144
loc_loss:0.798853 conf_loss:1.713999, pos_num:109
loc_loss:0.823239 conf_loss:1.953249, pos_num:318
Epoch [50/50], Iter [1700/1701] Loss: 2.7765, average_loss: 2.8849
<!-- 150epoch -->
loc_loss:0.802718 conf_loss:1.943955, pos_num:284
loc_loss:0.867129 conf_loss:1.820582, pos_num:420
loc_loss:0.885825 conf_loss:1.830107, pos_num:358
loc_loss:0.811850 conf_loss:1.881572, pos_num:501
loc_loss:0.975667 conf_loss:1.921641, pos_num:540
Epoch [50/50], Iter [680/681] Loss: 2.8973, average_loss: 2.5820

<!--继续训练  -->
46	2.27359845486
47	2.27207741518
48	2.26043195595
49	2.26634732234

<!-- 修改了数据集选择的策略，小于20px的框被舍弃，结果loss一下提高了很多-->
loc_loss:2.556887 conf_loss:2.489368, pos_num:68
loc_loss:2.234761 conf_loss:2.448641, pos_num:111
loc_loss:2.569000 conf_loss:2.495923, pos_num:105
loc_loss:2.542970 conf_loss:2.470961, pos_num:74
loc_loss:2.530408 conf_loss:2.485945, pos_num:79
Epoch [1/50], Iter [1045/1677] Loss: 5.0164, average_loss: 5.0478
loc_loss:2.033609 conf_loss:2.263520, pos_num:173
loc_loss:2.454619 conf_loss:2.263516, pos_num:109
loc_loss:2.247968 conf_loss:2.262204, pos_num:257
loc_loss:2.366087 conf_loss:2.263796, pos_num:50
loc_loss:2.315195 conf_loss:2.261710, pos_num:192
Epoch [10/50], Iter [735/1677] Loss: 4.5769, average_loss: 4.5744


# adjust
- 修改1  
复现时，2倍致密是原scale的1/4，是2倍scale的anchor的1/8，写成了1/4，已更改
- 修改2  
为每一个box label都添加了与之IOU最大的box，不管IOU阈值是多少，这样导致有inf loc loss出现，是因为targets的宽和高有的为0，也就是dataset代码中random_crop有问题，添加了对box_label的宽和高限制为10像素后，问题不再出现。
- 修改3
使用Adam

不明白为什么突然loss爆炸了
```s
loc_loss:115.657501 conf_loss:39.798553, pos_num:2528
<!-- 300 epoch -->
Epoch [300/300], Iter [400/403] Loss: 3.4930, average_loss: 3.5764
loc_loss:1.732548 conf_loss:1.807370, pos_num:1120
loc_loss:1.832072 conf_loss:2.002608, pos_num:1711
loc_loss:1.265184 conf_loss:1.525407, pos_num:624
```