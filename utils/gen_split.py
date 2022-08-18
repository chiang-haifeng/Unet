# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: gen_split.py
Author: chenming
Create Date: 2022/2/6
Description：
-------------------------------------------------
"""
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: sd_segmentation
File Name: gen_split.py.py
Author: chenming
Create Date: 2022/2/6
Description：
-------------------------------------------------
"""
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project £ºai_train_code 
@File    £ºgen_txt.py
@Author  £ºChenmingSong
@Date    £º2021/12/31 18:42 
@Description£º
'''
import os
import random
import numpy as np

# annotations_foder_path = "E:/biye/gogogo/english/detection_yolov5/official/data/origin/VOC2012/Annotations"
annotations_foder_path = "/scm/data/seg/xianyu/skin_seg/data/ISBI2016_ISIC_Part1_Training_Data"
names = os.listdir(annotations_foder_path)
real_names = [name.split(".")[0] for name in names]
print(real_names)
random.shuffle(real_names)
print(real_names)
length = len(real_names)
split_point = int(length * 0.3)

val_names = real_names[:split_point]
train_names = real_names[split_point:]

# ¿ªÊ¼Éú³ÉÎÄ¼þ
np.savetxt('val.txt', np.array(val_names), fmt="%s", delimiter="\n")
np.savetxt('test.txt', np.array(val_names), fmt="%s", delimiter="\n")
np.savetxt('train.txt', np.array(train_names), fmt="%s", delimiter="\n")
# print("txtÎÄ¼þÉú³ÉÍê±Ï£¬Çë·ÅÔÚVOC2012µÄImageSets/MainµÄÄ¿Â¼ÏÂ")

np.savetxt('bbbbb.txt', np.array(real_names), fmt="%s", delimiter="\n")
