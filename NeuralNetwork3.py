#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:42:19 2023

@author: ubuntu
"""

import numpy as np
import math

softmax_output=[0.2,0.4,0.6]
softmax_output2=[0.2,0.4,0.6]
hotCode=[1,0,0]
hotCode2=[0,1,0]

#%%

class Entropy_loss:
    def __init__(self, softmax_output, hotCode):
        self.softmax_output=softmax_output
        self.hotCode=hotCode
        lossMatrix=[]
        for i,j in zip(softmax_output,hotCode):
            loss=-np.log(i)*j
            lossMatrix.append(loss)
        self.output=lossMatrix
        
        
        
case1=Entropy_loss(softmax_output2, hotCode2)
print(case1.output)


#%%
softmax_out=[[1,2,3],
             [2,3,4],
             [4,3,2]]

class_target=[0,1,2]

for target_index, dist in zip(class_target,softmax_out):
    print(dist[target_index])

#%%
import numpy as np
softmax_out=np.array([[1,2,3],
                     [2,3,4],
                     [4,3,2]])
class_target=[0,1,1]

loss_neg=-np.log(softmax_out[range(len(softmax_out)),class_target])
avg_loss=np.mean(loss_neg)
print(loss_neg,avg_loss)

#%%
import numpy as np

print(-np.log(1-1e-7))
 y_pred_clip=np.clip(y_pred, 1e-7,1-1e-7)
 










