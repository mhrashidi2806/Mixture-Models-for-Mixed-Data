#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import torch
import torch.optim as optim
from mixture_model import mixture_model

import matplotlib.pyplot as plt
np.random.seed(1)
torch.manual_seed(1)
torch.set_default_dtype(torch.float64)

data = np.load(".../data/data.npz")
X_tr=torch.tensor(data['X_train']).float()
Y_tr=torch.tensor(data['Y_train']).float()
X_te=torch.tensor(data['X_test']).float()
Y_te=torch.tensor(data['Y_test']).float()
#************3-b
rt = mixture_model(5, 100, 16, 64)

test2 = rt.fit(X_tr,Y_tr,0.01,30)[0]

miuu = rt.fit(X_tr,Y_tr,0.01,30)[1]
sigmaa = rt.fit(X_tr,Y_tr,0.01,30)[2]
phii = rt.fit(X_tr,Y_tr,0.01,30)[3]
pii = rt.fit(X_tr,Y_tr,0.01,30)[4]

#**

Xnew = X_te[:,None,:]
miuX = Xnew - miuu
normx = ((1 / (torch.sqrt(2 * torch.pi * ((sigmaa) ** 2)))) * (torch.exp((-((miuX) ** 2) / (2 * ((sigmaa) ** 2))))))
bbb = torch.prod(normx,2)
Ynew = Y_te[:,None,:]
bery = (((phii) ** (Ynew)) * ((1 - phii) ** (1 - Ynew)))
aaa = torch.prod(bery,2)

bernor = bbb * aaa
finaln = bernor@(pii)
ddd = finaln.detach().numpy()
ccc = bernor.detach().numpy()
sumfinal = -torch.sum(torch.log(finaln))
print(sumfinal)
print(test2)
######
answertest2 = test2.detach().numpy()
plt.plot(range(1,31), answertest2, label='alpha = 0.01')
plt.ylim(-1200000,-400000)
plt.xlabel('epochs')
plt.ylabel('NLML for Trainset')
plt.title('Question 3-b')
plt.legend()
plt.show()

#**********3-c



nmiusigm = miuu - 2*sigmaa
pmiusigm = miuu + 2*sigmaa
nnmiusigm = nmiusigm.detach().numpy()
ppmiusigm = pmiusigm.detach().numpy()

XTR = X_tr.detach().numpy()
YTR = Y_tr.detach().numpy()

fig, ax = plt.subplots()
ax.fill_between( range(1,101),nnmiusigm[0,:], ppmiusigm[0,:], label = 'Miu-+2*Sigma',alpha = 0.4)
plt.xlabel('d')
plt.ylabel('Miu-+2*Sigma')
plt.title('Question 3-c, K=1')
plt.legend()
plt.show()

fig2, ax2 = plt.subplots()
ax2.fill_between( range(1,101),nnmiusigm[1,:], ppmiusigm[1,:], label = 'Miu-+2*Sigma',alpha = 0.4)
plt.xlabel('d')
plt.ylabel('Miu-+2*Sigma')
plt.title('Question 3-c, K=2')
plt.legend()
plt.show()

fig3, ax3 = plt.subplots()
ax3.fill_between( range(1,101),nnmiusigm[2,:], ppmiusigm[2,:], label = 'Miu-+2*Sigma',alpha = 0.4)
plt.xlabel('d')
plt.ylabel('Miu-+2*Sigma')
plt.title('Question 3-c, K=3')
plt.legend()
plt.show()

fig4, ax4 = plt.subplots()
ax4.fill_between( range(1,101),nnmiusigm[3,:], ppmiusigm[3,:], label = 'Miu-+2*Sigma',alpha = 0.4)
plt.xlabel('d')
plt.ylabel('Miu-+2*Sigma')
plt.title('Question 3-c, K=4')
plt.legend()
plt.show()

fig5, ax5 = plt.subplots()
ax5.fill_between( range(1,101),nnmiusigm[4,:], ppmiusigm[4,:], label = 'Miu-+2*Sigma',alpha = 0.4)
plt.xlabel('d')
plt.ylabel('Miu-+2*Sigma')
plt.title('Question 3-c, K=5')
plt.legend()
plt.show()

valuephi = phii.detach().numpy()
#  Bar plot
plt.bar(range(1,17), valuephi[0,])
plt.xlabel("differenDy")
plt.ylabel("value of phi")
plt.title("bar chart for phi - k=1")
plt.show()

plt.bar(range(1,17), valuephi[1,])
plt.xlabel("differenDy")
plt.ylabel("value of phi")
plt.title("bar chart for phi - k=2")
plt.show()

plt.bar(range(1,17), valuephi[2,])
plt.xlabel("differenDy")
plt.ylabel("value of phi")
plt.title("bar chart for phi - k=3")
plt.show()

plt.bar(range(1,17), valuephi[3,])
plt.xlabel("differenDy")
plt.ylabel("value of phi")
plt.title("bar chart for phi - k=4")
plt.show()

plt.bar(range(1,17), valuephi[4,])
plt.xlabel("differenDy")
plt.ylabel("value of phi")
plt.title("bar chart for phi - k=5")
plt.show()
#**************3-d

k=[3,5,8,10,12,15]
d3 = torch.zeros(6)
for i in range(6):
    rm_i = mixture_model(k[i], 100, 16, 64)
    trainsetX = X_tr[0:6620,]
    trainsetY = Y_tr[0:6620,]
    test4 = rm_i.fit(trainsetX, trainsetY, 0.01, 30)[0]
    miuum = rm_i.fit(trainsetX, trainsetY, 0.01, 30)[1]
    sigmaam = rm_i.fit(trainsetX, trainsetY, 0.01, 30)[2]
    print(i)
    phiim = rm_i.fit(trainsetX, trainsetY, 0.01, 30)[3]
    piim = rm_i.fit(trainsetX, trainsetY, 0.01, 30)[4]

    validsetX = X_tr[6620:8274,]
    validsetY = Y_tr[6620:8274,]

    Xnewm = validsetX[:, None, :]
    miuXm = Xnewm - miuum
    normxm = ((1 / (torch.sqrt(2 * torch.pi * ((sigmaam) ** 2)))) * (torch.exp((-((miuXm) ** 2) / (2 * ((sigmaam) ** 2))))))
    bbbm = torch.prod(normxm, 2)
    Ynewm = validsetY[:, None, :]
    berym = (((phiim) ** (Ynewm)) * ((1 - phiim) ** (1 - Ynewm)))
    aaam = torch.prod(berym, 2)

    bernorm = bbbm * aaam
    finalnm = bernorm @ (piim)
    dddm = finalnm.detach().numpy()
    cccm = bernorm.detach().numpy()
    sumfinalm = -torch.sum(torch.log(finalnm))
    d3[i]=sumfinalm
    print('answer:',d3[i])
print('the final answer is:',d3)


answertest23 = d3.detach().numpy()
plt.plot(k, answertest23, label='alpha = 0.01')
plt.ylim(-250000 -200000)
plt.xlabel('k')
plt.ylabel('NLML for validation-set')
plt.title('Question 3-d')
plt.legend()
plt.show()

#***************3-e
rt1 = mixture_model(15, 100, 16, 64)
test20 = rt1.fit(X_tr,Y_tr,0.01,30)[0]
test210 = rt1.fit(X_tr,Y_tr,0.01,30)[1]
test310 = rt1.fit(X_tr,Y_tr,0.01,30)[2]
test410 = rt1.fit(X_tr,Y_tr,0.01,30)[3]
test510 = rt1.fit(X_tr,Y_tr,0.01,30)[4]

#print('javab',test)
test311 = rt1.predict_proba_Y(X_tr)
#print(test3)
answere01 = (1.0/8274.0)*(1.0/16.0)*(torch.sum((test311*Y_tr)+((1-test311)*(1-Y_tr))))
print(answere01)
aa = torch.transpose((test311),0,1)

#***************3-f
test41 = rt1.predict_mean_X(Y_tr)
answere211 = (1.0/8274.0)*(1.0/100.0)*(torch.sum(torch.abs(test41 - X_tr)))
print('*****',answere211)

