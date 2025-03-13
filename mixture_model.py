import numpy as np
import torch
import torch.optim as optim


class mixture_model:

    def __init__(self, K, Dx, Dy, batchSize=64):
        self.K = K
        self.Dx = Dx
        self.Dy = Dy
        self.batchSize = batchSize

        sfm = torch.nn.Softmax(dim=1)
        sfp = torch.nn.Softplus(beta=1, threshold=20)

        miudz = torch.normal(0,0.01,(Dx,K),requires_grad=True)
        self.miudz = miudz

        sigmadzp = torch.zeros((Dx, K), requires_grad=True)
        self.sigmadzp = sigmadzp
        sigmadz = sfp(sigmadzp)
        self.sigmadz = sigmadz
        sigmadz.requires_grad_(True)

        phidzp  = torch.normal(0,0.01,(Dy,K), requires_grad=True)
        self.phidzp = phidzp
        phidz = torch.sigmoid(phidzp)
        self.phidz = phidz
        phidz.requires_grad_(True)

        pidzp = torch.zeros(K, requires_grad=True)
        self.pidzp = pidzp
        pidz = sfm(pidzp)
        self.pidz = pidz
        pidz.requires_grad_(True)

    def nlml(self, X, Y):

        self.miudzp.requires_grad_(True)
        self.sigmadzp.requires_grad_(True)
        self.phidzp.requires_grad_(True)
        self.pidzp.requires_grad_(True)

        self.miudz.requires_grad_(True)
        self.sigmadz.requires_grad_(True)
        self.phidz.requires_grad_(True)
        self.pidz.requires_grad_(True)


        N = X.detach().numpy()
        NN = Y.detach().numpy()
        Nn = np.shape(N)
        nn = np.shape(NN)
        piz = tensor.zeros(K, Nn[0])
        nor = tensor.ones(K)
        ber = tensor.ones(K)
        logsumnl = torch.zeros(Nn[0])
        for n in range(Nn[0]):

            for i in range(self.K):
                for j in range(Nn[1]):
                    pinor = (1 / ((torch.sqrt(2 * torch.pi * ((self.sigmadz[j, i]) ^ 2))))) * (
                        torch.exp((-1 // 2 * ((self.sigmadz[j, i]) ^ 2))((X[n, j] - self.miudz[j, i]) ^ 2)))
                    pinorm = pinor * nor[i]
                    nor[i] = pinorm
                print("meghdar normal:", nor[i])
                for a in range(nn[1]):
                    piber = (((self.phidz[a, i]) ^ (Y[n, a])) * ((1 - self.phidz[a, i]) ^ (1 - Y[n, a])))
                    pibern = piber * ber[i]
                    ber[i] = pibern
                print("meghdar bernoli:", ber[i])
                piz[i, n] = self.pidz[i] * ber[i] * nor[i]
            sumpiz[n] = torch.sum(piz, 0)
            logsumnl[n] = torch.log(sumpiz[n])
        finalnlml = torch.sum(logsumnl)
        return finalnlml

    def fit(self, X, Y, alpha, epochs):
        alpha = torch.tensor(alpha)
        epochs = torch.tensor(epochs)

        self.miudzp.requires_grad_(True)
        self.sigmadzp.requires_grad_(True)
        self.phidzp.requires_grad_(True)
        self.pidzp.requires_grad_(True)

        self.miudz.requires_grad_(True)
        self.sigmadz.requires_grad_(True)
        self.pidz.requires_grad_(True)
        self.phidz.requires_grad_(True)
        N = X.detach().numpy()
        NN = Y.detach().numpy()
        Nn = np.shape(N)
        nn = np.shape(NN)
        SizeX = (Nn[0]//self.batchSize)
        optimizer = optim.Adam([self.miudz, self.sigmadz, self.pidz, self.phidz], lr=0.01)
        output1 = torch.zeros(SizeX)
        output2 = torch.zeros(self.K)

        for z in range(epochs):
            self.miudz.requires_grad_(True)
            self.sigmadz.requires_grad_(True)
            self.pidz.requires_grad_(True)
            self.phidz.requires_grad_(True)
            batchingX = torch.utils.data.DataLoader(X, batch_size=self.batchSize, shuffle=True, drop_last=True)
            batchingY = torch.utils.data.DataLoader(Y, batch_size=self.batchSize, shuffle=True, drop_last=True)
            XTrain = list(batchingX)
            YTrain = list(batchingY)
            rp = torch.randperm(Nn[0])

            for s in range(SizeX):
                optimizer.zero_grad()
                loss = self.nlml(XTrain[s],YTrain[s])
                loss.backward()
                optimizer.step()
                output1[s] = self.nlml(XTrain[s],YTrain[s]).detach().numpy()
                #print('javabe output1:',output1[s,i])
            print('javabe output1 KOLLI:',output1)
            output2[z] = output1[SizeX]
            print('check javabe output2:',output2[z])
        print('j a v output2***:',output2)
        return output2







    def predict_proba_Y(self, X):
        pass
        
    def predict_mean_X(self, Y):
        pass

        

