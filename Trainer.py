import torch
from dataset.dataLoader import DL
from network import Model
import torch.nn as nn
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
# from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.model_selection import train_test_split
import scipy.io as io
from tqdm import tqdm
from TorchPCA import PCA


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._init_data()
        self._init_model()

    def _init_model(self):
        self.net = Model(self.args).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), )
        self.svm = SVC(kernel='rbf')
        self.pca = PCA([])
        # self.pca = PCA(n_components=self.args.K)
        # self.pca = manifold.TSNE(n_components=self.args.K, init='pca')

    def _init_data(self):
        self.data = DL(self.args)
        self.dl = self.data.dl

    def feature_extract(self):
        outputs = []
        labels = []
        print("perform feature extraction...")
        for inputs, targets in tqdm(self.dl, ncols=90):
            inputs = inputs.to(self.device)
            targets = targets.numpy()
            output = self.net(inputs,self.args).detach().cpu().numpy()
            outputs.append(output)
            labels.append(targets)

        X = np.concatenate(outputs, axis=0)
        y = np.concatenate(labels, axis=0)

        data = {'X': X, 'y': y}
        io.savemat('results/%s.mat' % self.args.dataset, data)

    def train(self):
        print("Dataset: ", self.args.dataset)
        print("train ratio: ", self.args.ratio)

        print("read dataset...")
        data = io.loadmat('results/%s.mat' % self.args.dataset)
        X, y = data['X'], data['y'].squeeze()
        print("pca dimension reduction...")
        X = self.pca.Decomposition(X,self.args.K)
        print("Divide the dataset...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.args.ratio)
        self.svm.fit(X_train, y_train)
        pred = self.svm.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print('val_acc: %.6f' % acc)
