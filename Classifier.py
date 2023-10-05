import pandas as pd
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os

import torch
from torch.nn import Module, Conv2d, ReLU, ELU, MaxPool2d, Flatten, Linear, Softmax, BatchNorm2d, BatchNorm1d, Dropout
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

platform='Youtube'

"""
    Data structure for this file is
    - Youtube
        - actual
            -vid1
                -vid1_1.csv
                -vid1_2.csv
                -
            -vid2
            -
        - synth
            -vid1
                -1.csv
                -2.csv
                -
            -vid2
            -
"""
## ++++++ prepare indices for the synthesized and actual dataset ++++++
## actual data
def get_paths_actual_data(path_actual, num_of_vid, num_of_act_trace):
    all_paths = []
    all_gts = []
    for v in range(num_of_vid):
        for act in range(num_of_act_trace):
            all_paths.append(path_actual +'/vid'+str(v+1) +'/vid' + str(v + 1) + '_' + str(act + 1) + '.csv')
            all_gts.append(v)

    return all_paths, all_gts

## synth data
def get_paths_synth_data(path_synth, num_of_vid, num_of_synth):
    all_paths = []
    all_gts = []
    for v in range(num_of_vid):
        for synth in range(num_of_synth):
            all_paths.append(path_synth + '/vid' + str(v + 1) +'/' + str(synth + 1) + '.csv')
            all_gts.append(v)

    return all_paths, all_gts

## test data
def get_paths_test_data(path_actual, num_of_vid, test_path_begin, test_path_end):
    all_paths = []
    all_gts = []
    for v in range(num_of_vid):
        for act in range(test_path_begin, test_path_end):
            all_paths.append(path_actual+'/vid'+str(v+1)  + '/vid' + str(v + 1) + '_' + str(act + 1) + '.csv')
            all_gts.append(v)

    return all_paths, all_gts

# --------------------------------------------------------------------

## ++++++++++ Create class object for the Dataset +++++++++++++++

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, paths, gts, classes):
        self.paths = paths
        self.gts = gts # ground truth labels
        self.classes = classes

    # number of rows in the dataset
    def __len__(self):
        return len(self.paths)

    # get a row at an index
    def __getitem__(self, idx):
        # read the .csv files
        X= pd.read_csv(self.paths[idx], header=None).values
        X = torch.from_numpy(X)
        X = X.to(device)
        X = X.reshape([3, X.shape[0], X.shape[1]])
        X=X/255.0
        X = X.float()

        gt = torch.zeros(self.classes, device=device)
        gt[self.gts[idx]] = 1

        return [X, gt]

## ----------------------------------------------------------------

## +++++++ define ML model +++++++++++++


# forward propagate input

class DNN_model(Module):
    # define model elements
    def __init__(self, num_channels, classes):
        super(DNN_model, self).__init__()

        self.conv1 = Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm1 = BatchNorm2d(num_features=32)
        self.relu1 = ELU(alpha=1.0)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm2 = BatchNorm2d(num_features=32)
        self.relu2 = ELU(alpha=1.0)
        self.max_pool1 = MaxPool2d(kernel_size=(2, 2))
        self.dropout1= Dropout(p=0.1)

        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm3 = BatchNorm2d(num_features=64)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm4 = BatchNorm2d(num_features=64)
        self.relu4 = ReLU()
        self.max_pool2 = MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = Dropout(p=0.1)

        self.conv5 = Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm5 = BatchNorm2d(num_features=128)
        self.relu5 = ReLU()
        self.conv6 = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm6 = BatchNorm2d(num_features=128)
        self.relu6 = ReLU()
        self.max_pool3 = MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = Dropout(p=0.1)

        self.conv7 = Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm7 = BatchNorm2d(num_features=256)
        self.relu7 = ReLU()
        self.conv8 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm8 = BatchNorm2d(num_features=256)
        self.relu8 = ReLU()
        self.max_pool4 = MaxPool2d(kernel_size=(2, 2))
        self.dropout4 = Dropout(p=0.1)

        self.flatten1 = Flatten()
        # select the in_features appropriately
        self.linear1 = Linear(in_features=16384, out_features=128)
        self.batchnm1_1D = BatchNorm1d(num_features=128)
        self.relu9 = ReLU()
        self.dropout5 = Dropout(p=0.7)
        self.linear2 = Linear(in_features=128, out_features=64)
        self.batchnm2_1D = BatchNorm1d(num_features=64)
        self.relu10 = ReLU()
        self.dropout6 = Dropout(p=0.5)
        self.linear3 = Linear(in_features=64, out_features=classes)
        self.softmax = Softmax(dim=-1)

    # forward propagate input
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnm2(x)
        x = self.relu2(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.batchnm3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batchnm4(x)
        x = self.relu4(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.batchnm5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.batchnm6(x)
        x = self.relu6(x)
        x = self.max_pool3(x)
        x = self.dropout3(x)

        x = self.conv7(x)
        x = self.batchnm7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.batchnm8(x)
        x = self.relu8(x)
        x = self.max_pool4(x)
        x = self.dropout4(x)

        x = self.flatten1(x)

        x = self.linear1(x)
        x = self.batchnm1_1D(x)
        x = self.relu9(x)
        x = self.dropout5(x)
        x = self.linear2(x)
        x = self.batchnm2_1D(x)
        x = self.relu10(x)
        x = self.dropout6(x)
        x = self.linear3(x)
        #x = self.softmax(x)
        return x
## ------------------------------------

def main():
    num_of_vid = 20 # number of classes for classification
    num_of_act_trace = 80 # number of original traces to be used (here first 80 in each class folder will be used)
    max_num_of_synth_traces = 400 # total number of synthesized images for each class
    synth_group_size = 100

    test_path_begin = 80 # test data is the 81st-100th data in every class folder
    test_path_end =100

    main_path_actual = 'Youtube/actual' # path to original data
    main_path_synth = 'Youtube/synth'  # path to synthesized data

    # creating actual train paths and test paths only once
    actual_paths, actual_gts = get_paths_actual_data(main_path_actual, num_of_vid, num_of_act_trace)
    test_paths, test_gts = get_paths_test_data(main_path_actual, num_of_vid, test_path_begin, test_path_end)

    # creating test data loaded only once
    dataset_test = CSVDataset(test_paths, test_gts, num_of_vid)
    test_dl = DataLoader(dataset_test, batch_size=32, shuffle=True)
    all_synth_acc=[]

    for num_of_synth in range(0, max_num_of_synth_traces+1, synth_group_size):
        print('Num of synthesized data ' + str(num_of_synth))

        if num_of_synth > 0:
            synth_path, synth_gt = get_paths_synth_data(main_path_synth, num_of_vid, num_of_synth)

            # combine actual and synth paths ant gt data
            # uncomment the below three lines if you want to mis the original data with synth data
            #train_paths = np.concatenate([np.asarray(actual_paths), np.asarray(synth_path)], axis=0)
            #train_gts = np.concatenate([np.asarray(actual_gts), np.asarray(synth_gt)], axis=0)
            #train_paths, train_gts = shuffle(train_paths, train_gts, random_state=123)

            train_paths = np.asarray(synth_path)
            train_gts = np.asarray(synth_gt)
        else:
            train_paths = np.asarray(actual_paths)
            train_gts = np.asarray(actual_gts)
            # train_paths, train_gts = shuffle(train_paths, train_gts, random_state=123)

        dataset_train = CSVDataset(train_paths, train_gts, num_of_vid)
        train_dl = DataLoader(dataset_train, batch_size=32, shuffle=True)

        model = DNN_model(num_channels=3, classes=num_of_vid)
        model.to(device)

        ## +++++++++ train the model +++++++++++
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        if num_of_synth==0:
            epochs=100
        else:
            epochs=140
        for epoch in range(epochs):
            # enumerate mini batches
            correct = 0
            total = 0
            for i, (x, y) in enumerate(train_dl):
                # (x, y) = (x.to(device), y.to(device))
                optimizer.zero_grad()
                # compute the model output
                yhat = model(x)
                # calculate loss
                loss = criterion(yhat, y)
                # calculate the accuracy
                _, predicted = torch.max(yhat.data, 1)
                _, gt = torch.max(y.data, 1)
                total += y.size(0)
                correct += (predicted == gt).sum().item()
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

        ## +++++++ test the model ++++++++++++

        predict_gt = []
        correct = 0
        total = 0
        gt_all = []
        with torch.no_grad():
            for i, (x, y) in enumerate(test_dl):
                (x, y) = (x.to(device), y.to(device))
                yhat = model(x)
                _, predicted = torch.max(yhat.data, 1)
                for i in list(np.asarray(predicted.cpu())):
                    predict_gt.append(i)
                _, gt = torch.max(y.data, 1)
                for j in list(np.asarray(gt.cpu())):
                    gt_all.append(j)
                total += y.size(0)
                correct += (predicted == gt).sum().item()
        all_synth_acc.append(100 * correct / total)
        conf_mat = confusion_matrix(gt_all, predict_gt)
        print(conf_mat)
        data = pd.DataFrame(data=conf_mat)
        path_to_save =  'results/'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        data.to_csv(path_to_save + '/Synthesized_size_' + str(num_of_synth) + '.csv', index=False, header=False)
        print(f'Final Accuracy : {100 * correct / total} %')
    columns = list(np.arange(0, max_num_of_synth_traces+1, synth_group_size).astype(str))
    data2 = np.asarray(all_synth_acc).reshape([1, -1])
    df_acc = pd.DataFrame(columns=columns, data=data2)
    df_acc.to_csv(path_to_save + '/acc-df'+ '.csv', index=False)
    return

if __name__ == main():
    main()