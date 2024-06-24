import torch
import torch.nn as nn
import numpy as np

def inverse_transform(value, scale_param_1, scale_param_2, scaler_type):
    if scaler_type == 'StandardScaler':
        inverse_value = value * scale_param_2 + scale_param_1
    elif scaler_type == 'MinMaxScaler':
        inverse_value = value * (scale_param_2 - scale_param_1) + scale_param_1
    return inverse_value

def fit_quality(y_true, y_pred):
    Sum_err = np.sum((y_true-y_pred)**2)
    N = len(y_true)
    numerator = np.sqrt(Sum_err/N)
    donominator = np.max(y_true)
    fit_quality = 100 * numerator/donominator
    return fit_quality

def Cal_Composition(V_i):
    f_H, f_C, f_L = 0.795, 0.949, 0.626
    denominator =(V_i[:, 0]/f_H + V_i[:, 1]/f_C + V_i[:, 2]/f_L)
    C_i = np.zeros_like(V_i)
    C_i[:, 0] = V_i[:, 0]/f_H/denominator
    C_i[:, 1] = V_i[:, 1]/f_C/denominator
    C_i[:, 2] = V_i[:, 2]/f_L/denominator
    return C_i * 100

class DenoiseAutoEncoder_DTG(nn.Module):
    def __init__(self, no_of_filter):
        super(DenoiseAutoEncoder_DTG, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=no_of_filter, kernel_size=8, stride=3, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.Conv1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.Conv1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.Conv1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=6250, out_features=50),
            nn.LeakyReLU()  # LeakyReLU 추가
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(in_features=50, out_features=6250),
            nn.ReLU(),
            nn.Unflatten(1, (no_of_filter, 125)),  
            nn.ConvTranspose1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=7, stride=3, padding=1),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=no_of_filter, out_channels=1, kernel_size=8, stride=3, padding=1),
        )
                
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class DenoiseAutoEncoder_DDTG(nn.Module):
    def __init__(self, no_of_filter):
        super(DenoiseAutoEncoder_DDTG, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=no_of_filter, kernel_size=8, stride=3, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.Conv1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.Conv1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.Conv1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=6250, out_features=50),
            nn.LeakyReLU()  # LeakyReLU 추가
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(in_features=50, out_features=6250),
            nn.ReLU(),
            nn.Unflatten(1, (no_of_filter, 125)),  
            nn.ConvTranspose1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=no_of_filter, out_channels=no_of_filter, kernel_size=7, stride=3, padding=1),
            nn.BatchNorm1d(no_of_filter),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=no_of_filter, out_channels=1, kernel_size=8, stride=3, padding=1),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class CNNModel(nn.Module):
    def __init__(self, NoOfFilter, dropout_prob):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=NoOfFilter, kernel_size=8, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(NoOfFilter)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=NoOfFilter, out_channels=NoOfFilter, kernel_size=5, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm1d(NoOfFilter)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=NoOfFilter, out_channels=NoOfFilter, kernel_size=5, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm1d(NoOfFilter)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=3)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(390, 200)  
        self.relu3 = nn.ReLU()
        self.dense2 = nn.Linear(200, 20)
        self.relu4 = nn.ReLU()
        self.output = nn.Linear(20, 9)
        self.dropout = nn.Dropout(p=dropout_prob)  # 드롭아웃 레이어 추가
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.dropout(x)  # 첫 번째 드롭아웃 적용
        x = self.dense2(x)
        x = self.relu4(x)
        x = self.dropout(x)  # 두 번째 드롭아웃 적용
        x = self.output(x)
        return x