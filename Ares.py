import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt

import sklearn as sk
import statistics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import r2_score


class Ares():

    # Constructor
    def __init__(self):

        # Main Model
        self.regressor = None

        self.regressorName = ""

        # Num Features
        self.features = 11

        self.testSplitFrac = 0.25

        self.VERSION = 1

        # ANN Verbosity (1 is on, 0 is off)
        self.annVerbosity = 1

        # Data Lists
        self.trainingDataFrames   = []
        self.simulationDataFrames = []

        # Master training frames
        self.X_master = []
        self.Y_master = []

        self.X_train = []
        self.Y_train = []

        self.X_test = []
        self.Y_test = []

        # Data scaler
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        os.system('cls')

        print('\n[+] ARES-1 initialized.\n')

    # Load trainining data
    def load_training_sets(self, dataSets):

        loaded = 0

        # Acquire all the datasets
        for set in dataSets:
            try:
                df = pd.read_csv(set, usecols=["%psbl","StagePoints","HF","Time","A","B","C","D","M","NPM","NS","Proc"])
                self.trainingDataFrames.append(df)
                print(f'[+] Loaded dataset: {set}')
                loaded += 1
            except:
                print(f'[-] Failed to load dataset: {set}')

        print('\n[+] Aggregating and normalizing training data...')

        self.aggregate()

        self.normalize()

    # Aggregate data into two structures
    def aggregate (self):


        for idx in range(0, len(self.trainingDataFrames)):

            # This is gross....fix this
            if idx == 0:
                self.X_master = self.trainingDataFrames[0].iloc[:,1:].values
                self.Y_master = self.trainingDataFrames[0].iloc[:,0].values[:,None]

            # The first column has the y data
            x = self.trainingDataFrames[idx].iloc[:,1:].values
            y = self.trainingDataFrames[idx].iloc[:,0].values[:,None]

            # Append results
            self.X_master = np.vstack([self.X_master, x])
            self.Y_master = np.concatenate([self.Y_master,y])

        # Reshape Y_master
        self.Y_master = np.reshape(self.Y_master, (-1,1))

        print(f'[+] Aggregated {len(self.Y_master)} shooters across {len(self.trainingDataFrames)} matches.')

    # Normalize the data
    def normalize (self):

        # Before we rescale, we split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_master, 
                                       self.Y_master, 
                                       test_size=self.testSplitFrac, 
                                       random_state=0)

        self.X_train = self.x_scaler.fit_transform(self.X_train)
        self.Y_train = self.y_scaler.fit_transform(self.Y_train)

        print(f'[+] Data normalized and split (N_train = {len(self.Y_train)}).')

    # Model Training
    def train(self, regressor, *args):

        # Model Selection
        if (regressor == 'support_vector'):
            # args[0] = Kernel Function

            self.regressor = SVR(kernel=args[0])
            self.regressorName = "Support Vector Regressor"
            print(f'\n[+] Launching {args[0]} kernel Support Vector simulation. . .')

        if (regressor == 'random_forest'):
            # args[0] = N Trees

            self.regressor = RandomForestRegressor(n_estimators=args[0], random_state=0)
            self.regressorName = "Random Forest Regressor"
            print(f'\n[+] Launching Random Forest simulation with {args[0]} trees. . .')

        if (regressor == 'ann'):
            # args[0] = Hidden Layer Activation Function
            # args[1] = Number of Hidden layers
            # args[2] = Number of Neurons in Hidden Layers
            # args[3] = Output Activation Function
            # args[4] = Optimizer
            # args[5] = Loss Function
            # args[6] = Performance Metric
            # args[7] = Batch Size
            # args[8] = Epochs

            self.regressor = tf.keras.models.Sequential()
            self.regressorName = "ARES-I (ANN)"
            print(f'\n[+] Launching Aritificial Neural Network (ARES-{self.VERSION}). . .')

            # Add the first layer
            self.regressor.add(tf.keras.layers.Dense(units=self.features, activation=args[0]))

            # Iteratively add the hidden layers
            for idx in range(0,args[1]):
                self.regressor.add(tf.keras.layers.Dense(units=args[2], activation=args[0]))

            # Output Layer
            self.regressor.add(tf.keras.layers.Dense(units=1, activation=args[3]))

            # Compile the model
            self.regressor.compile(optimizer = args[4], loss = args[5], metrics = [args[6]])

        

        # Train the model
        start = time.time()
        if regressor == 'ann':
            self.regressor.fit(self.X_train, self.Y_train, batch_size=args[7], epochs=args[8], verbose=self.annVerbosity)
        else:
            self.regressor.fit(self.X_train, self.Y_train.ravel())

        # Evaluate performance
        # Transform test input before prediction and output after
        Y_prediction = self.y_scaler.inverse_transform(np.reshape(self.regressor.predict(self.x_scaler.transform(self.X_test)),(-1,1)))
        np.set_printoptions(precision=2)

        score = r2_score(self.Y_test, Y_prediction, multioutput='variance_weighted')
        stop = time.time()

        errors = []

        g10Count = 0
        
        # Index to inspect large errors
        idx = 0

        for y_true, y_pred in zip(self.Y_test, Y_prediction):
            eVal = abs(y_pred - y_true).item()
            if eVal >= 5.:
                #print(f'\n[+] True Rank: {y_true}\tPredicted Rank: {y_pred}\tError: {y_pred - y_true}\n')
                # print(f'\n[-] Large Error Detected on input: {self.X_test[idx,:]}')
                g10Count += 1

            errors.append(float (eVal))
            idx += 1

        print(f'[+] Training complete ({stop-start:.3f}s)')

        print(f'[+] Coefficient of Determination: {score:.4f}')

        print(f'[+] Max Error: {max(errors):.2f}%\n[+] Average Error: {statistics.mean(errors):.2f}%\n[+] Error Std. Dev: {statistics.stdev(errors):.2f}%\n[-] Errors over 5%: {g10Count} ({(g10Count / len(errors))*100:.2f}% of test set)')        

        # Plot Histogram of Error
        q75, q25 = np.percentile(errors, [75, 25])
        iqr = q75 - q25
        N = len(errors)
        binWidth = 2*(iqr) / (N)**(0.3333333)
        bins = 2*(np.max(errors) - np.min(errors)) / binWidth

        plt.hist(errors, bins=int(bins))
        plt.xlabel('Absolute Rank Error (%)')
        plt.ylabel('Counts')
        plt.title(f'{regressor} Error Distribution (µ = {statistics.mean(errors):.2f}%, σ = {statistics.stdev(errors):.2f}% , N = {len(errors)})')
        plt.show()

        return score

if __name__ == "__main__":

    ares = Ares()

    # Get all training CSV Names
    training_path = "./TrainingSets/"

    all_items = os.listdir(training_path)

    # Filter out only the files with '.csv' extension
    trainingCsvs = [(training_path + item) for item in all_items if os.path.isfile(os.path.join(training_path, item)) and item.endswith('.csv')]

    # Load Ares' training data
    ares.load_training_sets(trainingCsvs)

    ares.train('random_forest', 500)

    ares.train('support_vector', 'rbf')

    # To-Do: Add more complex hidden layer structure, e.g. growing and shrinking layer counts

    hiddenLayers = 300
    hlNeurons    = 30
    epochs       = 500
    batchSize    = 500
    
    hiddenLayerActFnct = 'relu'
    optimizer          = 'adam'
    outputActFnct      = 'sigmoid'
    lossFnct           = 'mse'
    performanceMetric  = 'accuracy'

    ares.annVerbosity = 0

    ares.train('ann', hiddenLayerActFnct, 
                      hiddenLayers, hlNeurons, 
                      outputActFnct, optimizer, 
                      lossFnct, performanceMetric,
                      batchSize, epochs)
