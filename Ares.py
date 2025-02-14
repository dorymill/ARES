import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import statistics
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import r2_score

class Ares():

    # Constructor
    def __init__(self):

        # Main Model
        self.regressor = None

        # Num Features
        self.features = 11

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

        print('\n[+] ARES-1 initialized.')

    # Load trainining data
    def load_training_sets(self, dataSets):

        loaded = 0

        # Acquire all the datasets
        for set in dataSets:
            try:
                df = pd.read_csv(set, usecols=["%psbl","StagePoints","HF","Time","A","B","C","D","M","NPM","NS","Proc"])
                self.trainingDataFrames.append(df)
                print(f'\n[+] Loaded dataset: {set}')
                loaded += 1
            except:
                print(f'\n[-] Failed to load dataset: {set}')

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

        print(f'\n[+] Data aggregated ({len(self.Y_master)} points).')

    def normalize (self):

        # Before we rescale, we split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_master, 
                                       self.Y_master, 
                                       test_size=0.25, 
                                       random_state=0)

        self.X_train = self.x_scaler.fit_transform(self.X_train)
        self.Y_train = self.y_scaler.fit_transform(self.Y_train)

        print(f'\n[+] Data normalized.')

    def train (self, trees):

        self.regressor = RandomForestRegressor(n_estimators=trees, random_state=0)

        print(f'\n[+] Launching training simulation . . .')
        start = time.time()
        
        # Train the model
        self.regressor.fit(self.X_train, self.Y_train.ravel())

        # Evaluate performance
        # Transform test input before prediction and output after
        Y_prediction = self.y_scaler.inverse_transform(np.reshape(self.regressor.predict(self.x_scaler.transform(self.X_test)),(-1,1)))
        np.set_printoptions(precision=2)

        score = r2_score(self.Y_test, Y_prediction, multioutput='variance_weighted')
        stop = time.time()

        errors = []

        g10Count = 0

        for y_true, y_pred in zip(self.Y_test, Y_prediction):
            eVal = y_pred - y_true
            if eVal >= 1.:
                #print(f'\n[+] True Rank: {y_true}\tPredicted Rank: {y_pred}\tError: {y_pred - y_true}\n')
                g10Count += 1

            errors.append(float (eVal))

        print(f'\n[+] Training complete ({stop-start:.3f}s)')

        print(f'\n[+] Coefficient of Determination: {score}')

        print(f'\n[+] Max Error: {max(errors)}\n[+] Average Error: {statistics.mean(errors)}\n[+] Minimum Error: {min(errors)}\n[-] Errors over 1%: {g10Count} ({g10Count / len(errors):.2f}% of test set)')

        # display = PredictionErrorDisplay(y_true=self.Y_test, y_pred=Y_prediction)
        # display.plot()



if __name__ == "__main__":

    ares = Ares()

    # Get all training CSV Names
    training_path = "./TrainingSets/"

    all_items = os.listdir(training_path)

    # Filter out only the files with '.csv' extension
    trainingCsvs = [(training_path + item) for item in all_items if os.path.isfile(os.path.join(training_path, item)) and item.endswith('.csv')]

    # Load Ares' training data
    ares.load_training_sets(trainingCsvs)

    ares.train(100)
