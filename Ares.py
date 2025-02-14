import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import os

from sklearn.preprocessing import StandardScaler

class Ares():

    # Constructor
    def __init__(self):

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
        self.sc = StandardScaler()

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

        print('[+] Aggregating and normalizing training data...')

        self.aggregate()

        self.normalize()

    # Aggregate data into two structures
    def aggregate (self):

        # This is gross....fix this
        self.X_master = self.trainingDataFrames[0].iloc[:,1:].values
        self.Y_master = self.trainingDataFrames[0].iloc[:,0].values[:,None]

        for idx in range(1, len(self.trainingDataFrames)):

            # The first column has the y data
            x = self.trainingDataFrames[idx].iloc[:,1:].values
            y = self.trainingDataFrames[idx].iloc[:,0].values[:,None]

        # Append results
        self.X_master = np.vstack([self.X_master, x])
        self.Y_master = np.concatenate([self.Y_master,y])

    def normalize (self):
        pass


if __name__ == "__main__":

    ares = Ares()

    # Get all training CSV Names
    training_path = "./TrainingSets/"

    all_items = os.listdir(training_path)

    # Filter out only the files with '.csv' extension
    trainingCsvs = [(training_path + item) for item in all_items if os.path.isfile(os.path.join(training_path, item)) and item.endswith('.csv')]

    # Load Ares' training data
    ares.load_training_sets(trainingCsvs)
