import datetime
import time
import numpy as np
import pandas as pd
import matplotlib
import sklearn as sk
import statistics
import os

matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from scipy.stats import loguniform

from tensorboard.plugins.hparams import api as hp

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import r2_score

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV


class Ares():

    # Constructor
    def __init__(self):

        # Main Model
        self.regressor = None

        self.regressorName = ""
        self.regressorNameShort = ""

        # Num Features
        self.features = 12

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
                df = pd.read_csv(set, usecols=["%psbl","Points","HF","Time","A","B","C","D","M","NPM","NS","Proc"])
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

    # Set the resgression model
    def set_regressor (self, regressor, *args):

        # Model Selection
        self.regressorNameShort = regressor

        if (regressor == 'support_vector'):
            # args[0] = Kernel Function

            self.regressor = SVR(kernel=args[0])
            self.regressorName = "Support Vector Regressor"

        elif (regressor == 'random_forest'):
            # args[0] = N Trees

            self.regressor = RandomForestRegressor(n_estimators=args[0], random_state=0)
            self.regressorName = "Random Forest Regressor"

        elif (regressor == 'ann'):
            # args[0] = Hidden Layer Activation Function
            # args[1] = Number of Hidden layers
            # args[2] = Number of Neurons in Hidden Layers
            # args[3] = Output Activation Function
            # args[4] = Optimizer
            # args[5] = Loss Function
            # args[6] = Performance Metric
            # args[7] = Batch Size
            # args[8] = Epochs

            self.hiddenLayerActFnct = args[0]
            self.hiddenLayers       = args[1]
            self.neurons            = args[2]
            self.outputActFnct      = args[3]
            self.optimizer          = args[4]
            self.lossFnct           = args[5]
            self.performanceMetric  = args[6]

            self.regressor = tf.keras.models.Sequential()
            self.regressorName = "ARES-I (ANN)"
            
            # Add the first layer
            self.regressor.add(tf.keras.layers.Dense(units=self.features, activation='relu'))

            # Iteratively add the hidden layers
            for idx in range(0,self.hiddenLayers):
                self.regressor.add(tf.keras.layers.Dense(units=self.neurons, activation=self.hiddenLayerActFnct))

            # Output Layer
            self.regressor.add(tf.keras.layers.Dense(units=1, activation=self.outputActFnct))

            # Compile the model
            self.regressor.compile(optimizer = self.optimizer, loss = self.lossFnct, metrics = [self.performanceMetric])
        
        else:
            print(f'\n[-] Invalid regressor passed.')
            print(f'[-] Valid regressors: \'support_vector\', \'random_forest\', \'ann\'')
            exit()

        print(f'\n[+] Regressor set to {self.regressorName}')

    # Optimize model hyperparameters
    def hyperparam_optimize(self, searchMethod, paramsDict):

        hyperSearcher = ''

        match searchMethod:

            case 'GridSearchCV':
                hyperSearcher = GridSearchCV(self.regressor,
                                             paramsDict)
                 
            case 'RandSearchCV':
                hyperSearcher = RandomizedSearchCV(self.regressor,
                                                   paramsDict)
                     
            case 'HalvingGridSearchCV':
                hyperSearcher = HalvingGridSearchCV(self.regressor,
                                                    paramsDict)
                

            case 'HalvingRandSearchCV':
                hyperSearcher = HalvingRandomSearchCV(self.regressor,
                                                      paramsDict)
                
            case _:
                print(f'\n[-] Invalid search method passed to hyperparam optimizer.')
                print(f'\n[-] Valid options: \'GridSearchCV\', \'RandSearchCV\', \'HalvingGridSearchCV\', \'HalvingRandSearchCV\'')
                exit()

        # Kick off the search with the TensorBoard callback
        print(f'[+] Searching for best hyperparameters. . .')
        hyperSearcher.fit(self.X_train, self.Y_train.ravel())

        print(f'[+] Hyperparameter search complete for {self.regressorNameShort} using {searchMethod}.')
        print(f'[+] Best Parameters: {hyperSearcher.best_params_}')
        print(f'[+] Accuracy: {hyperSearcher.score(self.X_train, self.Y_train.ravel()):.3f}')

        return hyperSearcher.best_params_

    # Optimize ANN hyperparameters
    def ann_hyperparam_optimize(self, hparams):
        
        HP_HDN = hp.HParam('hiddenLayers', hp.Discrete(hparams['hiddenLayers']))
        HP_NRN = hp.HParam('neurons', hp.Discrete(hparams['neurons']))
        HP_ACT = hp.HParam('activation', hp.Discrete(hparams['activation']))
        HP_OPT = hp.HParam('optimizer', hp.Discrete(hparams['optimizer']))
        HP_LSS = hp.HParam('loss', hp.Discrete(hparams['loss']))
        HP_MET = hp.HParam('metrics', hp.Discrete(hparams['metrics']))
        HP_BTC = hp.HParam('batch', hp.Discrete(hparams['batch']))
        HP_EPC = hp.HParam('epochs', hp.Discrete(hparams['epochs']))

        with tf.summary.create_file_writer('logs/').as_default():
            hp.hparams_config(
                hparams=[HP_HDN, HP_NRN, HP_ACT, HP_OPT, HP_LSS, HP_MET, HP_BTC, HP_EPC],
                metrics=[hp.Metric('mse', display_name='Accuracy')],
            )

        def train_model(hparams):
        
            # Instantiate model
            model = tf.keras.models.Sequential()

            # First Layer
            model.add(tf.keras.layers.Dense(units=self.features, activation='relu'))

            # Hidden Layers
            for i in range(0, hparams[HP_HDN]):
                model.add(tf.keras.layers.Dense(units=hparams[HP_NRN], activation=hparams[HP_ACT]))

            # Output layer
            model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
            # Compile with loss function and optimizer
            model.compile(optimizer=hparams[HP_OPT], loss=hparams[HP_LSS], metrics=[hparams[HP_MET]])
        
            # Fit the data
            model.fit(self.X_train, self.Y_train, batch_size=hparams[HP_BTC], epochs=hparams[HP_EPC], verbose=self.annVerbosity)
        
            Y_prediction = self.y_scaler.inverse_transform(np.reshape(model.predict(self.x_scaler.transform(self.X_test)),(-1,1)))

            score = r2_score(self.Y_test, Y_prediction, multioutput='variance_weighted')

            return score
        
        def run(dir, hparams):
            with tf.summary.create_file_writer(dir).as_default():
                hp.hparams(hparams)
                accuracy = train_model(hparams)
                tf.summary.scalar('accuracy', accuracy, step=1)
        
        # Iterate over all hyperparameters
        print(f'[+] Searching for best hyperparameters. . .')
        run_num = 0
        for hLayers in hparams['hiddenLayers']:
            for neurons in hparams['neurons']:
                for activation in hparams['activation']:
                    for optimizer in hparams['optimizer']:
                        for loss in hparams['loss']:
                            for metric in hparams['metrics']:
                                for batch in hparams['batch']:
                                    for epochs in hparams['epochs']:
                                        run_name = f'run_{run_num}'

                                        params = {
                                            HP_HDN : hLayers,
                                            HP_NRN : neurons,
                                            HP_ACT : activation,
                                            HP_OPT : optimizer,
                                            HP_LSS : loss,
                                            HP_MET : metric,
                                            HP_BTC : batch,
                                            HP_EPC : epochs
                                        }

                                        run('logs/' + run_name, params)
                                        run_num += 1
        
    # Model Training
    def train(self, *args):

        # Train the model
        start = time.time()
        if self.regressorNameShort == 'ann':
            print(f'\n[+] Training Aritificial Neural Network (ARES-{self.VERSION}). . .')
            self.regressor.fit(self.X_train, self.Y_train.ravel(), batch_size=args[0], epochs=args[1], verbose=self.annVerbosity)
        else:
            self.regressor.fit(self.X_train, self.Y_train.ravel())
            print(f'\n[+] Training {self.regressorNameShort}. . .')

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
        plt.title(f'{self.regressorName} Error Distribution (µ = {statistics.mean(errors):.2f}%, σ = {statistics.stdev(errors):.2f}% , N = {len(errors)})')
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

    # Random Forest
    # ares.set_regressor('random_forest', 1000)
    # best_params = ares.hyperparam_optimize('HalvingRandSearchCV',
    #                           {
    #                              'n_estimators'   : [10, 50, 100, 500, 1000],
    #                              'max_features'   : ['sqrt', 'log2', None ],
    #                              'max_depth'      : [None, 1, 12, 24, 50],
    #                              'max_leaf_nodes' : [None, 2, 10, 12, 24, 50],
    #                           })
    # # ares.train()

    # Support Vector
    # ares.set_regressor('support_vector', 'rbf')
    # best_params = ares.hyperparam_optimize('HalvingRandSearchCV',
    #                         {
    #                             'C'      : [1, 10, 100, 1000],            #loguniform(1e0, 1e3)
    #                             'gamma'  : [1, 0.1, 0.01, 0.001, 0.0001], #loguniform(1e0, 1e-4),
    #                             'kernel' : ['rbf', 'linear'],
    #                         })
    # ares.train()



    # Artificial Neural Network
    # # To-Do: Add more complex hidden layer structure, e.g. growing and shrinking layer counts

    hparams = {
        'hiddenLayers' : [1, 5, 10, 12, 24, 100, 500, 1000],
        'neurons'      : [6, 12, 20, 24, 50, 100, 1000],
        'activation'   : ['relu', 'exponential', 'gelu', 'linear', 'selu', 'softmax', 'softplus', 'tanh'],
        'optimizer'    : ['adam', 'sgd', 'rmsprop'],
        'loss'         : ['mse', 'mape'],
        'metrics'      : ['mse'],
        'batch'        : [10, 50, 100, 500, 1000, 2000],
        'epochs'       : [50, 100]
    }

    ares.ann_hyperparam_optimize(hparams)


    # hiddenLayers = 300
    # hlNeurons    = 30
    # epochs       = 500
    # batchSize    = 500
    
    # hiddenLayerActFnct = 'relu'
    # optimizer          = 'adam'
    # outputActFnct      = 'sigmoid'
    # lossFnct           = 'mse'
    # performanceMetric  = 'accuracy'

    # ares.annVerbosity = 0

    # ares.set_regressor('ann', hiddenLayerActFnct, 
    #                   hiddenLayers, hlNeurons, 
    #                   outputActFnct, optimizer, 
    #                   lossFnct, performanceMetric,
    #                   batchSize, epochs)
    
    # ares.train(batchSize, epochs)
