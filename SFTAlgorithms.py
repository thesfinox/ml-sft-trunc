import re

from scipy                   import stats
from sklearn.linear_model    import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm             import LinearSVR, SVR
from lightgbm                import LGBMRegressor
from tensorflow              import keras
from sklearn.model_selection import ParameterGrid
from skopt                   import gp_minimize
from skopt.utils             import use_named_args
from sklearn.metrics         import mean_squared_error, \
                                    mean_absolute_error, \
                                    r2_score, \
                                    explained_variance_score

def pretty_print(dictionary):
    '''
    Pretty print a dictionary.
    
    Required argument:
        dictionary: the dictionary.
    '''
    length = max([len(k) for k in dictionary.keys()]) #------------------- get max length
    
    for key, value in dictionary.items():
        key   = key.ljust(length).replace('_', ' ').upper() #------------- pad and replace underscores
        key   = re.sub(r'(TRAIN|DEV|TEST) *$',
                       lambda match: match.group(1).lower() + ' SET', #--- add human readable output
                       key
                      )
        value = value.lower() if isinstance(value, str) else value
        print('{} = {}'.format(key, value))
    print('\n')

def confidence_interval(y_true, y_pred, confidence = 0.95):
    '''
    Compute the confidence interval of the variance.
    
    Required arguments:
        y_true: true values,
        y_pred: predictions.
        
    Returns:
        the array of lower and upper bounds of the confidence interval.
    '''
    
    # compute the deviation of the data and the squared errors
    deviation = y_pred - y_true #-------------------------------------- > 0 if overestimating, < 0 if underestimating
    sq_errors = deviation ** 2 #--------------------------------------- squared errors

    conf_interval = stats.t.interval(confidence,
                                     sq_errors.shape[0] - 1,
                                     loc   = sq_errors.mean(),
                                     scale = stats.sem(sq_errors)
                                    ) #-------------------------------- compute the confidence interval
    
    return conf_interval

class Analysis:
    '''
    Build several algorithms and fit the data onto them.
    
    Public methods:
        linear_regression: linear regression algorithm,
        elastic_net:       elastic net regression algorithm (l1 and l2 regularised linear regression),
        lasso:             lasso regression algorithm (l1 regularised linear regression),
        ridge:             ridge regression algorithm (l2 regularised linear regression),
        linear_svr:        linear SVR algorithm,
        svr_rbf:           SVR algorithm with Gaussian (rbf) kernel,
        random_forest:     random forest of decision trees,
        gradient_boost:    boosted decision trees,
        ann:               artificial fully connected neural network.
        
    Private methods:
        __ann_model:       build the ANN model.
        
    Getters:
        get_train_data:    get training data (x_train, y_train),
        get_dev_data:      get dev data (x_dev, y_dev),
        get_test_data:     get test data (x_test, y_test),
        get_random_state:  get random state,
        get_n_jobs:        get no. of threads.
    
    Setters:
        set_train_data:    set training data,
        set_dev_data:      set dev data,
        set_test_data:     set test data,
        set_random_state:  set random state,
        set_n_jobs:        set no. of threads.
    '''
    
    def __init__(self,
                 train_data,
                 dev_data,
                 test_data,
                 random_state = None,
                 n_jobs = 1):
        '''
        Class constructor.
        
        Required arguments:
            train_data: tuple containing training data (x_train, y_train)
            dev_data:   tuple containing development data (x_dev, y_dev)
            test_data:  tuple containing test data (x_test, y_test)
            
        Optional arguments:
            random_state: the random seed,
            n_jobs:       the number of concurrent threads.
        '''
        # initialize the variables
        self.__x_train, self.__y_train = train_data
        self.__x_dev,   self.__y_dev   = dev_data
        self.__x_test,  self.__y_test  = test_data
        self.__random_state = random_state
        self.__n_jobs       = n_jobs
    
    # define the getters
    def get_train_data(self):
        '''
        Getter method for the training data.
        
        Returns:
            the training data
        '''
        return (self.__x_train, self.__y_train)
    
    def get_dev_data(self):
        '''
        Get method for the development data.
        
        Returns:
            the development data
        '''
        return (self.__x_dev, self.__y_dev)
    
    def get_test_data(self):
        '''
        Get method for the test data.
        
        Returns:
            the test data
        '''
        return (self.__x_test, self.__y_test)
    
    def get_random_state(self):
        '''
        Get method for the random state.
        
        Returns:
            the random state
        '''
        return self.__random_state
    
    def get_n_jobs(self):
        '''
        Get method for the no. of concurrent threads.
        
        Returns:
            the no. of concurrent threads
        '''
        return self.__n_jobs
    
    # define the setters
    def set_train_data(self, train_data):
        '''
        Set method for the training data.
        
        Required arguments:
            train_data: the training data.
        '''
        self.__x_train, self.__y_train = train_data
    
    def set_dev_data(self, dev_data):
        '''
        Set method for the development data.
        
        Required arguments:
            dev_data: the development data.
        '''
        self.__x_dev, self.__y_dev = dev_data
    
    def set_test_data(self, test_data):
        '''
        Set method for the test data.
        
        Required arguments:
            test_data: the test data.
        '''
        self.__x_test, self.__y_test = test_data
    
    def set_random_state(self, random_state):
        '''
        Set method for the random state.
        
        Required arguments:
            random_state: the random seed.
        '''
        self.__random_state = random_state
    
    def set_n_jobs(self, n_jobs):
        '''
        Set method for the no. of concurrent threads.
        
        Required arguments:
            n_jons: the no. of concurrent threads.
        '''
        self.__n_jobs = n_jobs
        
    # build a neural network model
    def __ann_model(self,
                    n_layers            = 1,
                    n_units             = 10,
                    learning_rate       = 0.1,
                    epochs              = 10,
                    activation          = 'relu',
                    slope               = 0.3,
                    dropout             = True,
                    dropout_rate        = 0.2,
                    batch_normalization = True,
                    momentum            = 0.99,
                    l1_reg              = 0.0,
                    l2_reg              = 0.0
                   ):
        '''
        Create and return a compiled Tensorflow model.

        Required parameters:

        Optional parameters:
            n_layers:            the number of fully connected layers to insert,
            n_units:             the number of units in each layer,
            learning_rate:       the learning rate of gradient descent,
            epochs:              the number of epochs for training,
            activation:          the name of the activation function ('relu' for ReLU or else for LeakyReLU)
            slope:               the slope of the LeakyReLU activation (ignored if ReLU),
            dropout:             whether to use dropout,
            dropout_rate:        the dropout rate (ignored if no dropout),
            batch_normalization: whether to use batch normalization,
            momentum:            the momentum of batch normalization,
            l1_reg:              amount of l1 regularisation,
            l2_reg:              amount of l2 regularisation.

        Returns:
            the fitted model and its history.
        '''

        # instantiate the model
        keras.backend.clear_session()
        model = keras.Sequential(name = 'sft_trunc') #--------------------------------------- create the model
        model.add(keras.layers.InputLayer(input_shape = self.__x_train.shape[1:], name='input'))

        # add FC layers
        for n in range(n_layers):
            model.add(keras.layers.Dense(units                = n_units,
                                         kernel_initializer   = keras.initializers.glorot_uniform(seed = self.__random_state),
                                         bias_initializer     = tf.zeros_initializer(),
                                         activity_regularizer = keras.regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                                         name                 = 'dense_{:d}'.format(n)
                                        )
                     ) #--------------------------------------------------------------------- add FC layer
            if activation == 'relu':
                model.add(keras.layers.Activation('relu',
                                                  name = 'activation_{:d}'.format(n)
                                                 )
                         )
            else:
                model.add(keras.layers.LeakyReLU(alpha = slope,
                                                 name  = 'activation_{:d}'.format(n)
                                                )
                         ) #----------------------------------------------------------------- add activation layer
            if batch_normalization:
                model.add(keras.layers.BatchNormalization(momentum = momentum,
                                                          name     = 'batch_norm_{:d}'.format(n)
                                                         )
                         ) #----------------------------------------------------------------- add batch normalization layer
            if dropout:
                model.add(keras.layers.Dropout(rate = dropout_rate,
                                               seed = self.__random_state,
                                               name = 'dropout_{:d}'.format(n)
                                              )
                         ) #----------------------------------------------------------------- add dropout layer

        # add the output layer
        model.add(keras.layers.Dense(units              = 1,
                                     kernel_initializer = keras.initializers.glorot_uniform(seed = self.__random_state),
                                     bias_initializer   = tf.zeros_initializer(),
                                     name               = 'output'
                                    )
                 ) #------------------------------------------------------------------------- add output layer

        # compile the model
        model.compile(keras.optimizers.Adam(learning_rate = learning_rate),
                      loss       = keras.losses.MeanSquaredError(),
                      metrics    = [keras.metrics.MeanSquaredError(), keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()]
                     )

        return model
    
    # define the algorithms
    def linear_regression(self, params, optimization = False):
        '''
        Linear regression model.
        
        Required arguments:
            params:       dict of hyperparameters.
        
        Optional arguments:
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (grid search).
            
        Returns:
            the fitted model, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # initialise parameters
            best_model           = None #--------------------------------------------------------- unknown for the moment
            best_hyperparameters = None #------------------------------------------------------------- unknown for the moment
            best_mae_train       = np.inf #------------------------------------------------------- highest possible value
            best_mae_dev         = np.inf #------------------------------------------------------- highest possible value
            best_mae_test        = np.inf #------------------------------------------------------- highest possible value
            best_mse_train       = np.inf #------------------------------------------------------- highest possible value
            best_mse_dev         = np.inf #------------------------------------------------------- highest possible value
            best_mse_test        = np.inf #------------------------------------------------------- highest possible value
            best_r2_train        = -np.inf #------------------------------------------------------ highest possible value
            best_r2_dev          = -np.inf #------------------------------------------------------ highest possible value
            best_r2_test         = -np.inf #------------------------------------------------------ highest possible value
            best_evr_train       = -np.inf #------------------------------------------------------ highest possible value
            best_evr_dev         = -np.inf #------------------------------------------------------ highest possible value
            best_evr_test        = -np.inf #------------------------------------------------------ highest possible value
            best_train_pred      = None #--------------------------------------------------------- unknown for the moment
            best_dev_pred        = None #--------------------------------------------------------- unknown for the moment
            best_test_pred       = None #--------------------------------------------------------- unknown for the moment
            best_cint_train      = None #--------------------------------------------------------- unknown for the moment
            best_cint_dev        = None #--------------------------------------------------------- unknown for the moment
            best_cint_test       = None #--------------------------------------------------------- unknown for the moment
            
            # generate a grid of choices and loop over it
            params_grid = list(ParameterGrid(params)) #------------------------------------------- create a grid of possible choices
            for i in range(len(params_grid)):
                log.debug('Exploring {}'.format(params_grid[i]))
                lin_reg = LinearRegression(n_jobs = self.__n_jobs,
                                           normalize = False,
                                           **params_grid[i]
                                          )
                lin_reg.fit(self.__x_train, self.__y_train.ravel()) #------------------------------ fit the estimator
                
                y_train_pred = lin_reg.predict(self.__x_train).reshape(-1,1) #--------------------- generate training predictions
                y_dev_pred   = lin_reg.predict(self.__x_dev).reshape(-1,1) #----------------------- generate development predictions
                y_test_pred  = lin_reg.predict(self.__x_test).reshape(-1,1) #---------------------- generate test predictions
                
                mae_train  = mean_absolute_error(self.__y_train, y_train_pred) #------------------- compute MAE for training data
                mae_dev    = mean_absolute_error(self.__y_dev, y_dev_pred) #----------------------- compute MAE for validation data
                mae_test   = mean_absolute_error(self.__y_test, y_test_pred) #--------------------- compute MAE for test data
                mse_train  = mean_squared_error(self.__y_train, y_train_pred) #-------------------- compute MSE for training data
                mse_dev    = mean_squared_error(self.__y_dev, y_dev_pred) #------------------------ compute MSE for validation data
                mse_test   = mean_squared_error(self.__y_test, y_test_pred) #---------------------- compute MSE for test data
                r2_train   = r2_score(self.__y_train, y_train_pred) #------------------------------ compute R2 for training data
                r2_dev     = r2_score(self.__y_dev, y_dev_pred) #---------------------------------- compute R2 for validation data
                r2_test    = r2_score(self.__y_test, y_test_pred) #-------------------------------- compute R2 for test data
                evr_train  = explained_variance_score(self.__y_train, y_train_pred) #-------------- compute EVR for training data
                evr_dev    = explained_variance_score(self.__y_dev, y_dev_pred) #------------------ compute EVR for validation data
                evr_test   = explained_variance_score(self.__y_test, y_test_pred) #---------------- compute EVR for test data
                cint_train = confidence_interval(self.__y_train, y_train_pred) #------------------- compute 95% confindence for training data
                cint_dev   = confidence_interval(self.__y_dev, y_dev_pred) #----------------------- compute 95% confindence for validation data
                cint_test  = confidence_interval(self.__y_test, y_test_pred) #--------------------- compute 95% confindence for test data
                
                if mse_dev < best_mse_dev: #------------------------------------------------------- update the best results (only dev set for comparison)
                    best_model           = lin_reg
                    best_hyperparameters = params_grid[i]
                    best_mae_train       = mae_train
                    best_mae_dev         = mae_dev
                    best_mae_test        = mae_test
                    best_mse_train       = mse_train
                    best_mse_dev         = mse_dev
                    best_mse_test        = mse_test
                    best_r2_train        = r2_train
                    best_r2_dev          = r2_dev
                    best_r2_test         = r2_test
                    best_evr_train       = evr_train
                    best_evr_dev         = evr_dev
                    best_evr_test        = evr_test
                    best_train_pred      = y_train_pred
                    best_dev_pred        = y_dev_pred
                    best_test_pred       = y_test_pred
                    best_cint_train      = cint_train
                    best_cint_dev        = cint_dev
                    best_cint_test       = cint_test
            
            # organise the predictions
            best_predictions = {'y_train_pred': best_train_pred,
                                'y_dev_pred':   best_dev_pred,
                                'y_test_pred':  best_test_pred
                               }
            
            # organise the metrics
            best_metrics = {'mae_train': best_mae_train,
                            'mae_dev':   best_mae_dev,
                            'mae_test':  best_mae_test,
                            'mse_train': best_mse_train,
                            'mse_dev':   best_mse_dev,
                            'mse_test':  best_mse_test,
                            'r2_train':  best_r2_train,
                            'r2_dev':    best_r2_dev,
                            'r2_test':   best_r2_test,
                            'evr_train': best_evr_train,
                            'evr_dev':   best_evr_dev,
                            'evr_test':  best_evr_test
                           }
            
            # organise the confidence intervals
            best_cint = {'train': best_cint_train,
                         'dev':   best_cint_dev,
                         'test':  best_cint_test
                        }
            
            return best_model, best_hyperparameters, best_predictions, best_metrics, best_cint
        else:
            # define the estimator and fit
            lin_reg = LinearRegression(n_jobs = self.__n_jobs, normalize = False, **params) #------- define the estimator
            lin_reg.fit(self.__x_train, self.__y_train.ravel()) #----------------------------------- fit the estimator
            
            # compute the predictions
            y_train_pred = lin_reg.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = lin_reg.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = lin_reg.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred), #------------------- compute 95% confindence for training data
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred), #----------------------- compute 95% confindence for validation data
                    'test':  confidence_interval(self.__y_test, y_test_pred) #---------------------- compute 95% confindence for test data
                   }
            
            return lin_reg, predictions, metrics, cint
        
    def elastic_net(self, params, optimization = False, n_calls = 10, max_iter = 1e3, tol = 1.0e-4):
        '''
        Elastic Net regression model.
        
        Required arguments:
            params:       dict of hyperparameters or list of skopt.Spaces.
        
        Optional arguments:
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (Bayes search),
            n_calls:      the number of iterations of the Bayes optimisation,
            max_iter:     maximum number of iterations (elastic net),
            tol:          tolerance (elastic net).
            
        Returns:
            the fitted model, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # define the estimator
            el_net = ElasticNet(random_state = self.__random_state,
                                normalize = False,
                                max_iter = max_iter,
                                tol = tol
                               )
            
            # get the list of names of the hyperparameters
            hyp_names = [s.name for s in params]
            
            # define the loss function
            @use_named_args(params)
            def objective(**args):
                # save iteration
                log.debug('Exploring {}'.format(args))
                
                # fit the estimator to the train set
                el_net.set_params(**args)
                el_net.fit(self.__x_train, self.__y_train.ravel())
                
                # compute predictions on the dev set
                y_dev_pred = el_net.predict(self.__x_dev).reshape(-1,1)
                
                # compute the mean squared error
                return mean_squared_error(self.__y_dev, y_dev_pred)
            
            # compute the minimisation
            el_net_res = gp_minimize(objective,
                                     params,
                                     n_calls = n_calls,
                                     random_state = self.__random_state,
                                     n_jobs = self.__n_jobs
                                    )
            best_hyperparameters = dict(zip(hyp_names, el_net_res.x))
            
            # set best hyperparameters and fit the model
            el_net.set_params(**best_hyperparameters)
            el_net.fit(self.__x_train, self.__y_train.ravel())
            
            # compute the predictions
            y_train_pred = el_net.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = el_net.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = el_net.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #---------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #----------- compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #--------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------ compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #---------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #-------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #---------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #-------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------ compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------ compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #---------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #--------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return el_net, best_hyperparameters, predictions, metrics, cint                
        else:
            # define the estimator and fit
            el_net = ElasticNet(random_state = self.__random_state, **params) #-------------------- define the estimator
            el_net.fit(self.__x_train, self.__y_train.ravel()) #----------------------------------- fit the estimator
            
            # compute the predictions
            y_train_pred = el_net.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = el_net.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = el_net.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #---------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #----------- compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #--------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------ compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #---------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #-------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #---------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #-------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------ compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------ compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #---------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #--------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return el_net, predictions, metrics, cint
        
    def lasso(self, params, optimization = False, n_calls = 10, max_iter = 1e3, tol = 1.0e-4):
        '''
        Lasso regression model.
        
        Required arguments:
            params:       dict of hyperparameters or list of skopt.Spaces.
        
        Optional arguments:
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (Bayes search),
            n_calls:      the number of iterations of the Bayes optimisation,
            max_iter:     maximum number of iterations (lasso),
            tol:          tolerance (lasso).
            
        Returns:
            the fitted model, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # define the estimator
            lasso = Lasso(random_state = self.__random_state,
                          normalize = False,
                          positive = False,
                          max_iter = max_iter,
                          tol = tol
                         )
            
            # get the list of names of the hyperparameters
            hyp_names = [s.name for s in params]
            
            # define the loss function
            @use_named_args(params)
            def objective(**args):
                # save iteration
                log.debug('Exploring {}'.format(args))
                
                # fit the estimator to the train set
                lasso.set_params(**args)
                lasso.fit(self.__x_train, self.__y_train.ravel())
                
                # compute predictions on the dev set
                y_dev_pred = lasso.predict(self.__x_dev).reshape(-1,1)
                
                # compute the mean squared error
                return mean_squared_error(self.__y_dev, y_dev_pred)
            
            # compute the minimisation
            lasso_res = gp_minimize(objective,
                                    params,
                                    n_calls = n_calls,
                                    random_state = self.__random_state,
                                    n_jobs = self.__n_jobs
                                   )
            best_hyperparameters = dict(zip(hyp_names, lasso_res.x))
            
            # set best hyperparameters and fit the model
            lasso.set_params(**best_hyperparameters)
            lasso.fit(self.__x_train, self.__y_train.ravel())
            
            # compute the predictions
            y_train_pred = lasso.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = lasso.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = lasso.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #--------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #---------- compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #-------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #------------ compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #----------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #--------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #--------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #----------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #----- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #--------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #-------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return lasso, best_hyperparameters, predictions, metrics, cint                
        else:
            # define the estimator and fit
            lasso = Lasso(random_state = self.__random_state,
                          normalize = False,
                          positive = False,
                          max_iter = max_iter,
                          tol = tol,
                          **params
                         )
            lasso.fit(self.__x_train, self.__y_train.ravel()) #--------------------------------------- fit the estimator
            
            # compute the predictions
            y_train_pred = lasso.predict(self.__x_train).reshape(-1,1) #---------------------------- compute training predictions
            y_dev_pred   = lasso.predict(self.__x_dev).reshape(-1,1) #------------------------------ compute validation predictions
            y_test_pred  = lasso.predict(self.__x_test).reshape(-1,1) #----------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #--------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return lasso, predictions, metrics, cint
        
    def ridge(self, params, optimization = False, n_calls = 10, max_iter = 1e3, tol = 1.0e-4):
        '''
        Ridge regression model.
        
        Required arguments:
            params:       dict of hyperparameters or list of skopt.Spaces.
        
        Optional arguments:
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (Bayes search),
            n_calls:      the number of iterations of the Bayes optimisation,
            max_iter:     maximum number of iterations (ridge),
            tol:          tolerance (ridge).
            
        Returns:
            the fitted model, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # define the estimator
            ridge = Ridge(random_state = self.__random_state,
                          normalize = False,
                          max_iter = max_iter,
                          tol = tol
                         )
            
            # get the list of names of the hyperparameters
            hyp_names = [s.name for s in params]
            
            # define the loss function
            @use_named_args(params)
            def objective(**args):
                # save iteration
                log.debug('Exploring {}'.format(args))
                
                # fit the estimator to the train set
                ridge.set_params(**args)
                ridge.fit(self.__x_train, self.__y_train.ravel())
                
                # compute predictions on the dev set
                y_dev_pred = ridge.predict(self.__x_dev).reshape(-1,1)
                
                # compute the mean squared error
                return mean_squared_error(self.__y_dev, y_dev_pred)
            
            # compute the minimisation
            ridge_res = gp_minimize(objective,
                                    params,
                                    n_calls = n_calls,
                                    random_state = self.__random_state,
                                    n_jobs = self.__n_jobs
                                   )
            best_hyperparameters = dict(zip(hyp_names, ridge_res.x))
            
            # set best hyperparameters and fit the model
            ridge.set_params(**best_hyperparameters)
            ridge.fit(self.__x_train, self.__y_train.ravel())
            
            # compute the predictions
            y_train_pred = ridge.predict(self.__x_train).reshape(-1,1) #---------------------------- compute training predictions
            y_dev_pred   = ridge.predict(self.__x_dev).reshape(-1,1) #------------------------------ compute validation predictions
            y_test_pred  = ridge.predict(self.__x_test).reshape(-1,1) #----------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return ridge, best_hyperparameters, predictions, metrics, cint                
        else:
            # define the estimator and fit
            ridge = Ridge(random_state = self.__random_state,
                          normalize = False,
                          max_iter = max_iter,
                          tol = tol,
                          **params
                         )
            ridge.fit(self.__x_train, self.__y_train.ravel()) #--------------------------------------- fit the estimator
            
            # compute the predictions
            y_train_pred = ridge.predict(self.__x_train).reshape(-1,1) #---------------------------- compute training predictions
            y_dev_pred   = ridge.predict(self.__x_dev).reshape(-1,1) #------------------------------ compute validation predictions
            y_test_pred  = ridge.predict(self.__x_test).reshape(-1,1) #----------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return ridge, predictions, metrics, cint
        
    def linear_svr(self, params, optimization = False, n_calls = 10, max_iter = 1e3):
        '''
        Linear SVR regression model.
        
        Required arguments:
            params:       dict of hyperparameters or list of skopt.Spaces.
        
        Optional arguments:
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (Bayes search),
            n_calls:      the number of iterations of the Bayes optimisation,
            max_iter:     maximum number of iterations (linear SVR).
            
        Returns:
            the fitted model, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # define the estimator
            lin_svr = LinearSVR(random_state = self.__random_state, max_iter = max_iter)
            
            # get the list of names of the hyperparameters
            hyp_names = [s.name for s in params]
            
            # define the loss function
            @use_named_args(params)
            def objective(**args):
                # save iteration
                log.debug('Exploring {}'.format(args))
                
                # fit the estimator to the train set
                lin_svr.set_params(**args)
                lin_svr.fit(self.__x_train, self.__y_train.ravel())
                
                # compute predictions on the dev set
                y_dev_pred = lin_svr.predict(self.__x_dev).reshape(-1,1)
                
                # compute the mean squared error
                return mean_squared_error(self.__y_dev, y_dev_pred)
            
            # compute the minimisation
            lin_svr_res = gp_minimize(objective,
                                      params,
                                      n_calls = n_calls,
                                      random_state = self.__random_state,
                                      n_jobs = self.__n_jobs
                                     )
            best_hyperparameters = dict(zip(hyp_names, lin_svr_res.x))
            
            # set best hyperparameters and fit the model
            lin_svr.set_params(**best_hyperparameters)
            lin_svr.fit(self.__x_train, self.__y_train.ravel())
            
            # compute the predictions
            y_train_pred = lin_svr.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = lin_svr.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = lin_svr.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return lin_svr, best_hyperparameters, predictions, metrics, cint                
        else:
            # define the estimator and fit
            lin_svr = LinearSVR(random_state = self.__random_state, max_iter = max_iter, **params)
            lin_svr.fit(self.__x_train, self.__y_train.ravel()) #----------------------------------- fit the estimator
            
            # compute the predictions
            y_train_pred = lin_svr.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = lin_svr.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = lin_svr.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return lin_svr, predictions, metrics, cint
        
    def svr_rbf(self, params, optimization = False, n_calls = 10, tol = 1.0e-4):
        '''
        SVR regression model using a Gaussian kernel (rbf).
        
        Required arguments:
            params:       dict of hyperparameters or list of skopt.Spaces.
        
        Optional arguments:
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (Bayes search),
            n_calls:      the number of iterations of the Bayes optimisation,
            tol:          tolerance (SVR).
            
        Returns:
            the fitted model, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # define the estimator
            svr_rbf = SVR(kernel = 'rbf', tol = tol)
            
            # get the list of names of the hyperparameters
            hyp_names = [s.name for s in params]
            
            # define the loss function
            @use_named_args(params)
            def objective(**args):
                # save iteration
                log.debug('Exploring {}'.format(args))
                
                # fit the estimator to the train set
                svr_rbf.set_params(**args)
                svr_rbf.fit(self.__x_train, self.__y_train.ravel())
                
                # compute predictions on the dev set
                y_dev_pred = svr_rbf.predict(self.__x_dev).reshape(-1,1)
                
                # compute the mean squared error
                return mean_squared_error(self.__y_dev, y_dev_pred)
            
            # compute the minimisation
            svr_rbf_res = gp_minimize(objective,
                                      params,
                                      n_calls = n_calls,
                                      random_state = self.__random_state,
                                      n_jobs = self.__n_jobs
                                     )
            best_hyperparameters = dict(zip(hyp_names, svr_rbf_res.x))
            
            # set best hyperparameters and fit the model
            svr_rbf.set_params(**best_hyperparameters)
            svr_rbf.fit(self.__x_train, self.__y_train.ravel())
            
            # compute the predictions
            y_train_pred = svr_rbf.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = svr_rbf.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = svr_rbf.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return svr_rbf, best_hyperparameters, predictions, metrics, cint                
        else:
            # define the estimator and fit
            svr_rbf = SVR(kernel = 'rbf', tol = tol, **params)
            svr_rbf.fit(self.__x_train, self.__y_train.ravel()) #----------------------------------- fit the estimator
            
            # compute the predictions
            y_train_pred = svr_rbf.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = svr_rbf.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = svr_rbf.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return svr_rbf, predictions, metrics, cint
        
    def random_forest(self, params, optimization = False, n_calls = 10):
        '''
        Random forest of decision trees.
        
        Required arguments:
            params:       dict of hyperparameters or list of skopt.Spaces.
        
        Optional arguments:
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (Bayes search),
            n_calls:      the number of iterations of the Bayes optimisation.
            
        Returns:
            the fitted model, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # define the estimator
            rnd_for = LGBMRegressor(boosting_type = 'rf', objective = 'regression', subsample_freq = 1, n_jobs = 1)
            
            # get the list of names of the hyperparameters
            hyp_names = [s.name for s in params]
            
            # define the loss function
            @use_named_args(params)
            def objective(**args):
                # save iteration
                log.debug('Exploring {}'.format(args))
                
                # fit the estimator to the train set
                rnd_for.set_params(**args)                    
                rnd_for.fit(self.__x_train, self.__y_train.ravel())
                
                # compute predictions on the dev set
                y_dev_pred = rnd_for.predict(self.__x_dev).reshape(-1,1)
                
                # compute the mean squared error
                return mean_squared_error(self.__y_dev, y_dev_pred)
            
            # compute the minimisation
            rnd_for_res = gp_minimize(objective,
                                      params,
                                      n_calls = n_calls,
                                      random_state = self.__random_state,
                                      n_jobs = self.__n_jobs
                                     )
            best_hyperparameters = dict(zip(hyp_names, rnd_for_res.x))
            
            # set best hyperparameters and fit the model
            rnd_for.set_params(**best_hyperparameters)
            rnd_for.fit(self.__x_train, self.__y_train.ravel())
            
            # compute the predictions
            y_train_pred = rnd_for.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = rnd_for.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = rnd_for.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return rnd_for, best_hyperparameters, predictions, metrics, cint                
        else:
            # define the estimator and fit
            rnd_for = LGBMRegressor(boosting_type = 'rf',
                                    objective = 'regression',
                                    subsample_freq = 1,
                                    n_jobs = self.__n_jobs,
                                    **params
                                   )
            rnd_for.fit(self.__x_train, self.__y_train.ravel()) #----------------------------------- fit the estimator
            
            # compute the predictions
            y_train_pred = rnd_for.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = rnd_for.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = rnd_for.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return rnd_for, predictions, metrics, cint
        
    def gradient_boosting(self, params, optimization = False, n_calls = 10):
        '''
        Boosted decision trees.
        
        Required arguments:
            params:       dict of hyperparameters or list of skopt.Spaces.
        
        Optional arguments:
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (Bayes search),
            n_calls:      the number of iterations of the Bayes optimisation.
            
        Returns:
            the fitted model, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # define the estimator
            grd_boost = LGBMRegressor(boosting_type = 'gbdt',
                                      objective = 'regression',
                                      subsample_freq = 1,
                                      n_jobs = 1
                                     )
            
            # get the list of names of the hyperparameters
            hyp_names = [s.name for s in params]
            
            # define the loss function
            @use_named_args(params)
            def objective(**args):
                # save iteration
                log.debug('Exploring {}'.format(args))
                
                # fit the estimator to the train set
                grd_boost.set_params(**args)                    
                grd_boost.fit(self.__x_train, self.__y_train.ravel())
                
                # compute predictions on the dev set
                y_dev_pred = grd_boost.predict(self.__x_dev).reshape(-1,1)
                
                # compute the mean squared error
                return mean_squared_error(self.__y_dev, y_dev_pred)
            
            # compute the minimisation
            grd_boost_res = gp_minimize(objective,
                                        params,
                                        n_calls = n_calls,
                                        random_state = self.__random_state,
                                        n_jobs = self.__n_jobs
                                       )
            best_hyperparameters = dict(zip(hyp_names, grd_boost_res.x))
            
            # set best hyperparameters and fit the model
            grd_boost.set_params(**best_hyperparameters)
            grd_boost.fit(self.__x_train, self.__y_train.ravel())
            
            # compute the predictions
            y_train_pred = grd_boost.predict(self.__x_train).reshape(-1,1) #------------------------ compute training predictions
            y_dev_pred   = grd_boost.predict(self.__x_dev).reshape(-1,1) #-------------------------- compute validation predictions
            y_test_pred  = grd_boost.predict(self.__x_test).reshape(-1,1) #------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return grd_boost, best_hyperparameters, predictions, metrics, cint                
        else:
            # define the estimator and fit
            grd_boost = LGBMRegressor(boosting_type = 'gbdt',
                                      objective = 'regression',
                                      subsample_freq = 1,
                                      n_jobs = self.__n_jobs,
                                      **params
                                     )
            grd_boost.fit(self.__x_train, self.__y_train.ravel()) #--------------------------------- fit the estimator
            
            # compute the predictions
            y_train_pred = grd_boost.predict(self.__x_train).reshape(-1,1) #------------------------ compute training predictions
            y_dev_pred   = grd_boost.predict(self.__x_dev).reshape(-1,1) #-------------------------- compute validation predictions
            y_test_pred  = grd_boost.predict(self.__x_test).reshape(-1,1) #------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #----------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return grd_boost, predictions, metrics, cint
        
    def ann_model(self, params, batch_size = 32, epochs = 10, verbose = 1, filename = 'ann_model.h5', optimization = False, n_calls = 10):
        '''
        Boosted decision trees.
        
        Required arguments:
            params:       dict of hyperparameters or list of skopt.Spaces.
        
        Optional arguments:
            batch_size:   the size of the mini-batch or batch gradient descent,
            epochs:       the training epochs,
            verbose:      enable verbosity,
            filename:     location of the saved model,
            optimization: if False then the estimators takes params as hyperparameters and outputs the predictions,
                          if True params are passed to the optimization procedure (Bayes search),
            n_calls:      the number of iterations of the Bayes optimisation.
            
        Returns:
            the fitted model, its history, the predictions, the metrics, the confidence intervals (if optimization is False),
            the best fitted model, its history, the best parameters, the predictions, the metrics, the confidence intervals (if optimization is True).
        '''
        if optimization:
            # get the list of names of the hyperparameters
            hyp_names = [s.name for s in params]
            
            # define the loss function
            @use_named_args(params)
            def objective(**args):
                # save iteration
                log.debug('Exploring {}'.format(args))
                
                # create the model
                ann_model = self.__ann_model(**args)
                
                # fit the estimator to the train set            
                ann_model_history = ann_model.fit(x               = self.__x_train,
                                                  y               = self.__y_train,
                                                  batch_size      = batch_size,
                                                  epochs          = epochs,
                                                  verbose         = 0,
                                                  validation_data = (self.__x_dev, self.__y_dev),
                                                  callbacks       = [keras.callbacks.ModelCheckpoint(filename,
                                                                                                     monitor        = 'val_loss',
                                                                                                     verbose        = 0,
                                                                                                     save_best_only = True
                                                                                                    )
                                                                    ]
                                                 )
            
                #restore best model
                ann_model = keras.models.load_model(filename)
                
                # compute predictions on the dev set
                y_dev_pred = ann_model.predict(self.__x_dev).reshape(-1,1)
                
                # compute the mean squared error
                return mean_squared_error(self.__y_dev, y_dev_pred)
            
            # compute the minimisation
            ann_model_res = gp_minimize(objective,
                                        params,
                                        n_calls = n_calls,
                                        random_state = self.__random_state,
                                        n_jobs = self.__n_jobs
                                       )
            best_hyperparameters = dict(zip(hyp_names, ann_model_res.x))
            
            # set best hyperparameters and fit the model
            ann_model = self.__ann_model(**best_hyperparameters)            
            ann_model_history = ann_model.fit(x               = self.__x_train,
                                              y               = self.__y_train,
                                              batch_size      = batch_size,
                                              epochs          = epochs,
                                              verbose         = 0,
                                              validation_data = (self.__x_dev, self.__y_dev),
                                              callbacks       = [keras.callbacks.ModelCheckpoint(filename,
                                                                                                 monitor        = 'val_loss',
                                                                                                 verbose        = 0,
                                                                                                 save_best_only = True
                                                                                                )
                                                                ]
                                             )
            
            #restore best model
            ann_model = keras.models.load_model(filename)
            
            # compute the predictions
            y_train_pred = ann_model.predict(self.__x_train).reshape(-1,1) #-------------------------- compute training predictions
            y_dev_pred   = ann_model.predict(self.__x_dev).reshape(-1,1) #---------------------------- compute validation predictions
            y_test_pred  = ann_model.predict(self.__x_test).reshape(-1,1) #--------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #--------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return ann_model, ann_model_history.history, best_hyperparameters, predictions, metrics, cint                
        else:
            # define the estimator and fit
            ann_model = self.__ann_model(**params)
            ann_model_history = ann_model.fit(x               = self.__x_train,
                                              y               = self.__y_train,
                                              batch_size      = batch_size,
                                              epochs          = epochs,
                                              verbose         = verbose,
                                              validation_data = (self.__x_dev, self.__y_dev),
                                              callbacks       = [keras.callbacks.ModelCheckpoint(filename,
                                                                                                 monitor        = 'val_loss',
                                                                                                 verbose        = verbose,
                                                                                                 save_best_only = True
                                                                                                )
                                                                ]
                                             )
            
            #restore best model
            ann_model = keras.models.load_model(filename)
            
            # compute the predictions
            y_train_pred = ann_model.predict(self.__x_train).reshape(-1,1) #------------------------ compute training predictions
            y_dev_pred   = ann_model.predict(self.__x_dev).reshape(-1,1) #-------------------------- compute validation predictions
            y_test_pred  = ann_model.predict(self.__x_test).reshape(-1,1) #------------------------- compute test predictions
            
            predictions = {'y_train_pred': y_train_pred,
                           'y_dev_pred':   y_dev_pred,
                           'y_test_pred':  y_test_pred
                          } #--------------------------------------------------------------------- collect predictions
            
            # compute the metrics
            metrics = {'mae_train': mean_absolute_error(self.__y_train, y_train_pred), #------------ compute MAE of training predictions
                       'mae_dev':   mean_absolute_error(self.__y_dev, y_dev_pred), #---------------- compute MAE of validation predictions
                       'mae_test':  mean_absolute_error(self.__y_test, y_test_pred), #-------------- compute MAE of test predictions
                       'mse_train': mean_squared_error(self.__y_train, y_train_pred), #------------- compute MSE of training predictions
                       'mse_dev':   mean_squared_error(self.__y_dev, y_dev_pred), #----------------- compute MSE of validation predictions
                       'mse_test':  mean_squared_error(self.__y_test, y_test_pred), #--------------- compute MSE of test predictions
                       'r2_train':  r2_score(self.__y_train, y_train_pred), #----------------------- compute R2 of training predictions
                       'r2_dev':    r2_score(self.__y_dev, y_dev_pred), #--------------------------- compute R2 of validation predictions
                       'r2_test':   r2_score(self.__y_test, y_test_pred), #------------------------- compute R2 of test predictions
                       'evr_train': explained_variance_score(self.__y_train, y_train_pred), #------- compute explained variance ratio of training predictions
                       'evr_dev':   explained_variance_score(self.__y_dev, y_dev_pred), #----------- compute explained variance ratio of validation predictions
                       'evr_test':  explained_variance_score(self.__y_test, y_test_pred) #---------- compute explained variance ratio of test predictions
                      }
            
            # compute the confidence intervals
            cint = {'train': confidence_interval(self.__y_train, y_train_pred),
                    'dev':   confidence_interval(self.__y_dev, y_dev_pred),
                    'test':  confidence_interval(self.__y_test, y_test_pred)
                   }
            
            return ann_model, ann_model_history.history, predictions, metrics, cint
