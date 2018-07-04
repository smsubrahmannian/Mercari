import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype,is_numeric_dtype
from sklearn.model_selection import train_test_split,KFold,ParameterGrid
from sklearn.ensemble import RandomForestRegressor,forest
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tqdm._tqdm_notebook import tqdm,tqdm_notebook
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
import scipy
from scipy.cluster import hierarchy as hc
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def train_cats(df):
    '''Change any columns of strings in a panda's dataframe to a column of
       catagorical values. This applies the changes inplace.
    Input:
        df: A pandas dataframe. Any columns of strings will be changed to categorical values.
    '''
    for col_name, col in df.items():
        if is_string_dtype(col): df[col_name] = col.astype('category').cat.as_ordered()
            
def apply_cats(df, trn):
    '''Changes any columns of strings in df into categorical variables using trn as
       a template for the category codes.
    Input:
        df: A pandas dataframe. Any columns of strings will be changed to categorical values.
        trn: A pandas dataframe. When creating a category for df, it looks up the
            what the category's code were in trn and makes those the category codes
            for df.
    '''
   
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)

def numericalize(df, col, name):
    '''Changes the column col from a categorical type to it's integer codes.
    Inputs:
        df: A pandas dataframe. df[name] will be filled with the integer codes from
            col.
        col: The column you wish to change into the categories.
        name: column name
    
    Outputs: converted dataframe
     '''
    if not is_numeric_dtype(col):
        df[name] = col.cat.codes+1
    return df
        
def proc_df(df, y_fld):
    '''proc_df takes a data frame df and splits off the response variable, and
       changes the df into an entirely numeric dataframe.
    Inputs:
        df: The data frame you wish to process.
        y_fld: The name of the response variable
    Outputs: 
        res : list containing the final dataframe and its y values
    '''  
    df = df.copy()
    y = df[y_fld].values
    df.drop([y_fld], axis=1, inplace=True)
    for n,c in df.items(): 
        df = numericalize(df, c, n)
    res = [df, y]
    return res

def reg_target_encoding(train, col,target_col, splits=5):
    """ Computes regularize mean encoding.
    Inputs:
        train: training dataframe
        col : column name on which you want to perfrom mean encoding
        target_col : name of the target column 
    Output:
        train: training data with regualarized mean encoded features
    """
    kf = KFold(n_splits=splits)
    global_mean = train[target_col].mean()
    for train_index,test_index in kf.split(train):
        kfold_mean = train.iloc[train_index,:].groupby(col)[target_col].mean()
        train.loc[test_index,col+'_mean_enc'] =  train.loc[test_index,col].map(kfold_mean) 
    train[col+"_mean_enc"].fillna(global_mean, inplace=True)
    train[col+"_mean_enc"] = train[col+"_mean_enc"].astype('float32')
    return train

def mean_encoding_test(test, train, col,target_col):
    """ Computes target encoding for test data.
    Inputs:
        test: test dataframe
        train: training dataframe
        col: column name on which you want to perfrom mean encoding
        target_col : name of the target column 
    Outputs: 
        train: training data with regualarized mean encoded features
    """
    global_mean = train[target_col].mean()
    mean_col = train.groupby(col)[target_col].mean()
    test[col+"_mean_enc"] = test[col].map(mean_col)
    test[col+"_mean_enc"].fillna(global_mean, inplace=True)
    test[col+"_mean_enc"] = test[col+"_mean_enc"].astype('float32')
    return test


def perform_regularised_cv(train,y_colname, grid,high_card_cols, folds = 5,metric = mean_absolute_error,model='XGBoost'):
    '''Performs grid search crossfold validation with support for regularised mean encoding
    Inputs:
        train: Input data set
        y_colname : target column name
        grid: Set of hyperparameters over which the model is to be tuned
        high_card_col : categorical columns you want to consider for mean encoding
        folds: Number of folds to be used for cross validation
    Outputs:
        all_scores: the list of final scores
    '''
    kf = KFold(folds, random_state=0, shuffle=True)
    param_grid = ParameterGrid(grid)
    all_scores = [] #Store all scores
    for params in tqdm_notebook(param_grid):
        errors = []
        for train_idx, test_idx in kf.split(train):
            # Split data into train and test
            kf_train, kf_test = train.iloc[train_idx,:], train.iloc[test_idx,:]
            kf_train.reset_index(inplace=True,drop=True)
            kf_test.reset_index(inplace=True,drop=True)
            _,error,_,_ = train_model(params,kf_train,kf_test,y_colname,high_card_cols,valid=True,metric= metric,model='XGBoost')
            errors.append(error)
        avg_score = np.mean(errors) #Average scores of all KFold
        all_scores.append((params, avg_score))
        rmsle = np.sqrt(avg_score)
        tqdm.write(f'Parameters: {params} RMSLE: {rmsle}')
    return all_scores

            
def print_score(model,X,y,metric = mean_absolute_error):
    '''Prints the score for the model
    Inputs:
        model : trained model
        X : training dataframe without the target column
        y : target column
        metric : your chosen metric
    '''
    preds = model.predict(X)
    metric_eval = metric(y,preds)
    return metric_eval

def train_model(params,train,test,y_colname,cols_to_mean_enc,valid=None,metric=mean_squared_error,model='XGBoost'):
    '''Train the model with given set of hyperparameters
    Inputs:
        params - Dict of hyperparameters and chosen values
        train  - Train Data
        test   - Validation Data/test data
        y_colname : target column name
        cols_to_mean_enc : columns to perform mean encoding
        valid: A flag to check if I am using a validation set
        metric - Metric to compute model performance on
    Outputs:
        rf : random forest model
        metric_eval : score according to the chosen metric
        train : final train data
        test : final test/validation data
    '''     
#     print('Category encoding')             
    train_cats(train)
    apply_cats(test,train)
    
    
#     print('Apply mean encoding on train dataset')
    for col in cols_to_mean_enc:
        train = reg_target_encoding(train,col,y_colname)
        test = mean_encoding_test(test,train,col,y_colname)
    
    for n,c in train.items(): 
        train = numericalize(train, c, n)
    for n,c in test.items(): 
        test = numericalize(test, c, n)

    # Train Random Forest Regressor
    metric_eval = None
    if model == 'RF':
#         print('Train Random Forest Regressor')
        mdl = RandomForestRegressor(**params) 
        mdl.fit(train.drop([y_colname], axis=1), train.loc[:,y_colname])
        if valid:
            preds = mdl.predict(test.drop([y_colname], axis=1))
            metric_eval = metric(test[y_colname], preds)

    elif model == 'XGBoost':
#         print('Train XGBoost Regressor')
        mdl = XGBRegressor(**params)
        mdl.fit(train.drop([y_colname], axis=1).values, train.loc[:,y_colname].values)
        if valid:
            preds = mdl.predict(test.drop([y_colname], axis=1).values)
            metric_eval = metric(test[y_colname].values, preds)


    return mdl, metric_eval,train,test

def n_estim_calculate(model,df):
    ''' Generates a plot that shows the progression of average RMSE after increading trees.
        This method helps us to estimate the  number of estimators required for the random forest.
    Inputs:
        model : trained model
        df : training dataframe
    '''
    
    preds = np.stack([t.predict(df.drop('price',axis=1)) for t in model.estimators_])
    plt.plot([mean_squared_error(y_pred=np.mean(preds[:i+1],axis=0), y_true=df['price']) for i in range(0, len(model.estimators_))])
    plt.title('Error Decay aggregated across the number of estimators')
    plt.xlabel('Number of estimators')
    plt.ylabel('RMSLE')
    plt.show()
    
def feature_importance(df,m):
    ''' Calculate feature importance
    Input:
        df: training dataframe
        m : trained model
    Output:
        fi : dataframe containing each column and its feature importance 
    '''    
    fi = pd.DataFrame({'cols': df.drop('price',axis=1).columns, 'importance': m.feature_importances_})
    fi.sort_values(by = 'importance', ascending=False, inplace=True)
    fi.reset_index(inplace=True,drop=True)
    return fi

def plot_feature_importance(fi):
    '''
    '''
    
    sns.set(style="darkgrid")
    g = sns.factorplot(x="importance", y="cols", data=fi,size=5, aspect= 2,kind="bar", color='green')
    g.set_ylabels("price")
    plt.show()

def pred_ci(model, X_train, y_train, percentile = 95):
    """computes the prediction interval, standard deviation for each observation. 
    Inputs:
        model = random forest model
        X_train = input (training) df
        y_train = response array
        percentile = required confidence level
        
    Outputs: 
        df_sorted: a sorted dataframe 

    """
    allTree_preds = np.stack([t.predict(X_train) for t in model.estimators_], axis = 0)
    err_down = np.percentile(allTree_preds, (100 - percentile) / 2.0  ,axis=0)
    err_up = np.percentile(allTree_preds, 100- (100 - percentile) / 2.0  ,axis=0)
    
    ci = err_up - err_down
    yhat = model.predict(X_train)
    y = y_train
    
    df = pd.DataFrame()
    df['y_05'] = err_down 
    df['y_95'] = err_up
    df['y'] = y
    df['yhat'] = yhat
    df['deviation'] = np.std(allTree_preds,axis=0)
    df.reset_index(inplace=True,drop =True)
    df_sorted = df.iloc[np.argsort(df['deviation'])[::-1]]
    return df_sorted

def dendrogram(X_train):
    '''plots a dendrogram which allows one to find collinear variables
    Input:
        X_train: Only the features of training dataframe
    '''
    corr = np.round(scipy.stats.spearmanr(X_train).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(16,15))
    dendrogram = hc.dendrogram(z, labels=X_train.columns, orientation='left', leaf_font_size=16)
    plt.show()
    
def get_sample(df,n):    
    ''' Takes a sample of the given dataframe
    Input:
        df : Input dataframe
        n = number of records to be sampled
    Output:
        sampled dataframe
    '''
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

def partial_dependence(X, fldname,m):
    ''' Computes partial dependence or the change in target with respect to the change in the selected feature
    Inputs:
        X: dataframe with features
        fldname: selected column
        m: trained model
        categorymap: maps each category with its id
    Output:
        final: dataframe that has partial dependence value for each value
    '''

    Xnew = X.copy()
    xreq = X[fldname].sort_values().unique()
    ypred = np.zeros((Xnew.shape[0],len(xreq)))
    counter =0
    for i in xreq:
        Xnew[fldname] = i
        ypred[:,counter] = m.predict(Xnew)  
        counter +=1
    final = pd.DataFrame(ypred,columns=[xreq]).apply(np.mean,axis=0).to_frame()
    final.reset_index(inplace=True)
    final['level_0']=final['level_0']
    final.columns = [fldname,'value']
    return final

def plot_partial_dependence(final):
    '''plots the partial dependence
    Input:
         final: dataframe that has partial dependence value for each value
    '''
    
    colist = final.columns
    sns.set(style="darkgrid")
    g = sns.factorplot(x= colist[0], y=colist[1], data=final,size=5, aspect= 2,kind="bar", color='green')


def category_mapper(train_df,train,fldname):
    '''It takes the raw dataframe and a trained dataframe and calculated the category mapping for a selected categorical
       feature
    Inputs:
        train_df : Raw training dataframe
        train : processed training dataframe
        fldname : input field name
    Outputs:
        job2ix : Output dataframe that maps each category to its categorical code
    '''
    trn_actual = train_df.copy()
    trn_final  = train.copy()
    train_cats(trn_actual)
    jtcodes = trn_actual[fldname].cat.codes.values
    jtcat = trn_actual[fldname].cat.categories.values
    s = pd.Series(pd.Categorical.from_codes(jtcodes,jtcat))
    trn_final[f'{fldname}Actual'] = s
    job2ix = dict(trn_final.loc[:,[fldname,f'{fldname}Actual']].drop_duplicates().to_dict('split')['data'])
    return job2ix

def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))
