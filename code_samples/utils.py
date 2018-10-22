import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO
logger = getLogger(__name__)
LOG_FMT = "{asctime} | {levelname:<5s} | {name} | {message}"
basicConfig(level=INFO, format=LOG_FMT, style="{")


import pandas as pd
import numpy as np
from sklearn import metrics


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns



# ============
# File Loader
# ============
def read_csv(trajectory_id, *, logger=getLogger(__name__+".read_csv")):
    """
    Read Single Trajectory CSV

    Args.
    -----
    - trajectory_id: int, Trajectory id rangeing from 0 to 630

    Return.
    -------
    - np.ndarray, shape = [Timestep, Features,]
    - > 2nd dimention = [logitude, latitude, sun_azmis, su_elevaion, fleg_daytime, elapsed_time, clock, days]
    - > Clock: seconds from midnight (0:00)
    """
    #Load data
    assert (trajectory_id >= 0) and (trajectory_id <= 630), "Invalid trajctory id, it shoud be between 0-630, but got id={}".format(
        trajectory_id)
    filename = os.path.join("../train/", "{:0=3}.csv".format(trajectory_id))
    cols = ['lon','lat','azimus','elevation','daytime','elapsed','clock','days']    
    df = pd.read_csv(filename, names=cols).fillna(0)

    # Convert Clock (Time to integer)
    def convert_time_to_sec(x):
        """
        Args.
        -----
        - x: string, format="HH:MM:SS"
        """
        (h,m,s) = x.split(":")
        return int(h)*3600 + int(m)*60 + int(s)

    df["clock"] = df["clock"].apply(convert_time_to_sec)
    return np.nan_to_num(df.values), cols
    

def read_labels(ids=range(631)):
    path = os.path.join("../", "train_labels.csv")
    df_label = pd.read_csv(path, names=["labels"]).reset_index(drop=True)
    assert len(df_label) == 631, "Invalid length, train_labes should contain 631 labels, but got ={}".format(len(df_label))
    df_label = df_label.loc[ids,:]
    assert len(df_label) == len(ids), "Invalid length, train_label (after droped) should be {}, bug got len(df_labels)={}".format(len(ids), df_label.shape)
    return df_label.values


def load_data(id_list=[], mode="all"):
    """
    Args.
    ------
    - id_list: list,
    - mode: {train, test}

    Return.
    -------
    - 3D np.ndarray
    """
    if len(id_list) > 0:
       ids = id_list
    elif mode == "all":
        ids = range(0,631)       
    elif mode == "train":
        ids = range(0,500)
    elif mode == "test":
        ids = range(500,631)
    else:
        raise ValueError("Invalid Call [id_list={}, mode={}]".format(is_list, mode,))

    X_ret = []
    for _id in ids:
        X_tmp, cols_tmp = read_csv(_id)
        X_ret.append(np.array(X_tmp))
    labels = read_labels(ids=ids)
    return X_ret, labels, cols_tmp



# ==================
# Feature Extractor
# ==================
def feature_extraction_broadcast(X, func=None, cols=[]):
    """
    Args.
    -----
    - X: list of np.ndarray, 
    - func: Callback function (feature extractor)

    Return.
    -------
    - 
    """
    Y_ret = []
    for X_tmp in X:
        Y_tmp, cols_tmp = func(X_tmp[:,cols])
        Y_ret.append(Y_tmp)
    return Y_ret, cols_tmp


def feature_concatenate(X1,X2, *, logger=getLogger(__name__+".feature_concatenate")):
    """
    Args.
    -----
    - X1, X2: list of np.ndarray, which shape is [Timestep, features] 

    Return.
    -------
    - np.ndarray, whose last dimention are concatenate [X1 <= X2]
    """
    assert len(X1) == len(X2), "Length of inputs do not much [X1={}, X2={}]".format(len(X1), len(X2),)
    X_out = []
    for X1_tmp, X2_tmp in zip(X1, X2):
        logger.debug("X1_tmp={}".format(X1_tmp.shape))
        logger.debug("X2_tmp={}".format(X2_tmp.shape))
        X_tmp = np.concatenate([X1_tmp, X2_tmp], axis=1)
        X_out.append(X_tmp)
    return X_out



def get_speed_and_acc(X):
    """
    Args.
    -----
    - X: 3D np.ndarray, [longitude, latitude, elapsed_time]

    Return.
    -------
    - 2D np.ndarray, [Timestep, Featuers]
    - > Features = [Speed, Speed_diff, Accerelation, Acceleration_diff]
    """
    assert (len(X.shape) == 2) and (X.shape[1] == 3), "Invaild input, the dimention of X should be (None, 2,) but got {}".format(
        X.shape)

    X_tmp = X[1:] - X[:-1]
    logger.debug("X_tmp={}, [X={}]".format(X_tmp.shape, X.shape))
    X_tmp = np.insert(X_tmp, 0, np.zeros(3), axis=0)
    logger.debug("X_tmp={}".format(X_tmp.shape))

    # Speed
    X_speed = np.sqrt(X_tmp[:,0]**2 + X_tmp[:,1]**2)*111000 / (X_tmp[:,2] + 1E-10)
    X_speed_diff = np.insert(X_speed[1:] - X_speed[:-1], 0, 0, axis=0)
    logger.debug("X_speed={}, X_speed_diff={}".format(X_speed.shape, X_speed_diff.shape))
    # Accereation
    X_acc = X_speed / (X_tmp[:,2] + 1E-10)
    X_acc_diff = np.insert(X_acc[1:] - X_acc[:-1], 0, 0, axis=0)
    logger.debug("X_acc={}, X_acc_diff={}".format(X_acc.shape, X_acc_diff.shape))

    # Return
    logger.debug(X_speed.reshape((-1,1)).shape,)
    X_ret = np.concatenate([
        X_speed.reshape((-1,1)),
        X_speed_diff.reshape((-1,1)),
        X_acc.reshape((-1,1)),
        X_acc_diff.reshape((-1,1)),
    ], axis=1)
    cols = ["speed", "speed_diff","acc","acc_diff",]
    return np.nan_to_num(X_ret), cols
    
    


# ===================
# Feature Aggrigator
# ===================
def feature_aggrigator(X, config, *, logger=getLogger(__name__+".feature_aggrigator")):
    """
    Args.
    -----
    - X      : List of np.ndarray, np.ndarray(s) have the shape of [Timestep, feature]
    - config : List of Ditionary(X and aggigate function)
    e.g.    
    {
    "aggfunc": function witch can be applied to 1D array,
    "col": int, column index to use ()
    "col_name": name of this column,
    }

    Returns.
    --------
    - np.ndarray [Trajectory, features, 1]
    """
    df = []
    cols = [d["col_name"] for d in config]

    # Loop by User
    for i,X_tmp in enumerate(X):
        row = []
        for d in config:
            aggfunc, col, col_name = d["aggfunc"], d["col"], d["col_name"]
            val  = aggfunc(X_tmp[:,col])
            logger.debug("val={}".format(val))
            assert (not isinstance(val, list)), "Invalid combinaition of X_tmp={} and aggfunc={}".format(X_tmp.shape, aggfunc)
            row.append(val)
        df.append(row)

    df = pd.DataFrame(df, columns=cols)
    return df
        
        


# =============================
# Split Training and Test Data
# =============================
def split_train_test(df_X, Y, use_std=True):
    """
    Args.
    -----
    - df_X: pd.DataFrame, [Rows=users,Columns=features]
    - Y   : 1D np.ndarray, labels for all data

    Returns.
    --------
    - pd.DataFrame (Features and labels for each sample)
    - tuple, Training samples and correponding labels
    - tuple, Test samples and corresponding labels
    """
    assert (len(df_X)==631) and (len(Y)==631), "Invalid length, this function needs all trajectoy, but got df_X={}, Y={}".format(df_X.shape, Y.shape)
    idx_train, idx_test = range(0,500), range(500,631)
    # Standardize
    if use_std:
        df_X = df_X.apply(lambda x : ((x - x.mean())*1/x.std()+0),axis=0)
    X = df_X.values
    X_train, X_test = X[idx_train], X[idx_test]
    Y_train, Y_test = Y[idx_train].ravel(), Y[idx_test].ravel()
    df_X["true"] = Y
    return df_X.reset_index(drop=True), (X_train, Y_train), (X_test, Y_test)



# =======
# Scores
# =======
def scores(df_pred):
    df_scores = pd.DataFrame({
        "Precision": metrics.precision_score(df_pred["true"], df_pred["pred"], labels=[0,1], average=None),
        "Recall": metrics.recall_score(df_pred["true"], df_pred["pred"], labels=[0,1], average=None),
        "F1": metrics.f1_score(df_pred["true"], df_pred["pred"], labels=[0,1], average=None),
    }, index=["Male","Female"], columns=["Precision","Recall", "F1"]).T
    df_scores["All"] = [
        metrics.precision_score(df_pred["true"], df_pred["pred"], average='micro'),
        metrics.recall_score(df_pred["true"], df_pred["pred"], average='micro'),
        metrics.f1_score(df_pred["true"], df_pred["pred"], average='micro'),
    ]
    return df_scores




def draw_cmx(df_pred, *,  logger=getLogger(__name__+'.draw_cmx')):
    logger.info("Start: Draw CMX")
    y_true, y_pred = df_pred["true"], df_pred["pred"]
    
    # Maek CMX
    cmx_data = metrics.confusion_matrix(y_true, y_pred,)
    df_cmx = pd.DataFrame(cmx_data).rename(index={0:"Male", 1:"Female",}, columns={0:"Male", 1:"Female",})

    # Raw Count
    plt.figure(figsize=(4,4))
    sns.heatmap(df_cmx, annot=True,  cmap='Blues', cbar=True)
    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.tight_layout()
    plt.show()
    return df_cmx
