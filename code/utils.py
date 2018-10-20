import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO
logger = getLogger(__name__)
LOG_FMT = "{asctime} | {levelname:<5s} | {name} | {message}"
basicConfig(level=DEBUG, format=LOG_FMT, style="{")


import pandas as pd
import numpy as np


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
    assert (trajectory_id >= 0) and (trajectory_id < 631), "Invalid trajctory id, it shoud be between 0-630, but got id={}".format(
        trajectory_id)
    filename = os.path.join("../train/", "{:0=3}.csv".format(trajectory_id))
    df = pd.read_csv(filename, names=['lon','lat','azimus','elevation','daytime','elapsed','clock','days']).fillna(0)

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
    return df.values
    



# ==================
# Feature Extractor
# ==================
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
    X_speed = np.sqrt(X_tmp[:,0]**2 + X_tmp[:,1]**2)*111000 / X_tmp[:,2]
    X_speed_diff = np.insert(X_speed[1:] - X_speed[:-1], 0, 0, axis=0)
    logger.debug("X_speed={}, X_speed_diff={}".format(X_speed.shape, X_speed_diff.shape))
    # Accereation
    X_acc = X_speed / X_tmp[:,2]
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
    return X_ret
    
    
