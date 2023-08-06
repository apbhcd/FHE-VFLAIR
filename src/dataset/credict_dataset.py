import numpy as np 
import pandas as pd 
import matplotlib.pyplot as mat
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

def remove_outliers(data,column):
    lower = 0
    upper = 90
    outliers_removed = data[data[column].between(lower, upper)]
    return outliers_removed

    
def load_GiveMeSomeCredit():
    print("******loading GiveMeSomeCredit******")
    train_df = pd.read_csv("/home/yaopan/Documents/FHE-VFLAIR/data/GiveMeSomeCredit/data/cs-training.csv", index_col=0) 
    test_df  = pd.read_csv("/home/yaopan/Documents/FHE-VFLAIR/data/GiveMeSomeCredit/data/cs-test.csv", index_col=0)
    submission = pd.read_csv('/home/yaopan/Documents/FHE-VFLAIR/data/GiveMeSomeCredit/data/sampleEntry.csv')
    train_df = train_df[train_df['RevolvingUtilizationOfUnsecuredLines'] <= 10]
    outlier_removed_train_df = remove_outliers(train_df,'NumberOfTimes90DaysLate')
    outlier_removed_train_df = remove_outliers(outlier_removed_train_df,'NumberOfTime30-59DaysPastDueNotWorse')
    outlier_removed_train_df = remove_outliers(outlier_removed_train_df,'NumberOfTime60-89DaysPastDueNotWorse')
    train_df = outlier_removed_train_df
    scaler = MinMaxScaler()
    train_df_normalize = pd.DataFrame(scaler.fit_transform(train_df), columns = train_df.columns)
    train_df_normalize.head()
    imputer = KNNImputer()
    imputer.fit(train_df_normalize)
    train_df_no_null_values = imputer.transform(train_df_normalize)
    train_data = pd.DataFrame(train_df_no_null_values)
    x = train_df.drop("SeriousDlqin2yrs", axis=1)
    y = train_df["SeriousDlqin2yrs"]
    print("******Finished loading GiveMeSomeCredit******")
    return x.values, y.values
