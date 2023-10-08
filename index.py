import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental import preprocessing
# from tensorflow.keras import regularizers
# #from sklearn import preprocessing
# from sklearn.model_selection import train_test_split

cropData=pd.read_csv('HistoricalCropData.csv').sample(frac=1)
print(cropData.to_string())

cropData.dtypes