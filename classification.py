
# coding: utf-8

#モジュールの読み込み
import sys 
import math
import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.layers.core import Dense, Activation
from util import ShowConfmat

import uuid