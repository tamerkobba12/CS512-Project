import time                 # provides timing for benchmarks
import numpy as np          # provides complex math and array functions
from sklearn import svm	    # provides Support Vector Regression
import csv
import math
import sys
from pathlib import Path

DATA_FOLDER = Path(__file__).resolve().parents[2] / "data" / "processed"

class DataFromFile:
    #------------------------------------------------------------------------------
    def getTwoDecPoint(self, x):
        return float("%.2f"%x)
    #------------------------------------------------------------------------------
    def placeDataIntoArray(self, fileName):
        with open(fileName, mode='r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
            dataArray = np.array([row for row in datareader], dtype=np.float64, order='C')

        if (min(dataArray.shape) == 1): # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray
    #------------------------------------------------------------------------------
    def getAllOfTheData(self):
        TrainX = self.placeDataIntoArray(DATA_FOLDER / 'Train-Data.csv')
        TrainY = self.placeDataIntoArray(DATA_FOLDER / 'Train-pIC50.csv')
        ValidateX = self.placeDataIntoArray(DATA_FOLDER / 'Validation-Data.csv')
        ValidateY = self.placeDataIntoArray(DATA_FOLDER / 'Validation-pIC50.csv')
        TestX = self.placeDataIntoArray(DATA_FOLDER / 'Test-Data.csv')
        TestY = self.placeDataIntoArray(DATA_FOLDER / 'Test-pIC50.csv')
        return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY
    #------------------------------------------------------------------------------
    def rescaleTheData(self, TrainX, ValidateX, TestX):
        TrainXVar = TrainX.var(axis = 0, ddof=1)
        TrainXMean = TrainX.mean(axis = 0)

        for i in range(0, TrainX.shape[0]):
            TrainX[i,:] = (TrainX[i,:] - TrainXMean)/np.sqrt(TrainXVar)
        for i in range(0, ValidateX.shape[0]):
            ValidateX[i,:] = (ValidateX[i,:] - TrainXMean)/np.sqrt(TrainXVar)
        for i in range(0, TestX.shape[0]):
            TestX[i,:] = (TestX[i,:] - TrainXMean)/np.sqrt(TrainXVar)

        return TrainX, ValidateX, TestX
    #------------------------------------------------------------------------------