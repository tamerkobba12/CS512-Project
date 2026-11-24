import time  # provides timing for benchmarks
import numpy as np          # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
from sklearn import linear_model
from sklearn import neural_network
from sklearn import cross_decomposition  #provides Partial Least Square Regression
import csv
import math
import sys
import pandas as pd      # reads the output file again at the end (pickBestModel)

# Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

# ------------------------------------------------------------------------------
# The numbers the template comments ask for, kept together so they are easy to find
MIN_FEATURES = 5        # getAValidrow: a row needs at least 5 features set to 1 ...
MAX_FEATURES = 15       # ... and at most 15
PARENT_RANKS = (0, 1)   # createANewPopulation: MOM = best row, DAD = second best row
NUM_CHILDREN = 2        # two children by one point cross over (they go to rows 3 and 4)
NUM_MUTATIONS = 1       # mutations per child (the slides say mutation is optional)
# The slides use a different recipe: parents = 3rd and 4th best, children fill all the
# other 48 rows. To use it, set PARENT_RANKS = (2, 3) and NUM_CHILDREN = 48.
# ------------------------------------------------------------------------------

class FitnessAnalyzer:
    def __init__(self):
        self.fitnessdata = FromFinessFileMLR.FitnessResults()
    # ------------------------------------------------------------------------------
    def getAValidrow(self, numOfFea, eps=0.015):
        # In the parameters, numOfFea refers to the number of columns in the data. 
        # This number is total number of your columns. In this routune, you need to
        # select a random number between 0 and 1: {r = np.random.uniform(0, 1)}.
        # If the value is less that or equal 0.015 (1.5% of the features) you can
        # set that index to 1; otherwise, the index is set to zero.
        # After going through the entire features (bits), you can add all the numbers
        # together to see if at least 5 features and at most 15 features are set to 1.
        # If that is the case you can return the vector; otherwise, you need to do it again.

        while True:
            # one random number between 0 and 1 for every feature (bit); 1 if <= eps
            V = (np.random.uniform(0, 1, numOfFea) <= eps).astype(int)
            if MIN_FEATURES <= V.sum() <= MAX_FEATURES:
                return V
    # ------------------------------------------------------------------------------
    def Create_A_Population(self, numOfPop, numOfFea):
        # this routine creates the first poprlation. We want to have 50 rows
        # (50 population) and number of features is the number of your columns.
        # Since this the first population, all 50 rows should be randomly selected. 
        # Therefore, for each row, you need to call the getAValidrow function 
        # and move the vector into the row. Once the first population is created
        # you can return it to Callee
 
        population = np.zeros((numOfPop, numOfFea), dtype=int)
        for i in range(numOfPop):
            population[i] = self.getAValidrow(numOfFea)
        return population
    # ------------------------------------------------------------------------------
    # The following creates an output file. Every time a model is created the
    # descriptors of the model, the name of the model (ex: "MLR" for multiple
    # linear regression of "SVM" support vector machine), the R^2 of training, Q^2
    # of training,R^2 of validation, and R^2 of test is placed in the output file

    def createAnOutputFile(self):
        file_name = None
        algorithm = None

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if ((file_name == None) and (algorithm != None)):
            file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__,
                                                    algorithm.model.__class__.__name__, algorithm.gen_max, timestamp)
        elif file_name == None:
            file_name = "{}.csv".format(timestamp)
        fileOut = open(file_name, 'w', newline='')
        self.fileOut = fileOut          # kept so main() can close the file at the end
        self.fileName = file_name       # kept so pickBestModel can read the file again at the end
        fileW = csv.writer(fileOut)

        fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

        return fileW
    # -------------------------------------------------------------------------------------------
    def onePointCrossover(self, mom, dad):
        # cut both parents at the same random place and swap the pieces
        cut = np.random.randint(1, len(mom))                # a place between 1 and numOfFea - 1
        child1 = np.concatenate((mom[:cut], dad[cut:]))     # mom's left part + dad's right part
        child2 = np.concatenate((dad[:cut], mom[cut:]))     # dad's left part + mom's right part
        return child1, child2
    # -------------------------------------------------------------------------------------------
    def mutate(self, child):
        # switch one feature: half the time one that is ON goes OFF, otherwise one that is OFF
        # goes ON (a plain random flip would almost always turn a feature on, most entries are 0)
        child = child.copy()
        for _ in range(NUM_MUTATIONS):
            selected = np.nonzero(child == 1)[0]
            notSelected = np.nonzero(child == 0)[0]
            if np.random.uniform(0, 1) < 0.5 and len(selected) > 1:
                child[np.random.choice(selected)] = 0
            else:
                child[np.random.choice(notSelected)] = 1
        return child
    # -------------------------------------------------------------------------------------------
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
        #   NewPopulation = create a 2D array of (50 rows by num of features (columns))
        #   Find the row with the best fitness in the OldPopulation and the
        #   row with the second best fitness in the OldPopulation and move them
        #   to row 1 and row 2 of the NewPopulation. Call the first one MOM
        #   and the second one DAD and create two children based on one point cross over.
        #   The two children go to position 3 and 4 of the NewPopulation. for the other 46
        #   rows of the NewPopulation do as you did before which means, call the
        #   getAValidrowcall for each of them
      
        NewPopulation = np.zeros((numOfPop, numOfFea), dtype=int)
        best = np.argsort(fitness, kind="stable")        # row numbers from best fitness to worst

        # the best and the second best row move to row 1 and row 2
        NewPopulation[0] = OldPopulation[best[0]]
        NewPopulation[1] = OldPopulation[best[1]]

        # MOM and DAD make the children (one point cross over) for the next rows
        MOM = OldPopulation[best[PARENT_RANKS[0]]]
        DAD = OldPopulation[best[PARENT_RANKS[1]]]
        row = 2
        lastChildRow = 2 + NUM_CHILDREN
        while row < lastChildRow:
            child1, child2 = self.onePointCrossover(MOM, DAD)
            for child in (child1, child2):
                if row < lastChildRow:
                    NewPopulation[row] = self.mutate(child)
                    row = row + 1

        # all the other rows: getAValidrow, as for the first population
        for row in range(lastChildRow, numOfPop):
            NewPopulation[row] = self.getAValidrow(numOfFea)

        return NewPopulation
    # -------------------------------------------------------------------------------------------
    def pickBestModel(self, title, R2req_train, Q2req_train, R2req_validate, R2req_test):
        # Reads the output file again and prints the best model: the lowest fitness among the
        # models whose R^2 of training, Q^2 of training, R^2 of validation and R^2 of test are
        # all at least the requirement (and below 1). The same model is written again in many
        # generations, so each model is counted once; the file shows the fitness with only
        # 2 decimals, so when several models tie, the one with the fewest descriptors is shown.
        table = pd.read_csv(self.fileName)
        scores = table[["R2", "Q2", "R2Pred_Validation", "R2Pred_Test"]]
        needed = [R2req_train, Q2req_train, R2req_validate, R2req_test]

        # every requirement for every model at once (and every value below 1)
        ok = ((scores >= needed).all(axis=1) & (scores < 1).all(axis=1))
        good = table[ok].drop_duplicates(subset="Descriptor ID")

        print()
        print(title)
        if good.empty:
            print("   no model meets these minimums ({} rows in the file)".format(len(table)))
            return None
        # lowest fitness first; among equal fitness, the fewest descriptors
        good = good.assign(NumDescriptors=good["Descriptor ID"].str.count(",") + 1)
        best = good[good["Fitness"] == good["Fitness"].min()].sort_values("NumDescriptors", kind="stable")
        row = best.iloc[0]
        print("   {} rows in the file, {} different models qualify".format(len(table), len(good)))
        print("   best fitness {} ({} models tied; the one with the fewest descriptors is shown)".format(row["Fitness"], len(best)))
        print("   descriptors ({}): {}".format(row["NumDescriptors"], row["Descriptor ID"]))
        print("   R2 train {}, Q2 {}, R2 validation {}, R2 test {}".format(
              row["R2"], row["Q2"], row["R2Pred_Validation"], row["R2Pred_Test"]))
        return row["Descriptor ID"]
    # --------------------------------------------------------------------------------------------
    def PerformOneThousandIteration(self, numOdPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
        NumOfGenerations = 1
        OldPopulation = population
        while (NumOfGenerations < 1000):
            population = self.createANewPopulation(numOdPop, numOfFea, OldPopulation, fitness)
            fittingStatus, fitness = self.fitnessdata.validate_model(model, fileW, population,
                                            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
            OldPopulation = population      # breed the next generation from this one (without this line
                                            # the first population would be used every time)
            NumOfGenerations = NumOfGenerations + 1
            print(NumOfGenerations, "best fitness:", round(fitness.min(), 4))
        return
    # --------------------------------------------------------------------------------------------

def main():
    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    np.random.seed(42)    # same seed -> same results every run (remove for a different run each time)
    model = mlr.MLR()
    #model = svm.LinearSVR()
    #model = cross_decomposition.PLSRegression()

    filedata = FromDataFileMLR.DataFromFile()           # constructor for FromDataFileMLR class
    fitnessdata = FromFinessFileMLR.FitnessResults()    # constructor for FromFitnessFileMLR object
    analyzer = FitnessAnalyzer()                        # constructor for MainMLR class

    # create an output file. Name the object to be FileW
    fileW = analyzer.createAnOutputFile()

    # Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50
    numOfFea = 3800 # this is total number of your columns (features); it is set from the data file below

    # we continue exhancing the model; however if after 1000 iteration no
    # enhancement is done, we can quit
    unfit = 1000

    # Final model requirements: The following is used to evaluate each model. The minimum
    # values for R^2 of training should be 0.6, R^2 of Validation should be 0.5 and R^2 of
    # test should be 0.5
    # We ask for 0.6 on all four values (R^2 of training, Q^2 of training, R^2 of validation and
    # R^2 of test). That is stricter than the numbers above (0.6 / 0.5 / 0.5) and than the slides (0.5).
    R2req_train = .6
    R2req_validate = .6
    R2req_test = .6
    Q2req_train = .6

    # getAllOfTheData is in FromDataFileMLR file. The following places the data
    # (training data, validation data, and test data) into associated matrices
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = filedata.getAllOfTheData() # gets descriptor values and target values
    # Note that rescalling is already done in your cleaning section and you do not need to do it again
    # TrainX, ValidateX, TestX = filedata.rescaleTheData(TrainX, ValidateX, TestX)
    numOfFea = TrainX.shape[1]      # the number of columns (features) in the data file

    fittingStatus = unfit
    population = analyzer.Create_A_Population(numOfPop, numOfFea)
    fittingStatus, fitness = fitnessdata.validate_model(model, fileW, population,
                                            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    analyzer.PerformOneThousandIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    analyzer.fileOut.close()
    print("Done. Every model that was scored is in the file", analyzer.fileName)

    # the final model: the lowest fitness among the models that meet the requirements above
    best = analyzer.pickBestModel("Best model (lowest fitness) with R2 train >= {}, Q2 >= {}, R2 validation >= {}, R2 test >= {}:".format(
                                  R2req_train, Q2req_train, R2req_validate, R2req_test),
                                  R2req_train, Q2req_train, R2req_validate, R2req_test)
    if best is None:        # nothing reaches that: use the slides' bar, all four at least 0.5
        analyzer.pickBestModel("Best model (lowest fitness) with all four at least 0.5 (the slides):", .5, .5, .5, .5)


# main routine ends in here
# ------------------------------------------------------------------------------
main()
# ------------------------------------------------------------------------------