from naiveBayes import *


PX, PY, PXY = train("X_train.txt", "y_train.txt")


with open("X_test.txt",'r') as df:
    with open("prediction.csv",'w') as lf:
        lf.write("Id,Category\n")
        for i,line in enumerate(df):
            lf.write(str(i) + "," + str(classify(tokenize(line),PX,PY,PXY)) + "\n")
