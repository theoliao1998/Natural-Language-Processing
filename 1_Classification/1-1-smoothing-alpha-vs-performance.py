from naiveBayes import *

res = []
for i in range(41):
    alpha = 0.05* i
    PX, PY, PXY = train("X_train.txt", "y_train.txt", alpha, tokenize)
    with open ("X_dev.txt",'r') as df:
        with open ("y_dev.txt",'r') as lf:
            data = []
            desired = []
            for x,y in zip(df,lf):
                data.append(classify(tokenize(x),PX,PY,PXY))
                desired.append(int(y))
    res.append(f1_score(desired,data))
    # print(str(alpha) + "," + str(f1_score(desired,data)))

plt.plot([0.05 * i for i in range(41)], res)
plt.xlabel("smoothing_alpha")
plt.ylabel("f1 score")
plt.show()