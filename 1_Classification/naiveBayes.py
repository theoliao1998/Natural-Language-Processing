from collections import defaultdict, Counter
from sklearn.metrics import f1_score
from math import log
import re
import random
import matplotlib.pyplot as plt


def tokenize(s):
    return s.split()


def train(data_file, label_file, smoothing_alpha = 0, tokenize_func = tokenize):

    neg_pos_words = Counter(), Counter()
    neg_pos_cnt = [0, 0]
    PX = defaultdict(float)
    PXY = set()

    with open(data_file,'r') as df:
        with open(label_file,'r') as lf:
            for line, label in zip(df, lf):
                label = int(label)
                for word in tokenize_func(line):
                    PX[word] += 1
                    PXY.add(word)
                    neg_pos_cnt[label] += 1
                    neg_pos_words[label][word] += 1
    
    PY = neg_pos_cnt[1] / (neg_pos_cnt[0]+neg_pos_cnt[1])
    PY = (1-PY, PY)

    cnt = neg_pos_cnt[0] + neg_pos_cnt[1]
    for word in PX:
        PX[word] = PX[word] / cnt

    V = len(PXY)
    PXY = defaultdict(lambda: (smoothing_alpha/(neg_pos_cnt[0] + V*smoothing_alpha),\
        smoothing_alpha/(neg_pos_cnt[1] + V*smoothing_alpha)),\
        {word: ((neg_pos_words[0][word] + smoothing_alpha) / (neg_pos_cnt[0] + V * smoothing_alpha),\
        (neg_pos_words[1][word] + smoothing_alpha) / (neg_pos_cnt[1] + V * smoothing_alpha)) for word in PXY})

    return PX, PY, PXY

def classify(tokenized_doc, PX, PY, PXY):
    p0, p1 = 1, 1

    for x in set(tokenized_doc):
        # if x not in PXY: # uncomment to ignore words not in the vocabulary
        #     continue
        p0 *= PXY[x][0]
        p1 *= PXY[x][1]
    
    if p0 == 0 and p1 == 0:
        return 0

    p0, p1 = PY[0] * p0 / (PY[0] * p0 + PY[1] * p1), PY[1] * p1 / (PY[0] * p0 + PY[1] * p1)

 
    return 0 if p0 >= p1 else 1

# PX, PY, PXY = train("X_train.txt", "y_train.txt", 0)
# with open ("X_dev.txt",'r') as df:
#     with open ("y_dev.txt",'r') as lf:
#         data = []
#         desired = []
#         for x,y in zip(df,lf):
#             data.append(classify(tokenize(x),PX,PY,PXY))
#             desired.append(int(y))
# print(f1_score(desired,data))

# smoothing-alpha vs. F1-score value
# res = []
# for i in range(21):
#     alpha = 0.1* i
#     PX, PY, PXY = train("X_train.txt", "y_train.txt", alpha, tokenize)
#     with open ("X_dev.txt",'r') as df:
#         with open ("y_dev.txt",'r') as lf:
#             data = []
#             desired = []
#             for x,y in zip(df,lf):
#                 data.append(classify(tokenize(x),PX,PY,PXY))
#                 desired.append(int(y))
#     res.append(f1_score(desired,data))
#     print(str(alpha) + "," + str(f1_score(desired,data)))

# plt.plot([0.1 * i for i in range(21)], res)
# plt.show()

# PX, PY, PXY = train("X_train.txt", "y_train.txt")
# print(PX["many"])

# with open("X_test.txt",'r') as df:
#     with open("predict3.csv",'w') as lf:
#         lf.write("Id,Category\n")
#         for i,line in enumerate(df):
#             lf.write(str(i) + "," + str(classify(tokenize(line),PX,PY,PXY)) + "\n")


# with open ("X_dev.txt",'r') as df:
#     with open ("y_dev.txt",'r') as lf:
#         data = []
#         desired = []
#         for x,y in zip(df,lf):
#             data.append(classify(tokenize(x),PX,PY,PXY))
#             desired.append(int(y))

# print(f1_score(desired,data))






    

    
