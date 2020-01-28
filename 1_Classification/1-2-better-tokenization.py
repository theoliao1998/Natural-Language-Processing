from naiveBayes import *

stopwords = ["and","a","an","i","they","on","with","as","it","were"]

def better_tokenize(s):
    s = s.lower()

    s = re.sub(r' (http:.+) ',r' ',s)
    s = re.sub(r' @\w+ ',r' ',s)
    s = re.sub(r'(\'s |\'m |\'re )',r' ',s)
    s = re.sub(r'([a-z0-9])([\,\.] )',r'\1 ',s)
    s = re.sub(r'([a-z0-9])([\!\?] )',r'\1 \2 ',s)
    
    return [w for w in s.split() if w not in stopwords]

res = []
for i in range(41):
    alpha = 0.05* i
    PX, PY, PXY = train("X_train.txt", "y_train.txt", alpha, better_tokenize)
    with open ("X_dev.txt",'r') as df:
        with open ("y_dev.txt",'r') as lf:
            data = []
            desired = []
            for x,y in zip(df,lf):
                data.append(classify(better_tokenize(x),PX,PY,PXY))
                desired.append(int(y))
    res.append(f1_score(desired,data))
    # print(str(alpha) + "," + str(f1_score(desired,data)))

plt.plot([0.05 * i for i in range(41)], res)
plt.xlabel("smoothing_alpha")
plt.ylabel("f1 score")
plt.show()