from logisticsRegression import *
from sklearn.metrics import f1_score

import re

stopwords = []

def logistic_regression(X,Y,learning_rate,num_step,compute_ll=False,interval=1):
    X = np.append(X,np.ones((X.shape[0],1)),1)
    B = np.zeros(X.shape[1])
    if compute_ll:
        ll = [log_likelihood(B,X,Y)]

    for step in range(num_step):
        i = randrange(Y.shape[0])
        B += learning_rate * compute_gradient(X[i],Y[i],sigmoid(np.inner(B,X[i])))
        if compute_ll and (step+1) % interval == 0:
            ll.append(log_likelihood(B,X,Y))

    return B,ll if compute_ll else B

def better_tokenize(s):
    # s = s.lower()

    # s = re.sub(r' (http:.+) ',r' http ',s)
    # s = re.sub(r' @\w+ ',r' @user ',s)
    # s = re.sub(r'(\'s )',r' is ',s)
    # s = re.sub(r'(\'s |\'m |\'re )',r' ',s)
    s = re.sub(r'([a-z0-9])([\,\.] )',r'\1 \2',s)
    s = re.sub(r'([a-z0-9])([\!\?] )',r'\1 \2 ',s)
    
    return ["<start>"] + [w for w in s.split() if w not in stopwords] + ["<end>"]

with open("X_train.txt",'r') as df:
    with open("y_train.txt",'r') as lf:
        Y_2 = []
        words_2 = {}
        for line,label in zip(df,lf):
            Y_2.append(int(label))
            tokens = better_tokenize(line)
            for word in zip(tokens[:-1],tokens[1:]):
                if word not in words_2:
                    words_2[word] = len(words_2)
        
        Y_2 = np.array(Y_2)


with open("X_train.txt",'r') as df:
    X_2 = np.zeros((Y_2.shape[0],len(words_2)))
    for i,line in enumerate(df):
        tokens = better_tokenize(line)
        for word in zip(tokens[:-1],tokens[1:]):
            X_2[i,words_2[word]] += 1

B2, t2 = logistic_regression(X_2,Y_2,0.25,500000)

# B2, t2 = logistic_regression(X_2,Y_2,0.25,500000,True,1000)

# def visualize(ll,alpha):
#   n = np.arange(len(ll))*1000
#   fig, ax = plt.subplots()
#   ax.plot(n, ll)
#   ax.set(xlabel='step number', ylabel='log-likelihood',
#       title=f'log-likelihood every step (l_r={alpha})')
#   ax.grid()
#   plt.show()

# visualize(t2,0.25)

with open ("y_dev.txt",'r') as lf:
    desired = []
    for y in lf:
        desired.append(int(y))

with open ("X_dev.txt",'r') as df:
    data = []
    for line in df:
        x = np.zeros(len(words_2)+1)
        x[len(words_2)] = 1
        tokens = better_tokenize(line)
        for word in zip(tokens[:-1],tokens[1:]):
            if word in words_2:
              x[words_2[word]] += 1
        data.append(predict(x,B2))
            
print(str(f1_score(desired,data)))

with open ("X_test.txt",'r') as df:
    with open("lr_predict_2.csv",'w') as lf:
        lf.write("Id,Category\n")
        for i,line in enumerate(df):
            x = np.zeros(len(words_2)+1)
            x[len(words_2)] = 1
            tokens = better_tokenize(line)
            for word in zip(tokens[:-1],tokens[1:]):
                if word in words_2:
                    x[words_2[word]] += 1
            
            lf.write(str(i) + "," + str(predict(x,B2)) + "\n")