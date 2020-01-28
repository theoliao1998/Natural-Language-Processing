from logisticsRegression import *
from sklearn.metrics import f1_score

with open("X_train.txt",'r') as df:
    with open("y_train.txt",'r') as lf:
        Y = []
        words = {}
        for line,label in zip(df,lf):
            Y.append(int(label))
            for word in tokenize(line):
                if word not in words:
                    words[word] = len(words)
        
        Y = np.array(Y)


with open("X_train.txt",'r') as df:
    X = np.zeros((Y.shape[0],len(words)))
    for i,line in enumerate(df):
        for word in tokenize(line):
            X[i,words[word]] += 1

B, ll = logistic_regression(X,Y,0.25,400000,True,10000)

def visualize(ll,alpha):
  n = np.arange(len(ll))*10000
  fig, ax = plt.subplots()
  ax.plot(n, ll)
  ax.set(xlabel='step number', ylabel='log-likelihood',
      title=f'log-likelihood every step (l_r={alpha})')
  ax.grid()
  plt.show()

visualize(ll,0.25)


with open ("y_dev.txt",'r') as lf:
    desired = []
    for y in lf:
        desired.append(int(y))

with open ("X_dev.txt",'r') as df:
    data = []
    for line in df:
        x = np.zeros(len(words)+1)
        x[len(words)] = 1
        for word in tokenize(line):
            if word in words:
              x[words[word]] += 1
        data.append(predict(x,B))

print(str(f1_score(desired,data)))

with open ("X_test.txt",'r') as df:
    with open("lr_predict.csv",'w') as lf:
        lf.write("Id,Category\n")
        for i,line in enumerate(df):
            x = np.zeros(len(words)+1)
            x[len(words)] = 1
            for word in tokenize(line):
                if word in words:
                    x[words[word]] += 1
            
            lf.write(str(i) + "," + str(predict(x,B)) + "\n")

