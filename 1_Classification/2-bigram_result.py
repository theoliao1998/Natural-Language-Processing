from logisticsRegression import *
from sklearn.metrics import f1_score

with open("X_train.txt",'r') as df:
    with open("y_train.txt",'r') as lf:
        Y_2 = []
        words_2 = {}
        for line,label in zip(df,lf):
            Y_2.append(int(label))
            tokens = tokenize(line)
            for word in zip(tokens[:-1],tokens[1:]):
                if word not in words_2:
                    words_2[word] = len(words_2)
        
        Y_2 = np.array(Y_2)


with open("X_train.txt",'r') as df:
    X_2 = np.zeros((Y_2.shape[0],len(words_2)))
    for i,line in enumerate(df):
        tokens = tokenize(line)
        for word in zip(tokens[:-1],tokens[1:]):
            X_2[i,words_2[word]] += 1

B2, t2 = logistic_regression(X_2,Y_2,0.25,500000,True,1000)

def visualize(ll,alpha):
  n = np.arange(len(ll))*1000
  fig, ax = plt.subplots()
  ax.plot(n, ll)
  ax.set(xlabel='step number', ylabel='log-likelihood',
      title=f'log-likelihood every step (l_r={alpha})')
  ax.grid()
  plt.show()

visualize(t2,0.25)

with open ("y_dev.txt",'r') as lf:
    desired = []
    for y in lf:
        desired.append(int(y))

with open ("X_dev.txt",'r') as df:
    data = []
    for line in df:
        x = np.zeros(len(words_2)+1)
        x[len(words_2)] = 1
        tokens = tokenize(line)
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
            tokens = tokenize(line)
            for word in zip(tokens[:-1],tokens[1:]):
                if word in words_2:
                    x[words_2[word]] += 1
            
            lf.write(str(i) + "," + str(predict(x,B2)) + "\n")