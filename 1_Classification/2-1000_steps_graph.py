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

B1, ll1 = logistic_regression(X,Y,0.05,1000,True)
B0, ll0 = logistic_regression(X,Y,5e-5,1000,True)
B2, ll2 = logistic_regression(X,Y,5e-8,1000,True)
print(len(ll0))
n = np.arange(len(ll0))
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(n, ll1,'b',label='l_r = 0.05')
ax.plot(n, ll0,'r',label='l_r = 5e-5')
ax.plot(n, ll2,'k',label='l_r = 5e-8')

ax.set(xlabel='step number', ylabel='log-likelihood',
    title=f'log-likelihood every step')
plt.legend(loc=2)
# plt.draw()
plt.show()


