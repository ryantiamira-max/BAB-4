# Exercise 4.2 - Modifikasi Example 4.2 menjadi 3 input (logika OR 3 input)
from sklearn.neural_network import MLPClassifier
X = [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.],
     [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]]
y = [0, 1, 1, 1, 1, 1, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
print(clf.predict([[2., 2., 2.], [-1., -2., -1.]]))
print([coef.shape for coef in clf.coefs_])
