import json

input = []
output = []

with open('output') as data:
    for line in data:
        result = json.loads(line)
        expected = int(1000 * result['output'][0])
        if expected != 999 and expected != -999:
            input.append(result['input'])
            output.append(expected)

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(max_iter=20000, tol=1e-9, activation='logistic')
regressor.fit(input[:-1], output[:-1])

import pickle

print(pickle.dumps(regressor))
