# Exercise 4.1 - Modifikasi Example 4.1 agar melakukan logika OR
# Tabel kebenaran OR untuk urutan input: [0,0] [1,1] [1,0] [0,1]
#   OR  -> [0, 1, 1, 1]
#   (AND -> [0, 1, 0, 0], ada di baris komentar di bawah sebagai pembanding)

from numpy import exp, array, random, dot

#Set up parameters ===============================================
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])
outputs = array([[0, 1, 1, 1]]).T          # logika OR
#outputs = array([[0, 1, 0, 0]]).T         # logika AND
random.seed(1)
weights = 2 * random.random((2, 1)) -1

#Define a Single Neuron function =================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

#Train the Neuron ================================================
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

#Test the Neuron: tampilkan seluruh tabel OR =====================
for x in inputs:
    print(x, '->', neuron(x, weights))
