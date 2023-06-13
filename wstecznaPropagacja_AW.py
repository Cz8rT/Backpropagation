import numpy as np
from siec_neuronowa.siec_neuronowa import Siec_neuronowa


# Zbiór danych dla przykładu XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

siec_neuronowa_1 = Siec_neuronowa([2, 2, 1], alfa=0.15, beta=1)
siec_neuronowa_1.pokaz_wagi()
siec_neuronowa_1.ucz_siec(X, Y, liczba_epok=10000)
siec_neuronowa_1.testuj(X, Y)
