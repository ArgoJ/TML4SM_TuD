"""
Tutorial Machine Learning in Solid Mechanics (WiSe 23/24)
Task 4: Viscoelasticity
==================
Authors: Dominik K. Klein
         
01/2024
"""


# %%   
"""Import modules"""
import seaborn as sns
import datetime
now = datetime.datetime.now

from keras import optimizers
from keras import losses
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
sns.set_style('darkgrid')


# %% 
"""Own modules"""
from src.plots import plot_data, plot_model_pred, plot_loss
from src.data import generate_data_harmonic, generate_data_relaxation
from src.models import RNNModel 


# %%   
"""Load and visualize data"""
E_infty = 0.5
E = 2
eta = 1

n = 100
omegas = [1]
As = [1]

eps, eps_dot, sig, dts = generate_data_harmonic(E_infty, E, eta, n, omegas, As)

plot_data(eps, eps_dot, sig, omegas, As)



# %%   
"""Load and evaluate model"""
model = RNNModel([32, 2], ['softplus', 'linear'], [False, False])
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.002),
    loss=losses.MeanSquaredError()
)

t1 = now()
print(t1)
h = model.fit([eps, dts], [sig], epochs = 100,  verbose = 2)
t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')
plot_loss(h.history['loss'], suptitle='training loss')

eps, eps_dot, sig, dts = generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = model([eps, dts])
plot_data(eps, eps_dot, sig, omegas, As)
plot_model_pred(eps, sig, sig_m, omegas, As)

As = [1,1,2]
omegas = [1,2,3]

eps, eps_dot, sig, dts = generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = model([eps, dts])
plot_data(eps, eps_dot, sig, omegas, As)
plot_model_pred(eps, sig, sig_m, omegas, As)


eps, eps_dot, sig, dts = generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = model([eps, dts])
plot_data(eps, eps_dot, sig, omegas, As)
plot_model_pred(eps, sig, sig_m, omegas, As)

# %%
