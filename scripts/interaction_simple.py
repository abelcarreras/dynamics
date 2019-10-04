from dynamics.potential import Potential, Potential_par, Surface_1D_analytic, Surface_2D_analytic, Surface_2D, Surface_1D
from dynamics.integration import MolecularDynamics
from dynamics.plot_data import *

import numpy as np
import matplotlib.pyplot as plt


def custom_function(x):
    return np.sin(x-1)


def interaction(x, k=0.01):
    return k * np.square(x)


pot_harm_0 = Surface_1D_analytic(np.square, relative=False, offset=0)
pot_harm_1 = Surface_1D_analytic(np.square, relative=False, offset=1)
pot_harm_i = Surface_1D_analytic(interaction, relative=True, offset=1)

masses = [10, 10]
complete_interaction = [{'function': pot_harm_0, 'coordinates': [0]},
                        {'function': pot_harm_1, 'coordinates': [1]},
                        {'function': pot_harm_i, 'coordinates': [0, 1]}]

potential_1 = Potential(complete_interaction)

i_positions = [0, 1]
i_velocities = [1, 1]

print ('initial energy {} '.format(potential_1.total_potential(i_positions)))
print ('gradient energy {} '.format(potential_1.partial_derivatives([0,0])))

initial_conditions = {'coordinates': i_positions,
                      'velocity': i_velocities,
                      'masses': masses}

md = MolecularDynamics(initial_conditions=initial_conditions,
                       potential=potential_1,
                       number_of_time_steps=5000,
                       time_step=0.001)

results = md.calculate_nve()


time = results['time']
coordinates = np.array(results['trajectory'])
total_energy = results['total_energy']
potential_energy = results['potential_energy']
kinetic_energy = results['kinetic_energy']
velocity = np.array(results['velocity'])

plt.figure()
plot_energy(time, total_energy, kinetic_energy, potential_energy)


plt.figure()
plot_trajectory(coordinates, potential_1, [0, 1], separated=True)
plt.figure()
plot_trajectory(coordinates, potential_1, [0, 1], separated=False)
plt.show()