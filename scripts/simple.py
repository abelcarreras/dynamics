from dynamics.potential import Potential, Potential_par, Surface_1D_analytic, Surface_2D_analytic, Surface_2D, Surface_1D
from dynamics.integration import MolecularDynamics
from dynamics.plot_data import *

import numpy as np
import matplotlib.pyplot as pl

import dynamics.iodata as iodata


def custom_function(x):
    return -np.cos(x) + 1


pot_sine = Surface_1D_analytic(np.sin, relative=False)
pot_cosine = Surface_1D_analytic(np.cos, relative=False)
pot_custom = Surface_1D_analytic(custom_function, relative=True)


num_rotors = 5

mass = [10, 1, 1, 1, 1]

complete_interaction = []
masses = []
for i in range(num_rotors):
    j = np.roll(range(num_rotors), -i)
    # complete_interaction.append({'function': pot_cosine, 'coordinates': [j[0]]})
    if i < num_rotors-1:
        print j[0], j[1]
        complete_interaction.append({'function': pot_custom, 'coordinates': [j[0], j[1]]})
    masses.append(mass[i])

potential_1 = Potential(complete_interaction)

i_positions = [0, 0, 0, 0, 0]
i_vel = [10, 0, 0, 0, 0]

print ('initial energy {} '.format(potential_1.total_potential(i_positions)))

initial_conditions = {'coordinates': i_positions,
                      'velocity': i_vel,
                      'masses': masses}

md = MolecularDynamics(initial_conditions=initial_conditions,
                       potential=potential_1,
                       number_of_time_steps=500,
                       time_step=0.01,
                       temperature=400)

#results = md.calculate_nvt()
results = md.calculate_nve()


time = results['time']
coordinates = np.array(results['trajectory'])
total_energy = results['total_energy']
potential_energy = results['potential_energy']
kinetic_energy = results['kinetic_energy']
velocity = np.array(results['velocity'])

pl.figure()
plot_energy(time, total_energy, kinetic_energy, potential_energy)

pl.figure()
plot_trajectory(coordinates, potential_1, [0, 1, 2, 3, 4], separated=True)

pl.figure()
plot_trajectory(coordinates, potential_1, [0, 4], separated=False)

pl.show()


# print total_energy
# np.savetxt('trajectory', coordinates, header='positions')
# np.savetxt('velocity', velocity, header='velocity')
# np.savetxt('energy', np.array([time, total_energy, potential_energy, kinetic_energy]).T,
#             header='time(ps)\t total(kcal/mol)\t potential(kcal/mol)\t kinetic (kcal/mol)')
