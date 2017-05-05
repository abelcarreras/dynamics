from potential import Potential, Potential_par, Surface_1D_analytic, Surface_2D_analytic, Surface_2D, Surface_1D
from integration import MolecularDynamics
import numpy as np
import iodata



# Initial random positions function
def random_initial(num_rotors, potential, diff=2):
    i_positions = np.array([45.0]*num_rotors)
    for i in range(num_rotors):
        t_positions = i_positions
        ini_pot = potential.total_potential(t_positions)
        while True:
            t_positions[i]=np.random.random()*360
            pot = potential.total_potential(t_positions)
            print i, pot, ini_pot, pot - ini_pot

            if pot - ini_pot < diff:
                break

        i_positions = t_positions
    return i_positions


# Define interpolated functions
x, y, data = iodata.read_data('potential_hh.pot')
pot_surface_hh = Surface_2D(x, y, data, boundary=[[0, 360], [0, 360]])

x, y, data = iodata.read_data('potential_ff.pot')
pot_surface_ff = Surface_2D(x, y, data, boundary=[[0, 360], [0, 360]])

x, y, data = iodata.read_data('potential_hf.pot')
pot_surface_hf = Surface_2D(x, y, data, boundary=[[0, 360], [0, 360]])

x, y, data = iodata.read_data('potential_fh.pot')
pot_surface_fh = Surface_2D(x, y, data, boundary=[[0, 360], [0, 360]])

x, data = np.loadtxt('potential_h.pot').T
pot_curve_h = Surface_1D(x, data, boundary=[[0, 360]])

x, data = np.loadtxt('potential_f.pot').T
pot_curve_f = Surface_1D(x, data, boundary=[[0, 360]])

mass_h = 89.57079
mass_f = 296.5215

num_rotors = 10

type_list = np.random.choice(2, num_rotors, p=[1.0, 0.0])

print type_list

complete_interaction = []
masses = []
for i in range(num_rotors):

    j = np.roll(range(num_rotors), -i)

    print '{} {} : {} {}'.format(j[0], j[1], type_list[j[0]], type_list[j[1]])
    if type_list[j[0]] == 0 and type_list[j[1]] == 0:
        pot_surface = pot_surface_hh
        pot_curve = pot_curve_h
        mass = mass_h

    if type_list[j[0]] == 0 and type_list[j[1]] == 1:
        pot_surface = pot_surface_hf
        pot_curve = pot_curve_h
        mass = mass_h

    if type_list[j[0]] == 1 and type_list[j[1]] == 0:
        pot_surface = pot_surface_fh
        pot_curve = pot_curve_f
        mass = mass_f

    if type_list[j[0]] == 1 and type_list[j[1]] == 1:
        pot_surface = pot_surface_ff
        pot_curve = pot_curve_f
        mass = mass_f

    complete_interaction.append({'function': pot_curve, 'coordinates': [j[0]]})
    complete_interaction.append({'function': pot_surface, 'coordinates': [j[0], j[1]]})
    masses.append(mass)


print masses

np.savetxt('type', np.array([type_list, masses]).T)

#print masses
#exit()


# Set the potential topology
#complete_interaction = [{'function': pot_surface_hf, 'coordinates': [0, 1]},
#                        {'function': pot_surface_fh, 'coordinates': [1, 2]},
#                        {'function': pot_surface_hf, 'coordinates': [2, 3]},
#                        {'function': pot_surface_fh, 'coordinates': [3, 4]}]

potential1 = Potential(complete_interaction)

if False:
    for i in range(0, 360, 10):
        for j in range(0, 360, 10):
            a = np.array([0.0]*num_rotors)
            a[0] = i
            a[1] = j
            print i, j, potential1.total_potential(a)
        print

#exit()

i_positions = random_initial(num_rotors, potential1, diff=0)
print ('initial energy {} '.format(potential1.total_potential(i_positions)))

import matplotlib.pyplot as pl
pl.hist(i_positions, bins=50)
pl.show()

initial_conditions = {'coordinates': i_positions,
                      'velocity': np.random.random(num_rotors)*1,
                      'masses': np.array(masses)}

i_vel = np.array([0]*num_rotors)
i_positions = np.array([37]*num_rotors)
#i_vel[0] = 1
#masses[0] = 10000

initial_conditions = {'coordinates': i_positions,
                      'velocity': i_vel,
                      'masses': np.array(masses)}

md = MolecularDynamics(initial_conditions=initial_conditions,
                       potential=potential1,
                       number_of_time_steps=500000,
                       time_step=0.01,
                       temperature=400)

#results = md.calculate_nvt()
results = md.calculate_nve()

#print results['trajectory']
#results = md.calculate_nvt(compute_nose=False)

#from plot_data import plot_trajectory, plot_energy

#plot_energy(results['time'],
#            results['total_energy'],
          # results['nose_hamiltonian'],
#            results['kinetic_energy'],
#            results['potential_energy'])

time = results['time']
coordinates = np.array(results['trajectory'])
total_energy = results['total_energy']
potential_energy = results['potential_energy']
kinetic_energy = results['kinetic_energy']
velocity = np.array(results['velocity'])

#print total_energy
np.savetxt('trajectory', coordinates, header='positions')
np.savetxt('velocity', velocity, header='velocity')
np.savetxt('energy', np.array([time, total_energy, potential_energy, kinetic_energy]).T,
           header='time(ps)\t total(kcal/mol)\t potential(kcal/mol)\t kinetic (kcal/mol)')
