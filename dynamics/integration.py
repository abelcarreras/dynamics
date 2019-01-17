import numpy as np


class MolecularDynamics:
    def __init__(self,
                 initial_conditions,
                 potential,
                 number_of_time_steps=100,
                 time_step=0.001,
                 q_thermo=1,
                 temperature=300,
                 units=None,
                 kB=0.0019872041 # kcal/(mol*K)
                 ):

        if units == None:
            print "Default units"
            # Default Units: Energy=kcal/mol, mass=amu, distance=Amstrong, time=ps
            kB=0.0019872041 # kcal/(mol*K)
            initial_conditions['masses'] = initial_conditions['masses'] * 0.00239117  # For now all unit conversion is set to the mass
            n_dim=len(initial_conditions['coordinates'])
            q_thermo = n_dim*kB*temperature

        #intial parameters
        self.time_step = time_step
        self.q = q_thermo
        self.T = temperature
        self.kB = kB

        self.initial_conditions = initial_conditions
        self.potential = potential

        self.degrees_of_freedom = len(initial_conditions['coordinates'])
        self.number_of_time_steps = number_of_time_steps

    def calculate_nve(self):

        pot_obj = self.potential
        time_step = self.time_step


        #initial consitios
        coor = [self.initial_conditions['coordinates']]
        vel = [self.initial_conditions['velocity']]
        mass = self.initial_conditions['masses']

        time = [0]
        ac = [0]

        kinetic_energy = [1 / 2. * np.sum(np.multiply(mass, vel[0] ** 2))]
        potential_energy = [pot_obj.total_potential(coor[0])]
        total_energy = [potential_energy[0] + kinetic_energy[0]]

        for i in range(self.number_of_time_steps):
            #Verlet algorithm
            coor.append(coor[-1] + vel[-1] * time_step + 1 / 2. * ac[-1] * time_step ** 2)
            v_n = np.array(vel[-1] + 1 / 2. * (ac[-1]) * time_step)
            dv = pot_obj.partial_derivatives(coor[-1])
            ac.append(-np.multiply(1./mass, dv))

            vel.append(np.array(v_n + 1 / 2. * (ac[-1]) * time_step))

            potential_energy.append(pot_obj.total_potential(coor[-1]))
            kinetic_energy.append(1 / 2. * np.sum(np.multiply(mass, vel[-1]**2)))
            total_energy.append(potential_energy[-1] + kinetic_energy[-1])
            time.append(time[-1] + time_step)

        results = {'trajectory': coor,
                   'velocity': vel,
                   'time': time,
                   'total_energy': total_energy,
                   'kinetic_energy': kinetic_energy,
                   'potential_energy': potential_energy}

        return results

    def calculate_nvt(self, compute_nose=False):

        time_step = self.time_step
        q = self.q
        kB = self.kB
        pot_obj = self.potential
        T = self.T

        time = [0]
        ac = [0]
        friction = [0]
        temp = [0]

        coor = [self.initial_conditions['coordinates']]
        vel = [self.initial_conditions['velocity']]
        mass = self.initial_conditions['masses']

        kinetic_energy = [1 / 2. * np.sum(np.multiply(mass, vel[0] ** 2))]
        potential_energy = [pot_obj.total_potential(coor[0])]
        total_energy = [potential_energy[0] + kinetic_energy[0]]

        if compute_nose:
            s = [0]
            h_nose = [potential_energy[-1] + kinetic_energy[-1] + friction[-1]**2 * q/2.0 + (self.degrees_of_freedom+1) * kB * T * s[-1]]

        for i in range(self.number_of_time_steps):
            coor.append(coor[-1] + vel[-1] * time_step + 1 / 2. * (ac[-1] - friction[-1]*vel[-1])* time_step ** 2)
            v_n = np.array(vel[-1] + 1 / 2. * (ac[-1] - friction[-1]*vel[-1]) * time_step)

            if compute_nose:
                s_n = (s[-1] + friction[-1]*time_step/2)

            dv = pot_obj.partial_derivatives(coor[-1])
            ac.append(-np.multiply(1/mass, dv))
            friction_n = friction[-1] + time_step/(4*q) * (np.sum(np.multiply(mass, vel[-1]**2)) - (self.degrees_of_freedom+1)*kB*T)
            friction.append(friction_n + time_step/(4*q) * (np.sum(np.multiply(mass, v_n**2)) - (self.degrees_of_freedom+1)*kB*T))
            vel.append(np.array(v_n + 1 / 2. * (ac[-1]) * time_step)/(1+time_step/2*friction[-1]))

            if compute_nose:
                s.append(s_n + friction[-1]*time_step/2)

            potential_energy.append(pot_obj.total_potential(coor[-1]))
            kinetic_energy.append(1 / 2. * np.sum(np.multiply(mass, vel[-1]**2)))
            total_energy.append(potential_energy[-1] + kinetic_energy[-1])
            time.append(time[-1] + time_step)
            if compute_nose:
                h_nose.append(potential_energy[-1] + kinetic_energy[-1] + friction[-1]**2 * q/2.0 + (self.degrees_of_freedom+1) * kB * T * s[-1])

            temp.append(kinetic_energy[-1] * 2 / (kB * (self.degrees_of_freedom+1)))

        results = {'trajectory': coor,
                   'velocity': vel,
                   'time': time,
                   'total_energy': total_energy,
                   'kinetic_energy': kinetic_energy,
                   'potential_energy': potential_energy}

    #    print results.keys()

        if compute_nose:
            results.update({'nose_hamiltonian': h_nose,
                            'thermostat_variable': s})
        return results

if __name__ == "__main__":

    from potential import Potential, Surface_1D_analytic

    pot_surface3 = Surface_1D_analytic(np.cos, relative=True)

    complete_interaction = [{'function': pot_surface3,  'coordinates': [0, 1]},
                        {'function': pot_surface3, 'coordinates': [1, 2]},
                        {'function': pot_surface3,  'coordinates': [2, 3]},
                        {'function': pot_surface3, 'coordinates': [3, 4]}]


    pot_obj = Potential(complete_interaction)

    initial_conditions = {'coordinates': np.array([4, 0, 0, 0, 0]),
                          'velocity': np.array([3, 0, 0, 0, 0]),
                          'masses': np.array([300.0, 1.0, 1.0, 1.0 ,1.0])}

    md = MolecularDynamics(initial_conditions=initial_conditions,
                           potential=pot_obj)

    md.calculate_nve()
    md.calculate_nvt()

