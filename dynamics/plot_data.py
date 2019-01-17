import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UnicodeWarning)
# from potential import interaction_function_periodic as potential_function


def plot_energy(time, total_energy, kinetic_energy, potential_energy):

    plt.title('Energy')
    plt.plot(time, total_energy, label='Total')
    plt.plot(time, kinetic_energy, label='Kinetic')
    plt.plot(time, potential_energy, label='Potential')
    plt.legend()


def plot_trajectory(coor, potential_function, index, show_surface=False, separated=False):

    if show_surface:
        x = np.linspace(-50, 400, 100)
        y = np.linspace(-50, 400, 100)

        X, Y = np.meshgrid(x, y)
        try:
            Z = [[potential_function.total_potential(np.array([x1, y1]))[0] for x1 in x] for y1 in y]
        except IndexError:
            Z = [[potential_function.total_potential(np.array([x1, y1])) for x1 in x] for y1 in y]

        levels = [0, 10, 20, 30, 40, 50]
        CS = plt.contour(X, Y, Z, levels, colors='k')

        plt.clabel(CS, inline=0, fontsize=1)

    if isinstance(index, int):
        #path = np.array(coor)[index]
        plt.plot(np.array(coor).T[index])

    else:
        if separated:
            for i, path in zip(index, np.array(coor).T[index]):
                plt.plot(path, label='traj {}'.format(i))
            plt.legend()
        else:
            path = np.array(coor).T
            plt.plot(path[index[0]], path[index[1]])


    plt.title('Trajectory {}'.format(index))
