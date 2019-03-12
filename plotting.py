import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio



# def plot_for_offset(power, y_max):
#     # Data for plotting
#     t = np.arange(0.0, 100, 1)
#     s = t**power
#
#     fig, ax = plt.subplots(figsize=(10,5))
#     ax.plot(t, s)
#     ax.grid()
#     ax.set(xlabel='X', ylabel='x^{}'.format(power),
#            title='Powers of x')
#
#     # IMPORTANT ANIMATION CODE HERE
#     # Used to keep the limits constant
#     ax.set_ylim(0, y_max)
#
#     # Used to return the plot as an image rray
#     fig.canvas.draw()       # draw the canvas, cache the renderer
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#     image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#     return image



def plot_pulse(gdd, tod):

    print(gdd, tod)

    tmax = 100e-15
    N = 256
    dt = (2 * tmax) / N
    t = dt * np.arange(-N/2, N/2, 1)
    df = 1 / (dt * N)
    f = df * np.arange(-N/2, N/2, 1)

    f0 = 12e13
    Ef = np.exp(-(f - f0)**2 / (3e13)**2)
    phase = gdd * 1000e-30 * (f - f0)**2 + tod * 1000e-45 * (f - f0)**3

    Efprop = Ef * np.exp(1j * phase)
    Etprop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Efprop)))

    fig, ax = plt.subplots(1,2, figsize=(6,3))
    fig.subplots_adjust(wspace=0.2, left=0.1, right=0.9)


    # plotting time domain
    ax[0].plot(t, np.real(Etprop), color="blue")
    ax[0].plot(t, np.abs(Etprop), color="black", linestyle="dashed", alpha=0.7)
    ax[0].set_ylim(-0.05, 0.05)
    ax[0].set_xlim(np.min(t), np.max(t))
    ax[0].set_xlabel(r"Time $\rightarrow$")
    ax[0].set_ylabel("Electric Field")
    ax[0].set_title("Electric Field in Time")
    ax[0].yaxis.label.set_color("blue")
    ax[0].set_yticks([])
    ax[0].set_xticks([])


    # plotting freq domain
    ax[1].plot(f, np.abs(Efprop)**2, color="black")
    axtwin = ax[1].twinx()
    ax[1].set_xlim(0.5e14, 1.9e14)
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].set_ylabel("Intensity")
    ax[1].set_xlabel(r"Frequency $\rightarrow$")
    ax[1].set_title("Spectrum")
    ax[1].set_yticks([])
    ax[1].set_xticks([])


    # axtwin.plot(f, np.unwrap(np.angle(Efprop)), color="green")
    axtwin.plot(f, phase, color="green")
    axtwin.set_ylim(-15, 30)
    axtwin.yaxis.label.set_color("green")
    axtwin.tick_params(axis="y", colors="green")
    axtwin.set_ylabel("Phase [rad]")


    # plt.show()
    # exit(0)


    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image




kwargs_write = {'fps':1.0, 'quantizer':'nq'}
gdd = np.append(np.linspace(0, 5.0, 10), np.linspace(5.0, 0.0, 10))
gdd = np.append(gdd, np.array(-1 * gdd))
gdd = np.append(gdd, np.zeros_like(gdd))

tod = np.append(np.linspace(0, 60.0, 10), np.linspace(60.0, 0.0, 10))
tod = np.append(tod, np.array(-1 * tod))
tod = np.append(tod, np.zeros_like(tod))
tod = np.flip(tod)



imageio.mimsave('./powers.gif', [plot_pulse(gdd_i, tod_i) for gdd_i, tod_i in zip(gdd, tod)], fps=10)


