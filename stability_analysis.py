import numpy as np
from numpy import abs, sqrt, pi, sin, cos
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'


def get_min_h(theta0, alpha, step_ratio = 0.5):
    """
        Get minimum value of function h(w) on the interval [1/4, 4].
    """

    # Helper constants
    a1 = cos(alpha)
    a2 = sin(alpha)
    a3 = cos(alpha + pi/4)
    sqrt2 = sqrt(2)

    # Upper limit for the derivative for w in [1/4, 4]
    derivative_limit = 500 + abs(theta0)*(256*a1+32*a2+16*sqrt2*abs(a3))

    w = 1/4
    min_h = np.Inf

    while w < 4:
        # Get value of function h(w)
        h = w**4 - 4*(w*w*w)*sin(a1*w*theta0) + 4*(w*w) - \
            2*(w*w)*cos(a2*w*theta0) + 4*w*sin(sqrt2*w*theta0*a3) + 1

        # Get new w_safe such that h(w_safe) >= 0
        w_safe = w + h / derivative_limit

        # Update minimum value of function h on the interval
        minimum_potential_h = h * (1-step_ratio)
        if minimum_potential_h < min_h:
            min_h = minimum_potential_h

        # Move towards w_safe, according to specified step ratio
        w = w + (w_safe - w) * step_ratio

    return min_h


def get_theta_max(alpha, theta0, minimum_step = 1e-2):
    """
        Get maximum value for parameter theta (for given fixed alpha), such that
        the system remains stable. The stopping criterion is determined by the
        minimum_step.
    """

    # Helper constants
    c = 1/(sqrt(2) * (32*cos(alpha) + 4*sin(alpha)))

    delta_theta = np.Inf
    theta = theta0
    delta_theta_on_outer_interval = 1 / sqrt(2) * 0.1

    while delta_theta > minimum_step:
        minimum_h = get_min_h(theta, alpha)   

        delta_theta1 = c * sqrt(minimum_h)


        if delta_theta1 < delta_theta_on_outer_interval:
            delta_theta = delta_theta1
        else:
             delta_theta = delta_theta_on_outer_interval

        theta = theta + delta_theta

    return theta


def get_delay_margins(alpha, minimum_step = 1e-2):
    """
        Get maximum delay margins tau1 and tau2 in the direction alpha.
    """

    theta0 = 0
    theta_max = get_theta_max(alpha, theta0, minimum_step)

    tau1 = cos(alpha) * theta_max
    tau2 = sin(alpha) * theta_max

    return tau1, tau2


def graph_stability_region(step_count=60, minimum_step=1e-3):
    """
        Graph stability region in tau1/tau2 plane.
    """
    
    # Setup
    alpha_min = 0
    alpha_max = pi/2
    alpha_vector = np.linspace(alpha_min, alpha_max, step_count)
    delay_margins = np.empty((2, step_count))
    
    # Calculate delay margins in selected directions alpha
    for i in range(step_count):
        tau1, tau2 = get_delay_margins(alpha_vector[i], minimum_step)
        delay_margins[:, i] = np.array([tau1, tau2])
        print(f"tau1 = {tau1}\ntau2 = {tau2}")
    
    # Prepare ax    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlabel=r'$\tau_1$', ylabel=r'$\tau_2$')
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3.5)

    # Plot lines
    for i in range(step_count):
        tau1 = delay_margins[0, i]
        tau2 = delay_margins[1, i]
        N = 2
        x_vector = np.linspace(0, tau1, N)
        y_vector = np.linspace(0, tau2, N)
        ax.plot(x_vector, y_vector, 'g')

    # Plot end points
    ax.scatter(delay_margins[0,:], delay_margins[1,:])

    # Display the plot (use savefig function to save to a file instead)
    plt.show()


if __name__ == "__main__":
    """
        Run an example, obtaining the stability region plot.
    """

    graph_stability_region()