# Modified Euler Hohmann Transfer
# Explicit Trapezoidal Method
# This calculates the position and velocity of a spacecraft
# as it travels from one circular orbit to another.
# Units: km, seconds, km/s

import numpy as np
import matplotlib.pyplot as plt
import time


# Equations of motion due to Earth's gravity
# u contains x, y, vx, and vy
# The function gives us how position and velocity change over time
def orbit(u, mu):
    x, y, vx, vy = u
    r = np.sqrt(x**2 + y**2)

    ax = -mu*x/r**3
    ay = -mu*y/r**3

    return np.array([vx, vy, ax, ay])


# Run the Modified Euler method for the transfer orbit
# N is the number of time steps
def modified_euler(mu, r1, r2, N):
    a = (r1 + r2)/2
    T = np.pi*np.sqrt(a**3/mu)
    h = T/N

    t = np.linspace(0, T, N+1)
    u = np.zeros((N+1, 4))

    # Start at the first orbit with the velocity after the first burn
    vt1 = np.sqrt(mu*(2/r1 - 1/a))
    u[0] = [r1, 0, 0, vt1]

    # Calculate the next position and velocity at each time step
    for i in range(N):
        k1 = orbit(u[i], mu)
        k2 = orbit(u[i] + h*k1, mu)

        # Modified Euler averages the slope at the start
        # and the predicted end of the time step
        u[i+1] = u[i] + h*(k1 + k2)/2

    return t, u


def main():
    # Same inputs as RK4 so the comparison is fair
    mu = 398600.4418
    r1 = 7000
    r2 = 14000
    N = 200

    # Calculate the time needed for the Hohmann transfer
    a = (r1 + r2)/2
    T = np.pi*np.sqrt(a**3/mu)

    # Circular orbit speeds and speeds along the transfer ellipse
    vc1 = np.sqrt(mu/r1)
    vc2 = np.sqrt(mu/r2)
    vt1 = np.sqrt(mu*(2/r1 - 1/a))
    vt2 = np.sqrt(mu*(2/r2 - 1/a))

    # Ideal changes in velocity for the two burns
    dv1 = vt1 - vc1
    dv2 = vc2 - vt2
    total_dv = abs(dv1) + abs(dv2)

    # Time only the numerical integration
    start = time.perf_counter()
    t, u = modified_euler(mu, r1, r2, N)
    runtime = time.perf_counter() - start

    # Compare the numerical arrival state to the exact Hohmann arrival
    position = u[-1, 0:2]
    velocity = u[-1, 2:4]

    exact_position = np.array([-r2, 0])
    exact_velocity = np.array([0, -vt2])

    # Full position error includes both x and y errors
    pos_error = np.linalg.norm(position - exact_position)

    # Radius error only checks the distance from Earth's center
    radius_error = abs(np.linalg.norm(position) - r2)

    # Compare the calculated velocity to the expected transfer velocity
    vel_error = np.linalg.norm(velocity - exact_velocity)

    # Find the correction needed to reach the ideal final velocity
    # This is an error measurement, not an actual simulated engine burn
    desired_velocity = np.array([0, -vc2])
    dv2_num = np.linalg.norm(desired_velocity - velocity)
    dv2_error = abs(dv2_num - abs(dv2))

    # Apply the ideal second burn in the local tangential direction
    # We calculate the new velocity, but do not simulate another orbit
    x, y = position
    r = np.sqrt(x**2 + y**2)

    vx_after = velocity[0] + dv2*(-y/r)
    vy_after = velocity[1] + dv2*(x/r)

    radial_velocity = (x*vx_after + y*vy_after)/r

    print('\nMODIFIED EULER HOHMANN TRANSFER')
    print('------------------------------')
    print('Transfer time (s):', T)
    print('First burn (km/s):', dv1)
    print('Second burn (km/s):', dv2)
    print('Total delta-v (km/s):', total_dv)

    print('\nERRORS')
    print('Position error (km):', pos_error)
    print('Radius error (km):', radius_error)
    print('Velocity error (km/s):', vel_error)
    print('Second burn error (km/s):', dv2_error)
    print('Radial velocity after burn (km/s):', radial_velocity)

    print('\nPERFORMANCE')
    print('Runtime (s):', runtime)
    print('Function evaluations:', 2*N)

    # Repeat the simulation using different numbers of steps
    # If Modified Euler works as expected, the error should decrease with h^2
    steps = [25, 50, 100, 200, 400, 800]

    h_values = []
    pos_errors = []
    radius_errors = []
    vel_errors = []
    orders = []
    runtimes = []
    evaluations = []

    print('\nMODIFIED EULER CONVERGENCE STUDY')
    print('Steps     Step Size     Position Error     Order')

    for n in steps:
        start = time.perf_counter()
        times, states = modified_euler(mu, r1, r2, n)
        run_time = time.perf_counter() - start

        h = T/n

        p_error = np.linalg.norm(states[-1, 0:2] - exact_position)
        r_error = abs(np.linalg.norm(states[-1, 0:2]) - r2)
        v_error = np.linalg.norm(states[-1, 2:4] - exact_velocity)

        h_values.append(h)
        pos_errors.append(p_error)
        radius_errors.append(r_error)
        vel_errors.append(v_error)
        runtimes.append(run_time)
        evaluations.append(2*n)

        # Estimate how quickly the error decreases when h is cut in half
        if len(pos_errors) == 1:
            order = np.nan
        else:
            order = np.log(pos_errors[-2]/pos_errors[-1])/np.log(2)

        orders.append(order)

        print(f'{n:5d}    {h:9.3f}    {p_error:14.6e}    {order:6.3f}')

    # Save the convergence values so we can compare them to RK4
    results = np.column_stack((
        steps, h_values, pos_errors,
        radius_errors, vel_errors, orders,
        runtimes, evaluations
    ))

    np.savetxt(
        'me2_convergence.csv',
        results,
        delimiter=',',
        header='steps,h_seconds,position_error_km,radius_error_km,velocity_error_km_s,order,runtime_seconds,function_evaluations',
        comments=''
    )

    # Save all calculated positions and velocities over time
    trajectory = np.column_stack((t, u))

    np.savetxt(
        'me2_trajectory.csv',
        trajectory,
        delimiter=',',
        header='time_s,x_km,y_km,vx_km_s,vy_km_s',
        comments=''
    )

    # Plot the starting orbit, ending orbit, and calculated transfer path
    theta = np.linspace(0, 2*np.pi, 300)

    plt.figure()
    plt.plot(r1*np.cos(theta), r1*np.sin(theta),
             '--', label='Initial orbit')
    plt.plot(r2*np.cos(theta), r2*np.sin(theta),
             '--', label='Final orbit')
    plt.plot(u[:, 0], u[:, 1], label='Modified Euler transfer')
    plt.plot(-r2, 0, 'ko', label='Exact arrival')

    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title('Hohmann Transfer - Modified Euler')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('me2_orbit.png', dpi=150)

    # Plot the position error against step size to check convergence
    plt.figure()
    plt.loglog(h_values, pos_errors, 'o-',
               label='Modified Euler error')

    reference = pos_errors[-1]*(np.array(h_values)/h_values[-1])**2
    plt.loglog(h_values, reference, '--',
               label='Second-order reference')

    plt.xlabel('Step size (s)')
    plt.ylabel('Position error (km)')
    plt.title('Modified Euler Convergence')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('me2_convergence.png', dpi=150)

    plt.show()


# Only run the full study when this file is opened directly
# This lets Monte Carlo import modified_euler() without opening plots
if __name__ == '__main__':
    main()
