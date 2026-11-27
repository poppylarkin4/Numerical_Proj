# Monte Carlo Testing - RK4 vs Modified Euler
# Run the same random Hohmann transfers using both methods
# This lets us compare the errors over different orbital conditions

import numpy as np
import matplotlib.pyplot as plt
import time

from rk4 import rk4
from modified_euler import modified_euler

# Use the same values as our original simulations
mu = 398600.4418
N = 200
trials = 500

# Set a seed so we get the same random cases every time we run the code
np.random.seed(42)

# Store the inputs and results for all 500 cases
r1_values = []
r2_values = []
rk4_errors = []
me_errors = []
rk4_times = []
me_times = []


for i in range(trials):

    # Change the starting and ending orbit radii slightly each time
    # These are example ranges, not actual measured uncertainties
    r1 = np.random.uniform(6900, 7100)
    r2 = np.random.uniform(13800, 14200)

    # The exact arrival for each ideal Hohmann transfer is (-r2, 0)
    exact_position = np.array([-r2, 0])

    # Run RK4 and record how long the integration takes
    start = time.perf_counter()
    t1, u1 = rk4(mu, r1, r2, N)
    rk4_time = time.perf_counter() - start

    # Run Modified Euler with exactly the same inputs
    start = time.perf_counter()
    t2, u2 = modified_euler(mu, r1, r2, N)
    me_time = time.perf_counter() - start

    # Compare each numerical arrival position to the exact arrival point
    rk4_error = np.linalg.norm(u1[-1, 0:2] - exact_position)
    me_error = np.linalg.norm(u2[-1, 0:2] - exact_position)

    # Save this trial so we can look at the overall results afterward
    r1_values.append(r1)
    r2_values.append(r2)

    rk4_errors.append(rk4_error)
    me_errors.append(me_error)

    rk4_times.append(rk4_time)
    me_times.append(me_time)


# Convert lists to arrays so we can calculate averages and other statistics
rk4_errors = np.array(rk4_errors)
me_errors = np.array(me_errors)

rk4_times = np.array(rk4_times)
me_times = np.array(me_times)


# Show the typical error and runtime for each method
print('\nMONTE CARLO RESULTS')
print('------------------------------')
print('Number of trials:', trials)
print('Steps per trial:', N)

print('\nRK4')
print('Mean position error (km):', np.mean(rk4_errors))
print('Median position error (km):', np.median(rk4_errors))
print('Maximum position error (km):', np.max(rk4_errors))
print('Standard deviation (km):', np.std(rk4_errors))
print('Median runtime (s):', np.median(rk4_times))

print('\nMODIFIED EULER')
print('Mean position error (km):', np.mean(me_errors))
print('Median position error (km):', np.median(me_errors))
print('Maximum position error (km):', np.max(me_errors))
print('Standard deviation (km):', np.std(me_errors))
print('Median runtime (s):', np.median(me_times))

print('\nCOMPARISON')
print('Trials with lower RK4 error:', np.sum(rk4_errors < me_errors))
print('Trials with lower Modified Euler error:', np.sum(me_errors < rk4_errors))


# Save the orbital radii, errors, and runtimes from each trial
results = np.column_stack((
    r1_values, r2_values, rk4_errors,
    me_errors, rk4_times, me_times
))

np.savetxt(
    'monte_carlo_results.csv',
    results,
    delimiter=',',
    header='r1_km,r2_km,rk4_error_km,me_error_km,rk4_runtime_s,me_runtime_s',
    comments=''
)


# Show how the arrival error changes for each of the 500 transfers
plt.figure()
plt.semilogy(rk4_errors, label='RK4')
plt.semilogy(me_errors, label='Modified Euler')

plt.xlabel('Trial number')
plt.ylabel('Arrival position error (km)')
plt.title('Monte Carlo - Position Error by Trial')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('monte_carlo_trials.png', dpi=150)


# Compare the spread of errors for both methods
# Separate graphs keep the smaller RK4 errors visible
plt.figure()
plt.hist(rk4_errors, bins=20, alpha=0.7, label='RK4')

plt.xlabel('RK4 arrival position error (km)')
plt.ylabel('Number of trials')
plt.title('Monte Carlo - RK4 Error Distribution')
plt.grid()
plt.tight_layout()
plt.savefig('monte_carlo_rk4_errors.png', dpi=150)


plt.figure()
plt.hist(me_errors, bins=20, alpha=0.7, label='Modified Euler')

plt.xlabel('Modified Euler arrival position error (km)')
plt.ylabel('Number of trials')
plt.title('Monte Carlo - Modified Euler Error Distribution')
plt.grid()
plt.tight_layout()
plt.savefig('monte_carlo_me2_errors.png', dpi=150)

plt.show()
