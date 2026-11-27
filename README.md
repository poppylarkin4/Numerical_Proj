Hohmann Transfer - RK4 vs Modified Euler

Overview:

For our Numerical Methods project, we're looking at how RK4 and Modified Euler compare when simulating a Hohmann transfer. Basically, we want to see how accurately each method can predict a spacecraft's position and velocity as it moves from one orbit to another, and how much computation each method needs to do that.

We started by writing separate codes for RK4 and Modified Euler, then made sure both were using the same initial conditions and calculations so we could compare them fairly. From there, we added convergence testing to see how the errors change as we increase the number of steps, along with runtime and function evaluations to compare efficiency.

We also added Monte Carlo testing so we could see how both methods perform over different orbital conditions instead of just testing one transfer.

Files:

rk4.py

This code uses the fourth-order Runge-Kutta method to simulate the Hohmann transfer.

We start by calculating the transfer time and the velocity changes needed to go from the initial orbit to the final orbit. The code then uses RK4 to calculate the spacecraft's position and velocity at each time step as it travels along the transfer ellipse.

Once it reaches the end of the transfer, we compare the numerical position and velocity to the theoretical values to see how accurate the method was.

The code also runs a convergence study where we change the number of time steps and see how the errors decrease. We record the runtime and function evaluations for each case and save the results as CSV files and plots.

modified_euler.py

This code does essentially the same thing as the RK4 code, except we're using Modified Euler instead.

Both codes use the same equations of motion, initial conditions, and error calculations. The difference is how they calculate the next position and velocity.

RK4 uses four slopes per time step, while Modified Euler uses two. Because of this, we expect RK4 to be more accurate, but Modified Euler should require fewer calculations per step.

We run the same convergence study for Modified Euler so we can compare the results directly.

monte_carlo.py

For Monte Carlo testing, we wanted to see how the methods would perform if we changed the orbital conditions.

The code runs 500 different Hohmann transfers by randomly changing the initial and final orbital radii. For every trial, both RK4 and Modified Euler use the exact same inputs, so we're comparing the integration methods rather than different transfer conditions.

We record the final position errors and runtimes for both methods. After all the trials are finished, the code calculates the mean, median, maximum, and standard deviation of the errors.

We also generate plots showing how the errors change between trials and how they're distributed for each method.

Initial Conditions:

For our main simulations, we're using:

Earth's gravitational parameter-

398600.4418 km³/s²

Initial orbital radius-

7000 km

Final orbital radius-

14000 km

Number of steps-

200

We use the vis-viva equation to calculate the initial transfer velocity. From there, the numerical methods calculate the spacecraft's changing position and velocity throughout the transfer.

All distances are in kilometers, time is in seconds, and velocity is in km/s.

Numerical Methods:

Both methods use the same equations of motion, which account for Earth's gravitational acceleration.

For RK4, we calculate four slopes at each time step and use them to estimate the spacecraft's next position and velocity. RK4 is a fourth-order method, so when we cut the time-step size in half, we expect the error to decrease by about 16 times.

For Modified Euler, we calculate two slopes and average them to get the next position and velocity. Since it's a second-order method, we expect the error to decrease by about 4 times when we halve the step size.

The main thing we're looking at is whether the additional calculations in RK4 are worth the improvement in accuracy.

Convergence Study:

To check how well each method works, we run both simulations using different numbers of steps:

25, 50, 100, 200, 400, 800

For each one, we calculate the final position error, radius error, velocity error, runtime, and number of function evaluations.

We also calculate the convergence order to see whether RK4 and Modified Euler follow their expected fourth-order and second-order behavior.

This helps us see how much the accuracy improves as we make the time steps smaller.

Monte Carlo Testing:

For Monte Carlo testing, we run 500 simulations where we randomly change the orbital radii.

We use these ranges:

Initial orbital radius: 6900–7100 km

Final orbital radius: 13800–14200 km

We keep the number of steps at 200 and use the same orbital conditions for both methods in every trial.

The goal is to see whether the accuracy difference between RK4 and Modified Euler stays consistent when we change the transfer conditions.

How to Run:

The codes use Python along with NumPy and Matplotlib.

If you don't have the libraries installed, run:

pip install numpy matplotlib

Make sure rk4.py, modified_euler.py, and monte_carlo.py are all in the same folder.

To run each code:

python rk4.py
python modified_euler.py
python monte_carlo.py

Each program prints its results in the terminal and saves its graphs and CSV files.

The Monte Carlo code uses the RK4 and Modified Euler functions from the other two files, so those files need to be in the same folder.

Results and Outputs:

Both the RK4 and Modified Euler codes generate a plot of the Hohmann transfer and a convergence plot showing how the position error changes with the time-step size.

They also save CSV files containing the spacecraft's position and velocity throughout the transfer, along with the convergence results.

The Monte Carlo code saves the results from all 500 trials and generates graphs comparing the position errors for both methods.

We can use these results to compare accuracy, convergence, runtime, and the amount of computation each method requires.

Assumptions:

For this project, we're using a simplified two-body model, meaning we're only considering Earth's gravity.

We're assuming the spacecraft makes two instantaneous burns to complete the Hohmann transfer. The first burn is accounted for through the initial velocity, and the second burn is calculated at the end of the transfer.

We're not accounting for things like atmospheric drag, fuel consumption, or the engine continuously firing during the transfer.

For the Monte Carlo tests, we're changing the orbital radii rather than introducing errors in the burns or disturbances to the spacecraft.

Goal:

Overall, we're trying to see which numerical method is better suited for simulating a Hohmann transfer when considering both accuracy and computational cost.

We expect RK4 to give us better accuracy, while Modified Euler requires fewer calculations per step. By comparing the convergence studies, runtimes, and Monte Carlo results, we can see how much of a difference that makes and whether RK4's additional computation is worth it.
