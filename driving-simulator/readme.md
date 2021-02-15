<!-- Driving simulator Brecht Evens -->
## Driving simulator Brecht Evens

### Prerequisites
The code is written using Python 3, and mainly uses the following Python libraries:
* CasADI
* Customopengen
* Pyglet

The required packages can be easily installed using the provided Pipfiles using pipenv.

### Project structure

The project was built upon the original project code from the [driving-interactions](https://github.com/dsadigh/driving-interactions) GitHub project of D. Sadigh. Hence, the visualization assets and some parts of the the code of both driving simulators correspond.

* **car.py** : Classes for representing different vehicle types, i.e. UserControlledCar and GNEPOptimizerCar
* **collision.py** : Contains various collision avoidance constraints
* **constraints.py** : General class for representing constraints
* **dynamics.py** : Classes for representing the dynamics of a vehicle using a longitudinal model or a kinematic bicycle model
* **experiment.py** : Classes for representing different experiments
* **feature.py** : General class for representing cost function features
* **gaussseidelsolver.py** : Class for the Gauss-Seidel solution methodology proposed in the thesis
* **lagrangiansolver.py** : Class for an Augmented Lagrangian solution methodology using single optimization problem
* **lane.py** : Class for representing a lane of a road
* **learning.py** : Class for the online learning methodology
* **logger.py** : Class for the logger
* **penalty.py** : Class for the penalty handler
* **road.py** : Class for representing a road consisting of multiple lanes
* **run** : Main script for running an experiment
* **trajectory.py** : Class for representing a trajectory of an object
* **visualize.py** : Class for the main loop and the visualization of experiments
* **visualize_data.py** : function for visualizing information of an experiment
* **world.py** : Contains a class for representing traffic situations and contains various traffic situations

### How to use code
An experiment can be performed by executing
```console
~$ python3.exe run world_name "arg1" "args2" ...
```
where world_name corresponds to any of the worlds defined in world.py, e.g. "one_dimensional_GPG".


