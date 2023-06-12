# Master Thesis - Decoupling the Electric Vehicle Routing Problem: A Reinforcement Learning Approach

# (Deep) Reinforcement learning for grid network planning

Develop a RL tool to evaluate/test the efficiency of grid networks for transport operations. 

# Build / Dev

This project uses miniconda for packaging and dependency management.

# Set up
After installing miniconda for your system:

1. Create a venv and install the projects dependencies
    ```commandline
    conda env create --file environment.yml
    ```
3. Based on your OS, follow the instructions on this [website](https://pytorch.org/get-started/locally/) to install pytorch. You can also make use of CUDA or MPS, if you install the library correctly based on your OS.

    *If you use MacOS, follow the instructions [here](https://developer.apple.com/metal/pytorch/) to properly install pytorch and make use of cuda*

# Quick Start
### Run project
You can check the parameters in `./src/options.py` file

    ```commandline
    python run.py --problem tsp --graph_size 10 --run_name 'tsp10_rollout'
    ```

    or 

    ```commandline
    run.py --problem cvrp --graph_size 10 --run_name 'vrp10_rollout'
    ```

### Generate data
    ```commandline
    python generate_data.py --problem all --name validation --seed 4321
    ```

### Get plots
    ```commandline
    tensorboard --logdir 'path/to/master-thesis-2023-reinforcement-learning-in-grids/'
    ```
### Run tests
   ```commandline
   pytest
   ```
