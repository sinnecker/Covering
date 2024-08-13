# Set Covering Problem Solver

This project provides a comprehensive solution for instances of the Set Covering Problem (SCP). It includes code for generating problem instances, reducing problem matrices, solving the instances, and plotting the results.

## Project Structure

- **generate/**: Contains the code for generating all instances of the Set Covering Problem.
- **reduction/**: Contains the code for reducing the matrices of the problem instances.
- **solver_notebook.ipynb**: A Jupyter notebook that includes the code to solve the SCP instances and plot the results.

## Dependencies

Before running the project, ensure you have the following Python packages installed:

- `numpy`
- `matplotlib`
- `networkx`
- `time`
- `scipy`
- `gurobipy`

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Optimization Solver

This project uses **Gurobi** to solve the optimization problems related to the Set Covering Problem. Gurobi is a state-of-the-art mathematical programming solver that efficiently handles large-scale linear and mixed-integer programming problems.

To use Gurobi, you will need to have a valid license. Academic licenses are available for free on the [Gurobi website](https://www.gurobi.com/academia/academic-program-and-licenses/). After obtaining the license, follow the instructions to install and set up Gurobi on your system.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sinnecker/Covering.git
    cd Covering
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Gurobi:**

    Make sure Gurobi is properly installed and licensed on your system. You can refer to the [Gurobi installation guide](https://www.gurobi.com/documentation/) for assistance.

## Usage

1. **Generating Instances:**

    To generate problem instances, run the scripts located in the `generate/` directory. This will create the data necessary for testing and solving the SCP.

    ```bash
    python generate/generate_instances.py
    ```

2. **Matrix Reduction:**

    Use the scripts in the `reduction/` directory to reduce the matrices of the problem instances. This step helps simplify the problem and optimize the solution process.

    ```bash
    python reduction/reduce_matrix.py
    ```

3. **Solving Instances:**

    Open the `solver_notebook.ipynb` file in Jupyter Notebook. This notebook contains code to solve the SCP instances and visualize the results.

    ```bash
    jupyter notebook solver_notebook.ipynb
    ```

    In the notebook, you'll find cells to execute the solver, using Gurobi, and generate plots that illustrate the solutions.

## Contributions

Contributions to this project have been made by the following individuals:

- **Prof. Claudia D’Ambrosio**, École Polytechnique, France
- **Prof. Marcia Fampa**, Universidade Federal do Rio de Janeiro, Brazil
- **Prof. Jon Lee**, University of Michigan, Ann Arbor, MI, USA
- **Felipe Sinnecker**, Universidade Federal do Rio de Janeiro, Brazil
