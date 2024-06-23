# CSE3000 Research Project - Federated learning

_Author: **Alexander Nygård**_

_Project timeline: **April - June 2024**_

_Responsible Professor: **Jérémie Decouchant**_

_Supervisor: **Bart Cox**_

> _(Code for)_ A Thesis Submitted to EEMCS Faculty Delft University of Technology, In Partial Fulfilment of the Requirements For the Bachelor of Computer Science and Engineering

### Research question

**Improving the Accuracy of Federated Learning Simulations**

Using Traces from Real-world Deployments to Enhance the Realism of Simulation Environments

### Project Description

The project aims to improve the accuracy of federated learning simulators by using traces from real-world deployments to enhance the realism of FL simulation environments.

This is accomplished by running experiments using `Flower`, an [open-source](https://github.com/adap/flower) federated learning framework.

4 non-IID parameters are collected from each client in a deployment: `batch_size`, `local_epochs`, `data_volume`, and `data_labels`. These parameters are then used to simulate the same deployment in a simulation environment.

### How to use

The project has been tested for Python `3.9.13`.

1. Clone the repository
2. (Optional) Create a virtual environment

    ```bash
    python -m venv venv
    source venv/bin/activate # depending on your OS
    ```

3. Install the dependencies 

    ```bash
    pip install -r requirements.txt
    ```

4. Run a deployment

    ```bash
    python main.py --option="deployment" --num_clients=25
    ```
   
    _This will run the deployment locally, orchestrating subprocesses for both the server and client tasks. Note that this is quite resource intensive with more clients._

5. Run a simulation
    
    ```bash
    # Run a simulation with 25 clients using a non-IID batch size 
    # (will use the trace file from the deployment above to get non-IID batch size configurations)
    python main.py --option="simulation" --num_clients=25 --trace_file="path/to/client_config.json" --batch_size="noniid"
    ```

   _Note: `--trace_file` is used to identify the configuration gathered from a deployment. It contains the `batch_size`, `local_epochs`, `data_volume`, and `data_labels` of all clients. Then, passing args such as `--data_volume="noniid"` ensures the simulation replicates the non-IID'ness of the deployment in the simulation. If no arg is specified, `iid` is the default for all attributes (see code for more details)._

6. Visualize results by opening the `visualize.ipynb` notebook with Jupyter Lab or Notebook.

Alternatively, you can run `bash dep_locally.sh` or `bash sim_locally.sh` to run the deployment or simulation locally, respectively. You will still need a Python environment for this.
