# Forcing a Glocal Pattern on the Activity of an SNN
Framework to train and simulate a spiking neural network (SNN) using Brian2, that implements a learning rule that combines local learning rules with a global feedback mechanism to support strong, contrasty (activity) pattern formation. Such global, "unguided" pattern induction (e.g., through neuromodulatory signals) might lead to easily discriminable output activations even if the network is trained via local (biological-plausible) mechanisms.

## Installation
The simulation is written in *Python 3*.

Modules, so far, that have to be installed in order to run the simulation:

- brian2
- numpy
- sklearn
- matplotlib

Additionally, one has to download the datasets for which one ones to run the simulation. Please refer to the dedicated `README` files in the `data` folder for the desired dataset for instructions on how to get the dataset.

If all prerequisites are met, one has to configure the simulation in the file `src/configuration.py`. The file contains extensive comments to guide one through the configuration process.

Finally, one can run the simulation by hitting 

```
python3 main.py
```

in the `src` directory.

**Note**, if *cython* and/or no c++ compiler is installed, one has to change the *target* preference in the file `src/brian_preferences` from *cython* to *numpy*.

