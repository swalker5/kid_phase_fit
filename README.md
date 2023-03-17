# `kid_phase_fit`

## Project structure

This code is structured so that `kid_phase_fit.py` will take a microwave power sweep of microwave kinetic inductance detector (MKID) data (currently in .nc format for now) and fit its phase. The code requires at least two sweeps, one low power dataset and another dataset at any microwave power.

## Build

```
    $ git clone /url/to/this/repo
```

## Code Demo

```
    $ python kid_fit_phase.py config.yaml
```

## Notes

### Regarding `config.yaml`
The file `config.yaml` contains user-specified parameters to be used by `kid_phase_fit.py`, separated into different sections. The `load` section corresponds to the folders to find data and save the code output. The variable `sweep_name` corresponds to the beginning of a particular network and obsnum. The `save` section corresponds to how to save the data fit, as figures, a PDF, or in a .pkl file. The `preview` section corresponds to showing plots while running the code or not. The `weight` section corresponds using a weighting around a tone frequency and quality factor or not. The `flag_settings` section corresponds to guessing the optimal drive power for each resonator corresponding to a user-defined nonlinearity value as well as thresholds for the fits before flagging the data. The `fit_settings` section corresponds to setting the microwave power range to fit, how many resonator linewidths to fit, and which resonators to fit.


The flag settings are as follows:
0 = good fit, below user-defined thresholds
1 = above user-defined thresholds 
2 = failed fit

Also, an empty list corresponds to a skipped fit/likely not a resonator.
