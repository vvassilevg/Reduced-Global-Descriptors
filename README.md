# Reduced-Global-Descriptors
Python scripts to define the most relevant features of a ML descriptor as described in Kabylda, A., Vassilev-Galindo, V., Chmiela S., Poltavski, I., Tkatchenko, A.; arXiv preprint arXiv:2209.03985.

1. DOFs_prediction.py : Script that reads an ML model (ML_{original} in the original publication) and the dataset on which the model was trained. It assumes format of ML models and datasets of (s)GDML (.npz). Outputs one or more NPZ files containing the predictions of the ML model with each feature masked one by one (i.e., there is one set of predictions for each feature in the original descriptor). Here, "masked" means that the value of the feature is set to 0 for all configurations.

2. DOFs_errors.py : Script that reads the output(s) of DOFs_prediction.py, either a single NPZ file or a directory containing more than one NPZ files. Outputs NPZ files containing the indexes of the features to remove for each selected percentile (i.e., one file per selected percentile). These NPZ files are the ones that can be fed to GDML.

3. sgdml_routines : Directory containing the modified (s)GDML (http://sgdml.org) routines. Install the (s)GDML package by cloning the git repository (follow the instructions in https://github.com/stefanch/sGDML) and replace the original files with the files in this directory. if you want to use the reduced descriptor to train a model, here is an example:

sgdml all --gdml --desc_type 2 --DOFs_file rm_idxs.npz dataset.npz n_train n_valid n_test

--desc_type :  can be either 1 (default) and 2 (reduced descriptor). Please note that, for the moment, desc_type = 2 does not support the use of symmetries (i.e., you must include the --gdml argument when running)

--DOFs_files : NPZ with the indexes of the features to remove from the descriptor (file written by DOFs_error.py script). This is ignored if desc_type is set to 1.
