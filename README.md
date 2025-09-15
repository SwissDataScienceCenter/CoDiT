# CoDiT: Conditional Diffusion Trajectory Predictions

This code provides a conditional diffusion framework to predict floating drifters on the Ocean surface for multiple days.
At the moment, this repository only allows for using trained models for predictions, but soon we will also add the training code. It is recommended to run it on a GPU node.

## Setup

### 1. Setting up credentials for data download

This repo comes with a VSCode devcontainer and it is recommended to start in the built container. However, 
before building the container one should create a `config.ini` with the following content

```
[copernicus]
user = <cds-email>
password = <cds-password>
url = https://cds.climate.copernicus.eu/api
key =  <cds-key>

[copernicusmarine]
username = <copernicusmarine-username>
password = <copernicusmarine-password>
```

where you have to fill your credentials for the [Climate Data Storage](https://cds.climate.copernicus.eu/how-to-api) and [Copernicus Marine Service](https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction). 
It is recommended to put the `config.ini` file somewhere outside the CoDiT folder, so you do not commit it accidentely.

### 2. Setting the paths

Then make sure you set the paths as you want them in `CoDiT/src/codit/data_module/config_paths.py`. In particular that the credential path is set write.

### 3. Downloading bathymetry data

The CoDiT framework makes also use of bathymetry from [GEBCO](https://www.gebco.net/) which is static. You can download a version [here](https://polybox.ethz.ch/index.php/s/Gp6zM2JindN6nHA) and unpack the `bathymetry.zip` file this under `<DATA_PATH>/`, where `<DATA_PATH>` is the one you set in step 2.

### 4. Build the container and test

Now use VSCode to build the container and try to run the notebook `CoDiT/notebooks/get_started.ipynb`. If you set up everything correctly it should run smoothly.

__Note__: If you want GPU support go to the `.devcontainer/Dockerfile` and uncomment the line
 ```
# RUN pip install --upgrade "jax[cuda12]"
```
and rebuild the container.

## Cite

If you find this helpful, please cite

...

## Acknowledgements

Global Drifter Program, HYCOM, copernicusmarine, CDS, GEBCO, SDSC project