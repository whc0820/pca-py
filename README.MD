# PCA

Image compression implement by PCA.

### Follow the instructions to run this python script

1. 
   - Windows: open Anaconda Powershell Prompt
   - MacOS: open terminal
2. cd to current folder
3. setup the environment

        conda env create -f environment.yml -p ./env
        conda activate ./env

 4. run the script
        
        python pca.py

### Supported image format

- png
- jpg
- jpeg

### Problems met

- the `environment.yml` file encoded as `UTF-16 EL` when exporting the active environment through VSCode conda cli, this would cause problem dealing the import. Encoded as `UTF-8` solved.


### Result

![classification](/result.png)