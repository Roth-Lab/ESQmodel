# ESQmodel
A biologically informed evaluation of 2-D cell segmentation quality in multiplexed tissue images.

## Installation
1. Clone this repo
```commandline
git clone https://github.com/Roth-Lab/ESQmodel.git
```
2. Move into the repo and install ESQmodel
```commandline
pip install .
```
3. Run ESQmodel
```commandline
esq infer --exp-csv 'sample_exp.csv' --prior-csv 'sample_prior.csv' --out_dir .
```

## Quick Documentation
### Required Inputs
Running ESQmodel requires two files and an output location. Other parameters can also be modified. An example of all required files is given in the `example\` folder. We input the csv file name as the following: Let's talk about them one by one.
1. Expression Matrix: this is a cell by expression marker matrix. There should not be an index column.
   * `--exp-csv 'sample_exp.csv'` : the file path of the csv file holding expression data.
2. Prior Matrix: this is a cell type by marker prior matrix with entries of either 1, 2, 3 indicating a prior expectation of low, mid, high expression of a specific marker for each cell type. There should not be an index column.
   * `-p / --prior-csv`: the  file path of the csv file holding a cell type by marker prior matrix.
3. `-o / --out_dir`: this is the output directory to save results.

### Additional and Optional Inputs
4. `-t" / --num-iters`: this is the number of iterations to run inference, default at 10000.

### Outputs
1. `rho_est.csv`: these are the inferred rho vector for each cell from ESQmodel. The format is cell by cell type, in which each row indicates the percentage contribution of each cell type's profile to the observed cellular expression.
2. `entropy.csv`: these are the entropy scores calculated from `rho_est.csv` prior to normalization.

### Example Run
Let us run ESQmodel on one of our example datasets. Here we are running a sample dataset with a sample prior and saving the results in the current folder.
```commandline
esq infer --exp-csv 'example/y.csv' --prior-csv 'example/priors.csv' -o 'example/'
```

### Troubleshooting:
1. ESQmodel is not running: double check that all matrices should not have an index column. Check the example matrices to see a reference.
2. NAN values in expression matrices: ESQmodel does not supports nans.
3. ESQmodel is taking its time: ESQmodel can take up to an hour or two per image depending on image size.

## License
ESQmodel is licensed under the MIT License.

## Version
**0.1.0** - First release.

## Contact
Author: [Eric Lee](https://www.linkedin.com/in/ericlee0920/) 
