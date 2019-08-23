# Use K-means to group synthetic spectra
## This code downloads a little sample of [APOGEE](https://www.sdss.org/surveys/apogee-2/) spectra and performs K-means on it. Here I use Calinski-Harabasz score to find number of clusters that betters describe this dataset. 

## First install requirements using:

```
pip install -r requirements.txt
```

## To download the sample spectra data run:

```
bash ./src/download_data.sh
```

## To create spectral sample run:

```
python ./src/create_dataset.py
```

## After this you can run the tests to make sure your have the code properly settled. You do this by running:

```
bash run_test.sh
```

## To fit model run:

```
python ./src/model.py
```

## To use your own data you need to replace the variable *spectra* at **model.py** with your data. It can be a pandas dataframe or a numpy array.

## You can run all together by using:

```
bash run_all.sh
```
