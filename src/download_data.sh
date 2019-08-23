dirpath=${PWD}
mkdir $dirpath/data/fits/
wget -i $dirpath/data/list_of_spectra.txt --directory-prefix=$dirpath/data/fits/
