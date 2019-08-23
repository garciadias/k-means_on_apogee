export PYTHONPATH="${PYTHONPATH}:${PWD}"
pip install -r requirements.txt
bash ./src/download_data.sh
python ./src/create_dataset.py
bash run_test.sh
python ./src/model.py
