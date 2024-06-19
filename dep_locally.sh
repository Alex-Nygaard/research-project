#!/bin/bash

#python main.py --option="deployment" --num_clients=20  # --server_address="167.235.241.219:5040"

python main.py --option="deployment" --num_clients=20 --trace_file="logs/clients/clients20_1001.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
python main.py --option="deployment" --num_clients=20 --trace_file="logs/clients/clients20_1002.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
python main.py --option="deployment" --num_clients=20 --trace_file="logs/clients/clients20_1003.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
python main.py --option="deployment" --num_clients=20 --trace_file="logs/clients/clients20_1004.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
python main.py --option="deployment" --num_clients=20 --trace_file="logs/clients/clients20_1005.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"

