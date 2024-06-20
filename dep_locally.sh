#!/bin/bash

num_clients=20

for seed in {1001..1005}
do
    echo "Running deployment with num_clients=$num_clients and seed=$seed..."
    python main.py --option="deployment" --num_clients=$num_clients --trace_file="logs/clients/clients${num_clients}_${seed}.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
    echo "Finished running deployment with num_clients=$num_clients and seed=$seed."
done
