#!/bin/bash

num_clients=20

for seed in {1001..1005}
do
    trace_file="logs/clients/clients${num_clients}_${seed}.json"
    echo "Running simulations for clients${num_clients}_${seed}"

    # Blind
    echo "Running: python main.py --option=\"simulation\" --num_clients=$num_clients --trace_file=\"$trace_file\""
    python main.py --option="simulation" --num_clients=$num_clients --trace_file="$trace_file"

    ## Experiments
    echo "Running: python main.py --option=\"simulation\" --num_clients=$num_clients --trace_file=\"$trace_file\" --batch_size=\"noniid\""
    python main.py --option="simulation" --num_clients=$num_clients --trace_file="$trace_file" --batch_size="noniid"

    echo "Running: python main.py --option=\"simulation\" --num_clients=$num_clients --trace_file=\"$trace_file\" --local_epochs=\"noniid\""
    python main.py --option="simulation" --num_clients=$num_clients --trace_file="$trace_file" --local_epochs="noniid"

    echo "Running: python main.py --option=\"simulation\" --num_clients=$num_clients --trace_file=\"$trace_file\" --data_volume=\"noniid\""
    python main.py --option="simulation" --num_clients=$num_clients --trace_file="$trace_file" --data_volume="noniid"

    echo "Running: python main.py --option=\"simulation\" --num_clients=$num_clients --trace_file=\"$trace_file\" --data_labels=\"noniid\""
    python main.py --option="simulation" --num_clients=$num_clients --trace_file="$trace_file" --data_labels="noniid"

    ### Real
    echo "Running: python main.py --option=\"simulation\" --num_clients=$num_clients --trace_file=\"$trace_file\" --batch_size=\"noniid\" --local_epochs=\"noniid\" --data_volume=\"noniid\" --data_labels=\"noniid\""
    python main.py --option="simulation" --num_clients=$num_clients --trace_file="$trace_file" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"

    echo "Completed simulations for clients${num_clients}_${seed}"
done