#!/bin/bash

clients=("clients1001" "clients1002" "clients1003" "clients1004" "clients1005")

for client in "${clients[@]}"; do
  echo "Running simulations for $client"

  # Blind
  python main.py --option="simulation" --num_clients=50 --trace_file="logs/clients/${client}.json"

  ## Experiments
  python main.py --option="simulation" --num_clients=50 --trace_file="logs/clients/${client}.json" --batch_size="noniid"
  python main.py --option="simulation" --num_clients=50 --trace_file="logs/clients/${client}.json" --local_epochs="noniid"
  python main.py --option="simulation" --num_clients=50 --trace_file="logs/clients/${client}.json" --data_volume="noniid"
  python main.py --option="simulation" --num_clients=50 --trace_file="logs/clients/${client}.json" --data_labels="noniid"

  ### Real
  python main.py --option="simulation" --num_clients=50 --trace_file="logs/clients/${client}.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"

  echo "Completed simulations for $client"
done
