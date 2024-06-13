#python main.py --option="deployment" --num_clients=20 --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"

# Blind
python main.py --option="simulation" --num_clients=50 --trace_file="logs/clients/clients1001.json"

## Experiments
#python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_120/clients.json" --batch_size="noniid"
#python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_120/clients.json" --local_epochs="noniid"
#python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_120/clients.json" --data_volume="noniid"
#python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_120/clients.json" --data_labels="noniid"
#
## Real
#python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_120/clients.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
