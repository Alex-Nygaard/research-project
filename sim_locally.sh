
python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_100/clients.json" --batch_size="noniid"
python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_100/clients.json" --local_epochs="noniid"
python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_100/clients.json" --data_volume="noniid"
python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_100/clients.json" --data_labels="noniid"

python main.py --option="simulation" --num_clients=20 --trace_file="logs/run_100/clients.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"

