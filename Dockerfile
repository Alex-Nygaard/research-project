# Use the official Python 3.9 image from the Docker Hub
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all files from the current directory to the Docker image
COPY . .

# install dependencies from packages.txt
RUN /bin/bash -c "pip install -q -r packages.txt"

# Command to execute (replace 'your_script.py' with the actual script you want to run)
CMD ["/bin/bash", "-c", "python main.py --option=\"deployment\" --num_clients=20 --trace_file=\"logs/clients/clients20_01.json\" --batch_size=\"noniid\" --local_epochs=\"noniid\" --data_volume=\"noniid\" --data_labels=\"noniid\""]
