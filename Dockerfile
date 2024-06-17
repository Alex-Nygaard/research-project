# Use the official Python 3.9 image from the Docker Hub
FROM --platform=linux/amd64 python:3.9

# Set the working directory
WORKDIR /app

# Copy all files from the current directory to the Docker image
COPY . .

# install dependencies from packages.txt
RUN /bin/bash -c "pip install -q -r packages.txt"

CMD ["/bin/bash", "-c", "python main.py --option=\"deployment\" --num_clients=20 --server_address=\"167.235.241.219:5040\" --batch_size=\"noniid\" --local_epochs=\"noniid\" --data_volume=\"noniid\" --data_labels=\"noniid\""]
