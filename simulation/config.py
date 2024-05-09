from simulation.client import Client
from simulation.constants import NUM_CLIENTS


class Config:
    def __init__(self):
        self.clients_ids = Client.generate_ids(NUM_CLIENTS)
        self.clients
