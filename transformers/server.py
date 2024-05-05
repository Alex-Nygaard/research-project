import flwr as fl

def metrics_aggregation(metrics):
    """Aggregate metrics"""
    accuracies = [metric["accuracy"] for _, metric in metrics]
    avg_acc = sum(accuracies) / len(accuracies)
    return {"accuracy": avg_acc}

if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=metrics_aggregation,
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:18080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )
