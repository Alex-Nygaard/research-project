def create():
    from visualization.data import RunData
    from visualization.plotting import plot_metrics

    simulations = RunData.build_many(["run_3711645", "run_3711513", "run_3711520"])
    deployment = RunData.build("run_3711520")
    for simulation in simulations:

        plot_metrics(
            simulation.history,
            simulation.metrics,
            save_dir=f"from-delftblue/logs/{simulation.run_id}/{simulation.run_config.code}.png",
        )
