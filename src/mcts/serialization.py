import pickle


class _TrajectoryUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("src.algorithm.mcts."):
            module = "src.mcts." + module.removeprefix("src.algorithm.mcts.")
        return super().find_class(module, name)


def load_trajectories(file):
    """Load MCTS trajectories."""
    return _TrajectoryUnpickler(file).load()
