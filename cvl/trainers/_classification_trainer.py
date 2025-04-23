from . import methods


class ClassificationTrainer:
    def __init__(self, configs_file_path: str) -> None:
        methods.init_classification_trainer(self, configs_file_path=configs_file_path)

    def save_params(self):
        methods.save_classification_trainer_params(self)

    def save_checkpoint(self, epoch: int) -> None:
        methods.save_classification_checkpoint(self, epoch=epoch)

    def resume_checkpoint(self, params_path: str, checkpoint_path: str) -> None:
        methods.resume_classification_checkpoint(self, params_path=params_path, checkpoint_path=checkpoint_path)

    def fit(self, epoch: int = 1, modes: tuple[str] = ("train", "validation")) -> None:
        methods.fit_classification_trainer(self, epoch=epoch, modes=modes)
