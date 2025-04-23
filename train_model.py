from cvl.trainers import ClassificationTrainer


if __name__ == "__main__":
    engine = ClassificationTrainer("./try2.yaml")
    engine.save_params()
    engine.fit()