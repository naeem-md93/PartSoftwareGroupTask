from argparse import ArgumentParser

from cvl.trainers import ClassificationTrainer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("yaml_configs_path", type=str, help="Path to the Configs YAML file")
    args = parser.parse_args()


    engine = ClassificationTrainer(args.yaml_configs_path)
    engine.save_params()
    engine.fit()
    engine.visualize(engine.checkpoints_path)
