import os.path as osp
import torch
from argparse import ArgumentParser

from cvl import utils
from cvl.trainers import ClassificationTrainer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoints_path", type=str, help="Path to the Configs YAML file")
    args = parser.parse_args()

    params = utils.read_pickle_file(osp.join(args.checkpoints_path, "params.pk"))
    chkpt = torch.load(osp.join(args.checkpoints_path, "best.pt"), map_location="cpu", weights_only=False)


    engine = ClassificationTrainer(params["configs_file_path"])

    for k, v in params.items():
        setattr(engine, k, v)

    engine.model.load_state_dict(chkpt["model_state_dict"], strict=True)
    engine.fit_test()

    # engine.visualizer_target_layers = [
    #     "block2.branch1_relu",
    #     "block3.branch1_relu",
    #     "block4.branch1_relu",
    #     "block4_sp_attn",
    #     "block4_aux.5"
    # ]
    # engine.visualize(args.checkpoints_path)
