import os
import re
import json
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path as osp
from argparse import ArgumentParser


DATASET_PATHS = {
        "CA-LFW": ("datasets/calfw/images&landmarks/images&landmarks/images", "datasets/calfw/pairs_CALFW.txt"),
        "CP-LFW": ("datasets/cplfw/images", "datasets/cplfw/pairs_CPLFW.txt"),
        "LFW": ("datasets/lfw/lfw-funneled/lfw_funneled", "datasets/lfw/pairs.txt"),
    }


def load_lfw_pairs(images_path: str, pair_file: str):
    pairs = []
    with open(pair_file) as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue

            line = line.strip()
            line = line.replace("\t", " ")
            line = line.split(" ")

            assert len(line) in (3, 4)

            if len(line) == 3:
                d = (
                    osp.join(images_path, line[0], f"{line[0]}_{int(line[1]):04d}.jpg"),
                    osp.join(images_path, line[0], f"{line[0]}_{int(line[2]):04d}.jpg"),
                    1
                )
            else:
                d = (
                    osp.join(images_path, line[0], f"{line[0]}_{int(line[1]):04d}.jpg"),
                    osp.join(images_path, line[2], f"{line[2]}_{int(line[3]):04d}.jpg"),
                    0
                )
            assert osp.exists(d[0]), f"{d[0]} does not exist"
            assert osp.exists(d[1]), f"{d[1]} does not exist"
            pairs.append(d)

    return pairs


def load_caplfw_pairs(images_path: str, pair_file: str):
    pairs = []

    with open(pair_file, "r") as f:
        content = [line.strip() for line in f.readlines()]

    for i in range(0, len(content), 2):
        image1, label1 = content[i].split(" ")
        image2, label2 = content[i + 1].split(" ")
        assert label1 == label2, f"{label1} != {label2}"

        if int(label1) == 0:
            label = 0
        else:
            label = 1

        d = (
            osp.join(images_path, image1),
            osp.join(images_path, image2),
            label,
        )

        assert osp.exists(d[0]), f"{d[0]} does not exist"
        assert osp.exists(d[1]), f"{d[1]} does not exist"

        pairs.append(d)

    return pairs


def get_dataset_pairs(dataset):

    if dataset in ("CA-LFW", "CP-LFW"):
        pairs = load_caplfw_pairs(*DATASET_PATHS[dataset])
    else:
        pairs = load_lfw_pairs(*DATASET_PATHS[dataset])

    return pairs


def get_similarities(url, pairs, model):

    result = []

    for img1, img2, label in tqdm(pairs):
        with open(img1, "rb") as f1, open(img2, "rb") as f2:

            files = {
                "img1": f1,
                "img2": f2,
            }

            data = {
                "model": model,
            }

            res = requests.post(url=url, data=data, files=files)

            if res.status_code != 200:
                print("====================")
                print(img1, img2)
                print(vars(res))
                print("====================")
                continue

        sim = res.json()["similarity"]

        result.append({
            "img1": img1,
            "img2": img2,
            "sim": sim,
            "label": label,
        })

    return result


def calculate_metrics(file_path: str, threshold: float) -> None:

    with open(file_path, "r") as f:
        data = json.load(f)

    labels = np.array([x["label"] for x in data])
    labels = (labels != 0).astype(int)

    sims = np.array([x["sim"] for x in data])
    preds = (sims >= threshold).astype(int)

    cm = np.zeros((2, 2))
    for i in range(len(labels)):
        cm[labels[i], preds[i]] += 1

    TN, FN, FP, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = (TP + TN) / cm.sum()
    fmr = FP / (FP + TN)
    fnmr = FN / (FN + TP)

    metrics = {
        "cm": cm,
        "TN": TN,
        "FN": FN,
        "FP": FP,
        "TP": TP,
        "accuracy": accuracy,
        "fmr": fmr,
        "fnmr": fnmr,
    }

    return metrics



def evaluate(url, dataset, model, threshold):

    save_path = f"{dataset}_{model}.json"

    pairs = get_dataset_pairs(dataset)
    print(dataset)
    stats = {}
    for (img1, img2, lbl) in pairs:
        stats[lbl] = stats.get(lbl, 0) + 1
    print(stats)


    result = get_similarities(url, pairs, model)

    with open(f"{dataset}_{model}.json", "w") as f:
        json.dump(result, f)

    metrics = calculate_metrics(save_path, threshold)
    print(f"{dataset} - {model} - {metrics['TP']} - {metrics['FP']} - {metrics['FN']} - {metrics['TN']} - {metrics['accuracy']:0.4f} - {metrics['fmr']:0.4f} - {metrics['fnmr']:0.4f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-datasets", type=str, nargs="+", choices=("CA-LFW", "CP-LFW", "LFW"))
    parser.add_argument("-models", type=str, nargs="+", choices=("buffalo_l", "buffalo_s"))
    parser.add_argument("-threshold", type=float, default=0.22)
    args = parser.parse_args()
    print(args)

    URL = "http://localhost:8000/api/verify/"

    for d in args.datasets:
        for m in args.models:
            evaluate(URL, d, m, args.threshold)

