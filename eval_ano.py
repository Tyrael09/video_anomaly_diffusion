import os
import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path
import accelerate
import torch
from torch import optim
from torch import multiprocessing as mp
from torch.utils import data
from tqdm.auto import tqdm
from torchmetrics.functional import auroc, accuracy, precision, f1_score
import k_diffusion as K
import matplotlib.pyplot as plt
from custom_datasets import feature_dataset
import random
import numpy as np
import pandas as pd
import sklearn.metrics as skmetr

GLOBAL_SEED = 42


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    global GLOBAL_SEED
    GLOBAL_SEED = seed


def args_parser():

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, required=False, help="the configuration file")

    p.add_argument("--batch_test", type=int, default=256, help="the test batch size")
    p.add_argument("--lr", type=float, help="the learning rate")

    p.add_argument("--n_sample", type=int, help="evaluate sample size")

    p.add_argument("--name", type=str, default="model", help="the name of the run")
    p.add_argument("--num-workers", type=int, default=6, help="the number of data loader workers")

    p.add_argument("--seed", type=int, default=42, help="the random seed")
    p.add_argument(
        "--start-method",
        type=str,
        default="fork",
        choices=["fork", "forkserver", "spawn"],
        help="the multiprocessing start method",
    )

    return p.parse_args()


def get_datasets(dataset_config):
    dataset_root: str = "/local/scratch/Cataract-1K-Full-Videos/"
    test_set = feature_dataset.ClipDataset(
        dataset_root,
        clip_len=16,
        feat_model=dataset_config["feat_model"],
        split="test",
    )
    return test_set


def main():
    args = args_parser()
    seed_everything(seed=args.seed)
    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = K.config.load_config(open("/gris/gris-f/homestud/heschwee/video_anomaly_diffusion/configs/my_config.json"))
    model_config = config["model"]
    opt_config = config["optimizer"]
    sched_config = config["lr_sched"]
    ema_sched_config = config["ema_sched"]
    feat_size = model_config["input_size"]
    dataset_config = config["dataset"]
    test_set = get_datasets(dataset_config)

    if model_config["sampler"] == "lms":
        p_sampler_fn = partial(K.sampling.sample_lms, disable=True)
    elif model_config["sampler"] == "heun":
        p_sampler_fn = partial(K.sampling.sample_heun, disable=True)
    elif model_config["sampler"] == "dpm2":
        p_sampler_fn = partial(K.sampling.sample_dpm_2, disable=True)
    else:
        print("unknown sampler method")
        ValueError("Invalid sampler method")

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1, cpu=False)
    device = accelerator.device
    print(f"Process {accelerator.process_index} using device: {device}", flush=True)

    gvad_model = K.models.GVADModel(feat_size)

    if opt_config["type"] == "adamw":
        opt = optim.AdamW(
            gvad_model.parameters(),
            lr=opt_config["lr"] if args.lr is None else args.lr,
            betas=tuple(opt_config["betas"]),
            eps=opt_config["eps"],
            weight_decay=opt_config["weight_decay"],
        )
    elif opt_config["type"] == "sgd":
        opt = optim.SGD(
            gvad_model.parameters(),
            lr=opt_config["lr"] if args.lr is None else args.lr,
            momentum=opt_config.get("momentum", 0.0),
            nesterov=opt_config.get("nesterov", False),
            weight_decay=opt_config.get("weight_decay", 0.0),
        )
    else:
        raise ValueError("Invalid optimizer type")

    if sched_config["type"] == "inverse":
        sched = K.utils.InverseLR(
            opt, inv_gamma=sched_config["inv_gamma"], power=sched_config["power"], warmup=sched_config["warmup"]
        )
    elif sched_config["type"] == "exponential":
        sched = K.utils.ExponentialLR(
            opt, num_steps=sched_config["num_steps"], decay=sched_config["decay"], warmup=sched_config["warmup"]
        )
    else:
        raise ValueError("Invalid schedule type")

    assert ema_sched_config["type"] == "inverse"
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config["power"], max_value=ema_sched_config["max_value"])

    if accelerator.is_main_process:
        try:
            print("Number of items in testset:", len(test_set))
        except TypeError:
            pass

    test_dl = data.DataLoader(
        test_set, args.batch_test, shuffle=True, drop_last=False, num_workers=args.num_workers, persistent_workers=True
    )

    model, opt, test_dl = accelerator.prepare(gvad_model, opt, test_dl)

    sigma_min = model_config["sigma_min"]
    sigma_max = model_config["sigma_max"]

    seed_noise_path = f"seed_noise_{feat_size}.pth"
    if os.path.exists(seed_noise_path):
        seed_noise = torch.load(seed_noise_path, map_location=device)
    else:
        seed_noise = torch.randn([1, feat_size], device=device)
        torch.save(seed_noise, seed_noise_path)

    model = K.config.make_denoiser_wrapper(config)(model)
    model_ema = deepcopy(model)

    ckpt_path = "/gris/gris-f/homestud/heschwee/video_anomaly_diffusion/checkpoints_6/model_epoch_100.pth"  # TODO: Change to current version. Dont' forget to change "checkpoints_n" folder name if necessary.
    if Path(ckpt_path).exists():
        if accelerator.is_main_process:
            print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Load the state dictionaries
        model_state_dict = ckpt["model"]
        model_ema_state_dict = ckpt["model_ema"]
        opt_state_dict = ckpt["opt"]
        sched_state_dict = ckpt["sched"]
        ema_sched_state_dict = ckpt["ema_sched"]

        # Load the state dictionaries into the models
        accelerator.unwrap_model(gvad_model).load_state_dict(model_state_dict)
        accelerator.unwrap_model(model_ema).load_state_dict(model_ema_state_dict)

        # Load the optimizer and schedulers
        opt.load_state_dict(opt_state_dict)
        sched.load_state_dict(sched_state_dict)
        ema_sched.load_state_dict(ema_sched_state_dict)

        del ckpt
    else:
        print("state path does not exist.")
        print(ckpt_path)
        exit(1)

    def plot_prediction(g_dist, label, vids, idx, i, fps=60):
        """
        Plot the predictions and ground truths with x-axis in seconds.

        Args:
            g_dist: Normalized distance values.
            label: Ground truth values.
            vids: Video IDs corresponding to the data.
            idx: Frame indices of the clips.
            i: Identifier for saving the plot.
            fps: Frames per second of the video.
        """
        g_dist = K.utils.normalize(g_dist)
        uniques = np.unique(vids)

        for item in uniques:
            v_path = f"plots_2/{item}"
            K.utils.mkdir("plots_2")
            mask = np.where(vids == item)[0]
            sort_idx = np.argsort(idx[mask])

            vid_dist = g_dist[mask]
            vid_dist = vid_dist[sort_idx]

            vid_label = label[mask]
            vid_label = vid_label[sort_idx]

            # Convert frame indices to seconds
            time_in_seconds = idx[mask][sort_idx] / fps
            time_in_seconds -= time_in_seconds.min()

            plt.figure(figsize=(10, 6))

            # Distance plot
            plt.subplot(211)
            plt.plot(time_in_seconds, vid_dist, label="Distance", c="b")
            plt.title(f"Video: {item}", fontsize=14)
            plt.ylabel("Normalized Distance", fontsize=12)
            plt.ylim(0, 1.1)
            plt.legend(loc="upper right")
            plt.grid(visible=True, linestyle="--", alpha=0.6)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

            # Ground truth plot
            plt.subplot(212)
            plt.plot(time_in_seconds, vid_label, label="Ground Truth", c="r")
            plt.xlabel("Time (seconds)", fontsize=12)  # Updated label
            plt.ylabel("Ground Truth", fontsize=12)
            plt.ylim(-0.1, 1.1)  # Adjust to add some padding
            plt.legend(loc="upper right")
            plt.grid(visible=True, linestyle="--", alpha=0.6)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.tight_layout()
            plt.savefig(f"{v_path}_{i}.png")
            plt.close()

    def plot_auroc(fpr, tpr, thresholds, i):
        # Calculate the optimal threshold based on maximum distance from diagonal
        distances = tpr - fpr  # Vertical distance from the x = y line
        optimal_index = np.argmax(distances)  # Find the index of the maximum distance
        optimal_threshold = thresholds[optimal_index]
        optimal_fpr = fpr[optimal_index]
        optimal_tpr = tpr[optimal_index]
        print(f"threshold: {optimal_threshold}, fpr: {optimal_fpr}, tpr: {optimal_tpr}")
        auc = skmetr.auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random guessing")
        plt.scatter(
            optimal_fpr, optimal_tpr, color="red", label=f"Optimal Threshold = {optimal_threshold:.2f}", marker="o"
        )

        # Labels and title
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(f"plots_2/{i}_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate(n_start):
        if accelerator.is_main_process:
            tqdm.write("Evaluating...")

        sigmas = K.sampling.get_sigmas_karras(11, sigma_min, sigma_max, rho=7.0, device=device)
        sigmas = sigmas[n_start:]
        sample_noise = seed_noise.to(device) * sigmas[0]

        def sample_fn(x_real):
            x_real = x_real.to(device)
            x = sample_noise + x_real
            x_0 = p_sampler_fn(model_ema.forward, x, sigmas)
            gen_preds = model_ema.gvad_model.loss(x_real, x_0)
            return gen_preds

        gen_preds, labels, vid, idx = K.evaluation.compute_eval_outs_aot(accelerator, sample_fn, test_dl)
        preds_auc = auroc(gen_preds, labels, "binary").cpu().numpy()
        preds_acc = accuracy(gen_preds, labels, "binary").cpu().numpy() * 100
        preds_prec = precision(gen_preds, labels, "binary").cpu().numpy() * 100
        preds_f1 = f1_score(gen_preds, labels, "binary").cpu().numpy()
        fpr, tpr, thresholds = skmetr.roc_curve(
            np.array(labels.cpu().numpy()), np.array(gen_preds.cpu().numpy()), pos_label=1
        )

        # Ensure all tensors are on CPU and converted to numpy
        metrics = {
            "Sigma": n_start,
            "AUC": preds_auc,
            "Accuracy": preds_acc,
            "Precision": preds_prec,
            "F1-Score": preds_f1,
        }

        # Create a DataFrame
        df = pd.DataFrame([metrics])  # Wrap in a list to create a single-row DataFrame

        # Save to CSV (append if file exists)
        csv_path = f"metrics_new_2.csv"
        df.to_csv(csv_path, mode="a", index=False, header=not pd.io.common.file_exists(csv_path))

        vid = np.concatenate(vid)
        return gen_preds.cpu().numpy(), labels.cpu().numpy(), vid, idx.cpu().numpy(), fpr, tpr, thresholds

    for i in range(11):
        gen_preds, labels, vids, idx, fpr, tpr, thresholds = evaluate(i)
        if i == 0 or i > 7:
            plot_prediction(gen_preds, labels, vids, idx, i)
            plot_auroc(fpr, tpr, thresholds, i)


if __name__ == "__main__":
    print("Hello there!")
    main()
    print("Obiwan Kenobi.")
