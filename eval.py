import os
import torch
import wandb
import random
import argparse
import numpy as np

from hbird.hbird_eval import hbird_evaluation
from hbird.utils.feature_extractors import load_vggt, VGGTFeatureExtractor

def main(args):
    print(f"the script arguments are {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model.startswith("vggt"):
        # VGGT path
        vggt_model = load_vggt(
            backbone=args.model,
            ckpt_path=args.vggt_ckpt,VGGTFeatureExtractor
            device=device,
            hf_model_id=args.vggt_hf_id,
        )
        eval_spatial_resolution = args.input_size // args.patch_size
        feature_extractor = VGGTFeatureExtractor(
            vggt_model,
            eval_spatial_resolution=eval_spatial_resolution,
            d_model=None,
            normalize=args.vggt_normalize,
        )
        # Ensure d_model is known before memory allocation (e.g., when --memory-size is set)
        if getattr(feature_extractor, "d_model", -1) == -1:
            with torch.no_grad():
                _dummy = torch.zeros(1, 3, args.input_size, args.input_size, device=device)
                feature_extractor.forward_features(_dummy)
        print(f"[VGGT] Detected patch embedding dimension: {feature_extractor.d_model}")

        # Handle dataset-specific options
        train_bins = None
        val_bins = None
        train_fs_path = None
        val_fs_path = None

        if args.dataset_name == "mvimgnet":
            if args.train_bins is None or args.val_bins is None:
                raise ValueError(
                    "For MVImgNet, provide --train-bins and --val-bins (e.g., --train-bins 0,15,30 --val-bins 45)"
                )
            train_bins = args.train_bins.split(",") if isinstance(args.train_bins, str) else args.train_bins
            val_bins = args.val_bins.split(",") if isinstance(args.val_bins, str) else args.val_bins
        elif args.dataset_name == "voc":
            train_fs_path = args.train_fs_path
            val_fs_path = args.val_fs_path

        hbird_miou = hbird_evaluation(
            model=None,
            d_model=feature_extractor.d_model if feature_extractor.d_model != -1 else args.embeddings_size,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            input_size=args.input_size,
            augmentation_epoch=1,
            device=device,
            return_knn_details=False,
            n_neighbours=30,
            nn_params=None,
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            memory_size=args.memory_size,
            feature_extractor=feature_extractor,
            train_bins=train_bins,
            val_bins=val_bins,
            train_fs_path=train_fs_path,
            val_fs_path=val_fs_path,
        )
    else:
        # DINO and similar encoders
        model = torch.hub.load("facebookresearch/dino:main", args.model).to(device)

        def token_features(model, imgs):
            return model.get_intermediate_layers(imgs)[0][:, 1:], None

        # Handle dataset-specific options
        train_bins = None
        val_bins = None
        train_fs_path = None
        val_fs_path = None

        if args.dataset_name == "mvimgnet":
            if args.train_bins is None or args.val_bins is None:
                raise ValueError(
                    "For MVImgNet, provide --train-bins and --val-bins (e.g., --train-bins 0,15,30 --val-bins 45)"
                )
            train_bins = args.train_bins.split(",") if isinstance(args.train_bins, str) else args.train_bins
            val_bins = args.val_bins.split(",") if isinstance(args.val_bins, str) else args.val_bins
        elif args.dataset_name == "voc":
            train_fs_path = args.train_fs_path
            val_fs_path = args.val_fs_path

        hbird_miou = hbird_evaluation(
            model.to(device),
            d_model=args.embeddings_size,
            patch_size=args.patch_size,
            batch_size = args.batch_size,
            input_size=args.input_size,
            augmentation_epoch=1,
            device=device,
            return_knn_details=False,
            n_neighbours=30,
            nn_params=None,
            ftr_extr_fn=token_features,
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            memory_size=args.memory_size,
            train_bins=train_bins,
            val_bins=val_bins,
            train_fs_path=train_fs_path,
            val_fs_path=val_fs_path,
        )

    print(f"Hummingbird Evaluation (mIoU): {hbird_miou}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    # Standard arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")

    # Model arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")
    parser.add_argument("--patch-size", type=int, default=16, help="Size of the model patches")
    parser.add_argument("--memory-size", type=int, default=None, help="The size of the memory bank. Unbounded if not specified")
    parser.add_argument("--model", type=str, required=True, help="Model identifier (e.g., dino_vits16 or vggt-1b)")
    parser.add_argument("--embeddings-size", type=int, required=False, help="The size of the model embeddings (optional for VGGT)")
    # VGGT-specific
    parser.add_argument("--vggt-ckpt", type=str, default=None, help="Path to VGGT checkpoint (.pt/.pth). Optional when using --vggt-hf-id")
    parser.add_argument("--vggt-hf-id", type=str, default=None, help="Hugging Face model id, e.g. facebook/VGGT-1B")
    parser.add_argument("--vggt-normalize", action="store_true", help="Apply VGGT-specific normalization (if required)")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="VOCSegmentation", help="Path to the dataset")
    parser.add_argument("--dataset-name", type=str, default="voc", choices=["voc", "mvimgnet", "ade20k", "cityscapes", "coco-thing", "coco-stuff"], help="Name of the dataset to use")
    # MVImgNet-specific
    parser.add_argument("--train-bins", type=str, default=None, help="Comma-separated list of training angle bins for MVImgNet (e.g., '0,15,30')")
    parser.add_argument("--val-bins", type=str, default=None, help="Comma-separated list of validation angle bins for MVImgNet (e.g., '45')")
    parser.add_argument("--angle-bins-dir", type=str, default=None, help="Path to split_angles_mvimagenet (if not auto-detected)")
    parser.add_argument("--masks-dir", type=str, default=None, help="Path to masks root (e.g., datasets/mvimgnet/masks)")
    # Optional file-set overrides (VOC etc.)
    parser.add_argument("--train-fs-path", type=str, default=None, help="Path to training file set (e.g., file_sets/voc/full/trainaug.txt)")
    parser.add_argument("--val-fs-path", type=str, default=None, help="Path to validation file set (e.g., file_sets/voc/val.txt)")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
