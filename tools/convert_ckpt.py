import os
import glob
import argparse
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

def list_folders(parent_path, pattern="*"):
    all_dirs = glob.glob(os.path.join(parent_path, pattern))
    return [d for d in all_dirs if os.path.isdir(d)]

def main(ckpt_path: str):
    ckpt_path = os.path.expanduser(ckpt_path)
    checkpoints_dir = os.path.join(ckpt_path, "checkpoints")
    # list only subfolders matching "best-epoch*.ckpt"
    folders = list_folders(checkpoints_dir, pattern="best-epoch*.ckpt")
    if not folders:
        raise FileNotFoundError(f"No checkpoint folders found matching pattern: {os.path.join(checkpoints_dir, 'best-epoch*.ckpt')}")
    for folder in folders:
        print(f"Converting checkpoint folder: {folder}")
        output_path = f"{folder}.pt"  # or any suitable output name
        convert_zero_checkpoint_to_fp32_state_dict(folder, output_path)
        print(f"Done: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Deepspeed Zero checkpoint to FP32 PyTorch state dict.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the folder containing the checkpoint (e.g., python convert-ckpt.py --ckpt_path='outputs/2025-04-07/22-10-25')"
        )
    args = parser.parse_args()
    main(args.ckpt_path)
