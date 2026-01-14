# from example import NetLLMNextForConditionalGeneration, NetLLMNextLightningModel
# from te_gnn import NetLLMNextForConditionalGeneration, NetLLMNextLightningModel, init_model
from tellm_main import *
from src.utils import print_config, logging
import os
import glob
import hydra
from omegaconf import OmegaConf
import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert NetLLM checkpoint to pretrained model.")
    parser.add_argument(
        "--ckpt_path", "-c", type=str, required=True,
        help="Path to the model outputs/checkpoint directory (e.g., ./outputs/2025-06-23/14-11-55)"
    )
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    config_path = os.path.join(ckpt_path, "outputs", "config.yaml")
    pattern = os.path.join(ckpt_path, "checkpoints", "best-epoch*.*pt")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found matching the pattern.")
    config = OmegaConf.load(config_path)
    print_config(config, resolve=True)

    file_list = [f for f in checkpoint_files if os.path.isfile(f)]
    path = f"{file_list[0]}"
    model = init_model(config)
    model_module = TeLLMLightningModel.load_from_checkpoint(
        path,
        config=config,
        model=model
    )
    # if config.training.use_lora:
    #     model_module.model.language_model = model_module.model.language_model.merge_and_unload()
    model_module.model.save_pretrained(
        os.path.join(ckpt_path, "pretrained"),
    )
    logging.info(f"Model and processor saved to {os.path.join(ckpt_path, 'pretrained')}")
if __name__ == "__main__":
    main()

# ckpt_path = './outputs/2025-06-23/14-11-55'

# config_path = os.path.join(ckpt_path, "outputs", "config.yaml")
# pattern = os.path.join(ckpt_path, "checkpoints", "best-epoch*.ckpt")
# checkpoint_files = glob.glob(pattern)
# if not checkpoint_files:
#     raise FileNotFoundError("No checkpoint files found matching the pattern.")
# config = OmegaConf.load(config_path)
# print_config(config, resolve=True)


# path = f"{checkpoint_files[-1]}"

# processor, model = init_model(config)

# model_module = NetLLMNextLightningModel.load_from_checkpoint(
#     path,
#     config=config,
#     model=model
# )

# model = model_module.model.merge_and_unload()
# processor.save_pretrained(
#     os.path.join(ckpt_path, "pretrained"),
# )
# model.save_pretrained(
#     os.path.join(ckpt_path, "pretrained"),
# )
# logging.info(f"Model and processor saved to {os.path.join(ckpt_path, 'pretrained')}")