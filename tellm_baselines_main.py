import os
import copy
import random
from tqdm import tqdm
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
from src.utils import print_config, setup_rich_logging

import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from deepspeed.ops.adam import DeepSpeedCPUAdam
from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

from pytorch_lightning.callbacks import Callback, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


setup_rich_logging()

from src.model.tellm.modeling_tellm import TeLLMSolverModel, TEHeadMoE
from src.model.tellm.configuration_tellm import TeLLMSolverConfig
from src.model.figret.modeling_figret import FigretTeModel
from src.model.harp.modeling_harp_flash import HARPTeModel
from src.model.dote.modeling_dote import DoteTeModel


import transformers
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM
from transformers import CONFIG_MAPPING
from transformers.utils import logging
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

torch.set_float32_matmul_precision("medium")

from datasets import Dataset

from hydra.utils import instantiate

from src.data_utils.te_dataset import TEDatasetWithinCluster

logger = logging.get_logger()

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def collate_fn(batch):
    # Stack tensors that have batch dimension
    tms_hist = torch.stack([torch.as_tensor(item['tms_hist'], dtype=torch.float32) for item in batch])
    opt = torch.stack([torch.as_tensor(item['optimal_values'], dtype=torch.float32) for item in batch])
    tms = torch.stack([torch.as_tensor(item['tms'], dtype=torch.float32) for item in batch])
    tms_pred = torch.stack([torch.as_tensor(item['tms_pred'], dtype=torch.float32) for item in batch])
    capacities = torch.stack([torch.as_tensor(item['capacities'], dtype=torch.float32) for item in batch])
    node_features = torch.stack([torch.as_tensor(item['node_features'], dtype=torch.float32) for item in batch])
    node_features_norm = torch.stack([torch.as_tensor(item['node_features_norm'], dtype=torch.float32) for item in batch])
    policy_embeds = torch.stack([torch.as_tensor(item['policy_embeds'], dtype=torch.float32) for item in batch])

    # Get common items (without batch dimension) from first element
    edge_index = batch[0]['edge_index']
    pte = batch[0]['paths_to_edges']
    padded_edge_ids_per_path = batch[0]['padded_edge_ids_per_path']
    
    # Prepare multimodal inputs dictionary
    te_inputs = {
        'node_features': node_features,
        'node_features_norm': node_features_norm,
        'capacities': capacities,
        'tms': tms,
        'tms_pred': tms_pred,
        'tms_hist': tms_hist,
        'edge_index': edge_index,
        'paths_to_edges': pte,
        'padded_edge_ids_per_path': padded_edge_ids_per_path,
    }
    
    return dict(
        te_inputs=te_inputs,
        optimal_mlu=opt,
        policy_embeds=policy_embeds,
        return_loss=True
    )


class TeLLMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config=config

        self.batch_size = config.loader.batch_size
        self.eval_batch_size = config.loader.eval_batch_size
        self.test_batch_size = config.loader.get('test_batch_size', config.loader.eval_batch_size)
        self.num_workers = config.loader.num_workers
        self.policy_align_mode = config.task.policy_align_mode

        # Store datasets by name
        self.train_datasets = {}
        self.val_datasets = {}
        self.test_datasets = {}

    def setup(self, stage: str) -> None:
        # Iterate through all datasets in the config
        for dataset_name, dataset_config in self.config.dataset.items():
            # Create train dataset
            if dataset_config.train_set.cluster is not None and not isinstance(dataset_config.train_set.cluster, ListConfig):
                train_set_config = dataset_config['train_set']
                self.train_datasets[dataset_name] = TEDatasetWithinCluster(
                    dataset_config,
                    train_set_config['cluster'],
                    train_set_config['start'],
                    train_set_config['end'],
                    self.policy_align_mode
                )
            elif dataset_config.train_set.cluster is not None and isinstance(dataset_config.train_set.cluster, ListConfig):
                # If cluster is a list, create multiple datasets for each cluster
                for cluster in dataset_config.train_set.cluster:
                    self.train_datasets[f"{dataset_name}_{cluster}"] = TEDatasetWithinCluster(
                        dataset_config,
                        cluster,
                        dataset_config.train_set.start,
                        dataset_config.train_set.end,
                        self.policy_align_mode
                    )

            # Create validation dataset
            if dataset_config.val_set.cluster is not None and not isinstance(dataset_config.val_set.cluster, ListConfig):
                val_set_config = dataset_config['val_set']
                self.val_datasets[dataset_name] = TEDatasetWithinCluster(
                    dataset_config,
                    val_set_config['cluster'],
                    val_set_config['start'],
                    val_set_config['end'],
                    self.policy_align_mode
                )

            elif dataset_config.val_set.cluster is not None and isinstance(dataset_config.val_set.cluster, ListConfig):
                # If cluster is a list, create multiple datasets for each cluster
                for cluster in dataset_config.val_set.cluster:
                    self.val_datasets[f"{dataset_name}_{cluster}"] = TEDatasetWithinCluster(
                        dataset_config,
                        cluster,
                        dataset_config.val_set.start,
                        dataset_config.val_set.end,
                        self.policy_align_mode
                    )
            # Create test dataset
            if hasattr(dataset_config, 'test_set') and dataset_config.test_set.cluster is not None:
                if not isinstance(dataset_config.test_set.cluster, ListConfig):
                    test_set_config = dataset_config['test_set']
                    self.test_datasets[dataset_name] = TEDatasetWithinCluster(
                        dataset_config,
                        test_set_config['cluster'],
                        test_set_config['start'],
                        test_set_config['end'],
                        self.policy_align_mode
                    )
                elif isinstance(dataset_config.test_set.cluster, ListConfig):
                    # If cluster is a list, create multiple datasets for each cluster
                    for cluster in dataset_config.test_set.cluster:
                        self.test_datasets[f"{dataset_name}_{cluster}"] = TEDatasetWithinCluster(
                            dataset_config,
                            cluster,
                            dataset_config.test_set.start,
                            dataset_config.test_set.end,
                            self.policy_align_mode
                        )

        # Create ConcatDataset for training and validation
        self.train_dataset = ConcatDataset(list(self.train_datasets.values()))
        self.val_dataset = ConcatDataset(list(self.val_datasets.values()))
        self.test_dataset = ConcatDataset(list(self.test_datasets.values())) if self.test_datasets else None
 


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        if len(self.val_datasets) == 1:
            dataset_name = list(self.val_datasets.keys())[0]
            dataset = self.val_datasets[dataset_name]
            
            return [DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
            )]
        else:
            val_loaders = []
            
            # Add global loader first
            global_val_loader = DataLoader(
                self.val_dataset,
                collate_fn=collate_fn,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
            )
            val_loaders.append(global_val_loader)
            
            # Add individual dataset loaders
            for dataset_name, dataset in self.val_datasets.items():
                val_loader = DataLoader(
                    dataset,
                    collate_fn=collate_fn,
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=False,
                )
                val_loaders.append(val_loader)
            
            return val_loaders
    def test_dataloader(self):
        if not self.test_datasets or len(self.test_datasets) == 0:
            return None
            
        if len(self.test_datasets) == 1:
            dataset_name = list(self.test_datasets.keys())[0]
            dataset = self.test_datasets[dataset_name]
            
            return [DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
            )]
        else:
            test_loaders = []
            
            # Add global loader first
            global_test_loader = DataLoader(
                self.test_dataset,
                collate_fn=collate_fn,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
            )
            test_loaders.append(global_test_loader)
            
            # Add individual dataset loaders
            for dataset_name, dataset in self.test_datasets.items():
                test_loader = DataLoader(
                    dataset,
                    collate_fn=collate_fn,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=False,
                )
                test_loaders.append(test_loader)
            
            return test_loaders


class TeLLMLightningModel(pl.LightningModule):
    def __init__(self, config: DictConfig, model):
        super().__init__()
        self.config = config
        self.model = model
        self.hist_alpha_base = config.task.get('hist_alpha_base', 500)
        self.global_alpha_base = config.task.get('global_alpha_base', 5)
        self.cost_alpha_base = config.task.get('cost_alpha_base', 1)
        # self.norm = config.model.solver_config.norm

    def forward(self, batch):
        return self.model(**batch, 
                          hist_alpha_base=self.hist_alpha_base,
                          global_alpha_base=self.global_alpha_base,
                          cost_alpha_base=self.cost_alpha_base
                          )

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        loss = outputs.loss.loss
        norm_mlu = outputs.loss.norm_mlu
        optimal = outputs.loss.optimal
        history_term = outputs.loss.history_term
        global_term = outputs.loss.global_term
        cost_term = outputs.loss.cost_term
        max_sensitivity = outputs.loss.max_sensitivity
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("norm_mlu", norm_mlu, prog_bar=True, sync_dist=True)
        self.log("avg_max_ratio", max_sensitivity, prog_bar=True, sync_dist=True)
        if history_term is not None:
            self.log("history_term", history_term, prog_bar=True, sync_dist=True)
            self.log("global_term", global_term, prog_bar=True, sync_dist=True)
            self.log("cost_term", cost_term, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)

        loss = outputs.loss.loss
        norm_mlu = outputs.loss.norm_mlu
        optimal = outputs.loss.optimal
        history_term = outputs.loss.history_term
        global_term = outputs.loss.global_term
        cost_term = outputs.loss.cost_term
        max_sensitivity = outputs.loss.max_sensitivity

        if dataloader_idx == 0:
            self.log("val_loss", loss, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            self.log("val_norm_mlu", norm_mlu, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            self.log("avg_max_ratio", max_sensitivity, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            if history_term is not None:
                self.log("val_history_term", history_term, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                self.log("val_global_term", global_term, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                self.log("val_cost_term", cost_term, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        else:
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            self.log("val_norm_mlu", norm_mlu, prog_bar=True, sync_dist=True)
            self.log("avg_max_ratio", max_sensitivity, prog_bar=True, sync_dist=True)
            if history_term is not None:
                self.log("val_history_term", history_term, prog_bar=True, sync_dist=True)
                self.log("val_global_term", global_term, prog_bar=True, sync_dist=True)
                self.log("val_cost_term", cost_term, prog_bar=True, sync_dist=True)
        
        return loss
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)

        loss = outputs.loss.loss
        norm_mlu = outputs.loss.norm_mlu
        optimal = outputs.loss.optimal
        history_term = outputs.loss.history_term
        global_term = outputs.loss.global_term
        cost_term = outputs.loss.cost_term
        max_sensitivity = outputs.loss.max_sensitivity

        if dataloader_idx == 0:
            self.log("test_loss", loss, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            self.log("test_norm_mlu", norm_mlu, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            self.log("test_avg_max_ratio", max_sensitivity, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            if history_term is not None:
                self.log("test_history_term", history_term, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                self.log("test_global_term", global_term, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                self.log("test_cost_term", cost_term, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        else:
            self.log("test_loss", loss, prog_bar=True, sync_dist=True)
            self.log("test_norm_mlu", norm_mlu, prog_bar=True, sync_dist=True)
            self.log("test_avg_max_ratio", max_sensitivity, prog_bar=True, sync_dist=True)
            if history_term is not None:
                self.log("test_history_term", history_term, prog_bar=True, sync_dist=True)
                self.log("test_global_term", global_term, prog_bar=True, sync_dist=True)
                self.log("test_cost_term", cost_term, prog_bar=True, sync_dist=True)
            
        sample_norm_mlu = norm_mlu
        if not hasattr(self, 'test_norm_mlu_results'):
            self.test_norm_mlu_results = []
        self.test_norm_mlu_results.append(sample_norm_mlu)
        return loss

    def on_test_epoch_end(self):
        if hasattr(self, 'test_norm_mlu_results'):
            import numpy as np
            norm_mlu_numpy = np.array(self.test_norm_mlu_results)
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            
            with open(f'{output_dir}/test_norm_mlu.txt', 'w') as f:
                f.write(f"# Total samples: {len(norm_mlu_numpy)}\n")
                f.write(f"# Mean: {norm_mlu_numpy.mean():.6f}\n")
                f.write(f"# Std: {norm_mlu_numpy.std():.6f}\n")
                f.write(f"# Min: {norm_mlu_numpy.min():.6f}\n")
                f.write(f"# Max: {norm_mlu_numpy.max():.6f}\n")
                f.write("# Sample values:\n")
            with open(f'{output_dir}/norm_mlu_list.txt', 'w') as file:
                file.write(str(norm_mlu_numpy.tolist()))
                    
            print(f"Saved {len(self.test_norm_mlu_results)} norm_mlu samples to {output_dir}/norm_mlu_list.txt")
            print(f"Norm MLU stats: mean={norm_mlu_numpy.mean():.4f}, std={norm_mlu_numpy.std():.4f}")


    def predict_step(self, batch, batch_idx, dataloader_idx=0): 

        outputs = self(batch)

        loss = outputs.loss.mlu
        norm_mlu = outputs.loss.norm_mlu
        optimal = outputs.loss.optimal
        history_term = outputs.loss.history_term
        global_term = outputs.loss.global_term
        cost_term = outputs.loss.cost_term
        max_sensitivity = outputs.loss.max_sensitivity

        return outputs

    def predict_on_dataset(self, dataset: Dataset, batch_size: int = 1, gpu_id: int = 0):
        """Predict on a given dataset."""
        
        trainer = pl.Trainer(
            devices=[gpu_id] if torch.cuda.is_available() else 0,
            logger=False,
            enable_checkpointing=False,
            callbacks=[RichProgressBar()],
        )
        predict_loader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    collate_fn=collate_fn,
                                    shuffle=False)
        predictions = trainer.predict(self, dataloaders=predict_loader)
        avg_metrics = self.log_prediction_averages(predictions)
        
        return predictions

    def log_prediction_averages(self, predictions):
        """
        Efficiently calculate and log average metrics across all predictions.
        
        Args:
            predictions: List of dictionaries containing prediction results
        
        Returns:
            Dictionary with average values for each specified metric
        """
        if not predictions:
            return {}

        metrics = [pred["loss"] for pred in predictions]
        
        # Use list comprehensions to extract each metric
        metrics = {
            "avg_max_ratio": [metric["max_sensitivity"] for metric in metrics],
            "val_norm_mlu": [metric["norm_mlu"] for metric in metrics],
            "val_history_term": [metric["history_term"] for metric in metrics]
        }
        # norm_mlu_list = torch.as_tensor(metrics['val_norm_mlu'])
        # print('all_norm_mlu', norm_mlu_list)
        # def summarize_torch(x):
        #     import math
        #     n = x.numel()
        #     if n == 0:
        #         return dict(mean=float("nan"), std=float("nan"), sem=float("nan"),
        #                     ci95_low=float("nan"), ci95_high=float("nan"), n=0)
            
        #     # PyTorch built-in functions
        #     mean = torch.mean(x)
        #     std = torch.std(x, unbiased=True) if n > 1 else torch.tensor(0.0)
        #     sem = std / math.sqrt(n) if n > 1 else torch.tensor(0.0)
            
        #     # 95% CI bounds
        #     ci_half = 1.96 * sem
        #     ci95_low = mean - ci_half
        #     ci95_high = mean + ci_half
            
        #     return dict(
        #         mean=mean.item(), 
        #         std=std.item(), 
        #         sem=sem.item(),
        #         ci95_low=ci95_low.item(), 
        #         ci95_high=ci95_high.item(), 
        #         n=n
        #     )
        # s_norm_mlu = summarize_torch(norm_mlu_list)
        
        # # Calculate averages in one go
        avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
        # print(f'statistics: {s_norm_mlu}')
        
        # Print a single formatted summary
        print("Prediction Averages:", 
            ", ".join(f"{k}: {v}" for k, v in avg_metrics.items()))
        
        return avg_metrics

    def configure_optimizers(self):
        # Collect all parameters that require gradients
        params = [param for param in self.parameters() if param.requires_grad]
        
        base_lr = self.config.training.optimizer.get('lr', 2e-3)  # default lr

        optimizer_cfg = dict(self.config.training.optimizer)  # make a copy to tweak
        optimizer_cfg.pop('lr', None)

        optimizer = instantiate(
            optimizer_cfg,
            [{'params': params, 'lr': base_lr}]
        )
        
        lr_scheduler = {
            "scheduler": get_scheduler(
                optimizer=optimizer, **self.config.training.scheduler
            ),
            "name": "learning_rate",
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

def init_model(config: OmegaConf):

    # solver_config = OmegaConf.to_container(config.model.solver_config, resolve=True)
    # tellm_config = TeLLMSolverConfig(**solver_config)

    # if config.training.pretrained_path is not None:
    #     ckpt_path = config.training.pretrained_path
    #     MODEL_ID = os.path.join(ckpt_path, "pretrained")
    #     config_path = os.path.join(ckpt_path, "outputs", "config.yaml")
    #     ori_config= OmegaConf.load(config_path)
    #     print_config(ori_config, resolve=True)

    #     model = AutoModel.from_pretrained(
    #         MODEL_ID,
    #         trust_remote_code=True,
    #     )
    #     if config.training.train_moe:
    #         moe_config = OmegaConf.to_container(config.model.moe_config, resolve=True)
    #         model.config.update(moe_config)
    #         moedecoder = TeLLMHarpMoE(model.config)
    #         moedecoder.load_pretrained_weights(model.solver, freeze_non_moe=True)
    #         model.solver = moedecoder
    # else:
    #     model = TeLLMSolverModel(config=tellm_config)
    baseline = config.baseline
    modelconfig = OmegaConf.load(f"configs/model/{baseline}.yaml")
    
    num_nodes = config.dataset[list(config.dataset.keys())[0]].get('num_nodes', 12)
    modelconfig.num_nodes = num_nodes
    
    # temp_config = {
    #     "hist_len": 12,
    #     "num_nodes": num_nodes,
    #     "num_paths": 4,
    #     "num_layers": 2,
    #     "alpha": 0.0
    # }
    # temp_config = OmegaConf.create(temp_config)
    BASELINES = {
        "figret": FigretTeModel,
        "harp": HARPTeModel,
        "dote": DoteTeModel,
    }
    model = BASELINES[baseline](modelconfig)

    # if config.training.use_lora:
    #     lora_config = LoraConfig(
    #         target_modules=find_all_linear_names(model),
    #         **config.training.peft,
    #         # init_lora_weights="gaussian",
    #     )
    #     model = get_peft_model(model, lora_config)

    return model 

def train(config: OmegaConf):
    if config.seed is not None:
        pl.seed_everything(config.seed, workers=True)

    model = init_model(config)

    model_module = TeLLMLightningModel(config, model)

    dm = TeLLMDataModule(config)

    logger = WandbLogger(
        save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        project=config.project.name,
        tags=config.project.tags,
        name=config.run_name,
        config = OmegaConf.to_container(config, resolve=True)
    )
    logger.experiment.save(f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/config_tree.txt", policy="now")

    logger.log_hyperparams(config)

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval=None)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
        dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        + "/checkpoints/",
        filename="best-{epoch:02d}-{val_norm_mlu:.2f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=8, 
        verbose=True, 
        mode="min",
        log_rank_zero_only=True,
        )
    class PrintChangedWeightsCallback(pl.Callback):
        def __init__(self):
            super().__init__()
            self.previous_weights = {}

        def on_train_epoch_end(self, trainer, pl_module):
            changed = False
            for name, param in pl_module.named_parameters():
                if name not in self.previous_weights:
                    self.previous_weights[name] = param.detach().cpu().clone()
                    print(f"Initial weights for {name}:\n{param.data}\n")
                    changed = True
                else:
                    if not torch.equal(self.previous_weights[name], param.detach().cpu()):
                        print(f"Weights changed for {name}:\n{param.data}\n")
                        self.previous_weights[name] = param.detach().cpu().clone()
                        changed = True
            if not changed:
                print("No parameter weights changed this epoch.\n")

    trainer = Trainer(
        logger=logger,
        **config.training.trainer,
        callbacks=[
            lr_monitor_callback,
            checkpoint_callback,
            early_stop_callback,
            RichProgressBar(leave=True, refresh_rate=5),
            # PrintChangedWeightsCallback()
        ],
    )

    if config.training.resume_from_ckpt is not None:
        trainer.fit(model_module, datamodule=dm, ckpt_path=config.training.resume_from_ckpt)
    else:
        trainer.fit(model_module, datamodule=dm)
    trainer.test(ckpt_path='best', datamodule=dm)

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.2.0")
def main(config: OmegaConf):
    # Print Hydra path
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(
        f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    # Pretty print config using Rich library
    print_config(
        config,
        resolve=True,
        save_cfg=True,
        cfg_path=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )

    train(config)


if __name__ == "__main__":
    main()
