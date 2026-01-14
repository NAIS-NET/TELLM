#!/usr/bin/env python3
import os
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import json
import logging
from rich.logging import RichHandler

def setup_rich_logging():
    rich_handler = RichHandler(rich_tracebacks=True, show_path=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[rich_handler],
        force=True
    )
    import transformers
    transformers.logging.set_verbosity_info()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
#     handlers=[RichHandler()]
# )

@rank_zero_only
def print_config(
    config: DictConfig,
    resolve: bool = True,
    save_cfg=False,
    cfg_path=""
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    if save_cfg:
        cfg_path = os.path.join(cfg_path, "config_tree.txt")
        with open(cfg_path, "w") as fp:
            rich.print(tree, file=fp)
    return tree

def display_result(metrics):
    """Compact one-line display"""
    # m = {k: v.cpu().item() for k, v in metrics.items()}
    m = {k: v.cpu().item() if hasattr(v, 'cpu') else v for k, v in metrics.items()}

    from rich.console import Console
    from rich.table import Table

    console = Console()
    # Create table
    table = Table()
    # Add columns
    # for column in m.keys():
    table.add_column('Metrics', style="cyan")
    table.add_column('Results', style="purple")

    for key, value in m.items():
        table.add_row(key, str(value))
    console.print(table)


