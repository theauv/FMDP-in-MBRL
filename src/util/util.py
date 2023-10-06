from typing import Dict, Union
import yaml
import omegaconf
import git
import logging


def get_run_kwargs(configs: omegaconf.DictConfig,) -> Dict:
    """
    Gather the important informations to initialize the wandb api

    :param configs: general configs (see configs directory)
    :return: arguments to iniate wandb
    """

    experiment_config = configs.experiment
    wandb_config = experiment_config.wandb
    # Rename the run
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    wandb_config.name = f"{wandb_config.name}_{sha}"

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # TODO: Add pipeline recover from a checkpoint ? (Might be done in another process function)

    init_run_kwargs = omegaconf.OmegaConf.to_container(wandb_config, resolve=True)
    init_run_kwargs["config"] = omegaconf.OmegaConf.to_container(configs, resolve=True)

    return init_run_kwargs


def convert_yaml_config(
    config_path="configs/experiment/wandb.yaml",
    overrides: Dict = {},
    dictconfig: bool = True,
) -> Union[Dict, omegaconf.DictConfig]:
    """
    Convert a yaml file to a Dict or a DictConfig
    :param dictconfig: whether to return a Dict or a DictConfig, defaults to True
    """

    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Update with overrides
    config.update(overrides)

    # Create a DictConfig with omegaconf
    if dictconfig:
        return omegaconf.OmegaConf.create(config)

    return config
