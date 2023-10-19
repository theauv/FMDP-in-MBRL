import hydra
import omegaconf
import pathlib
import git
import logging
import torch
from typing import Dict, Union, Tuple, Optional
import yaml
import warnings

from mbrl.models.gaussian_mlp import GaussianMLP
from lassonet.model import LassoNet

from src.model.gaussian_lasso_net import LassoNetGaussianMLP


def get_weights_model(model: torch.nn.Module, verbose: bool = False):

    with torch.no_grad():

        biases = []
        weights = []
        deterministic = True
        for i, (name, param) in enumerate(model.named_parameters()):

            if verbose:
                print(name)
                print(param.shape)

            names = name.split(".")

            if "skip" in names:
                skip = torch.squeeze(param).numpy()
            if "hidden_layers" in names or "layers" in names:
                if "bias" in names:
                    bias = torch.squeeze(param).numpy()
                    biases.append(bias)
                elif "weight" in names:
                    weight = torch.squeeze(param).numpy()
                    weights.append(weight)
                else:
                    raise ValueError(f"Weird hidden layer named {name}")
            elif "mean_and_logvar" in names:
                if "bias" in names:
                    mean_and_logvar_biases = torch.squeeze(param).numpy()
                elif "weight" in names:
                    mean_and_logvar_weights = torch.squeeze(param).numpy()
                else:
                    raise ValueError(f"Weird mean_and_logvar named {name}")
            elif "min_logvar" in names:
                deterministic = False
                min_logvar = torch.squeeze(param).numpy()
            elif "max_logvar" in names:
                max_logvar = torch.squeeze(param).numpy()

        if isinstance(model, LassoNetGaussianMLP):
            all_weights = {
                "hidden_weights": weights,
                "hidden_biases": biases,
                "mean_and_logvar_weights": mean_and_logvar_weights,
                "mean_and_logvar_biases": mean_and_logvar_biases,
                "skip": skip,
            }
            if not deterministic:
                all_weights["min_logvar"] = min_logvar
                all_weights["max_logvar"] = max_logvar
        elif isinstance(model, GaussianMLP):
            all_weights = {
                "hidden_weights": weights,
                "hidden_biases": biases,
                "mean_and_logvar_weights": mean_and_logvar_weights,
                "mean_and_logvar_biases": mean_and_logvar_biases,
            }
            if not deterministic:
                all_weights["min_logvar"] = min_logvar
                all_weights["max_logvar"] = max_logvar
        elif isinstance(model, LassoNet):
            all_weights = {
                "hidden_weights": weights,
                "hidden_biases": biases,
                "skip": skip,
            }
        else:
            all_weights = {"hidden_weights": weights, "hidden_biases": biases}
            warnings.warn("This function might not support this model architecture")
        return all_weights


def create_one_dim_tr_model_overriden(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """
    This function overrides the mbrl equivalent functions to be able to handle
    a different configuration file (allows integration of new model trainer etc..)
    
    Creates a 1-D transition reward model from a given configuration.

    This method creates a new model from the given configuration and wraps it into a
    :class:`mbrl.models.OneDTransitionRewardModel` (see its documentation for explanation of some
    of the config args under ``cfg.algorithm``).
    The configuration should be structured as follows::

        -cfg
          -dynamics_model
            -model
              -_target_ (str): model Python class
              -in_size (int, optional): input size
              -out_size (int, optional): output size
              -model_arg_1
               ...
              -model_arg_n
          -algorithm
            -learned_rewards (bool): whether rewards should be learned or not
            -target_is_delta (bool): to be passed to the dynamics model wrapper
            -normalize (bool): to be passed to the dynamics model wrapper
          -overrides
            -no_delta_list (list[int], optional): to be passed to the dynamics model wrapper
            -obs_process_fn (str, optional): a Python function to pre-process observations
            -num_elites (int, optional): number of elite members for ensembles

    If ``cfg.dynamics_model.in_size`` is not provided, it will be automatically set to
    `obs_shape[0] + act_shape[0]`. If ``cfg.dynamics_model.out_size`` is not provided,
    it will be automatically set to `obs_shape[0] + int(cfg.algorithm.learned_rewards)`.

    The model will be instantiated using :func:`hydra.utils.instantiate` function.

    Args:
        cfg (omegaconf.DictConfig): the configuration to read.
        obs_shape (tuple of ints): the shape of the observations (only used if the model
            input or output sizes are not provided in the configuration).
        act_shape (tuple of ints): the shape of the actions (only used if the model input
            is not provided in the configuration).
        model_dir (str or pathlib.Path): If provided, the model will attempt to load its
            weights and normalization information from "model_dir / model.pth" and
            "model_dir / env_stats.pickle", respectively.

    Returns:
        (:class:`mbrl.models.OneDTransitionRewardModel`): the model created.

    """
    # This first part takes care of the case where model is BasicEnsemble and in/out sizes
    # are handled by member_cfg
    model_cfg = cfg.dynamics_model.model  # Changed from the original function
    if model_cfg._target_ == "mbrl.models.BasicEnsemble":
        model_cfg = model_cfg.member_cfg
    if model_cfg.get("in_size", None) is None:
        model_cfg.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    if model_cfg.get("out_size", None) is None:
        model_cfg.out_size = obs_shape[0] + int(cfg.algorithm.learned_rewards)

    # Now instantiate the model
    model = hydra.utils.instantiate(model_cfg)  # Changed from the original function

    name_obs_process_fn = cfg.overrides.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn)
    else:
        obs_process_fn = None
    dynamics_model = hydra.utils.instantiate(
        cfg.dynamics_model.wrapper,
        model,
        target_is_delta=cfg.algorithm.target_is_delta,
        normalize=cfg.algorithm.normalize,
        normalize_double_precision=cfg.algorithm.get(
            "normalize_double_precision", False
        ),
        learned_rewards=cfg.algorithm.learned_rewards,
        obs_process_fn=obs_process_fn,
        no_delta_list=cfg.overrides.get("no_delta_list", None),
        num_elites=cfg.overrides.get("num_elites", None),
    )
    if model_dir:
        dynamics_model.load(model_dir)

    return dynamics_model


def get_run_kwargs(configs: omegaconf.DictConfig,) -> Dict:
    """
    Gather the important informations to initialize the wandb api

    :param configs: general configs (see configs directory)
    :return: arguments to iniate wandb
    """

    experiment_config = configs.experiment
    wandb_config = experiment_config.run_configs
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
