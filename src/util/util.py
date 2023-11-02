import git
import logging
from typing import Dict, Union, Tuple, List
import yaml
import warnings

import omegaconf
import numpy as np
import torch

import mbrl
from mbrl.models.model import Model
from mbrl.models.gaussian_mlp import GaussianMLP
from lassonet.model import LassoNet


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

    if wandb_config.group is not None:
        wandb_config.group = f"{wandb_config.group}_{sha}"
    else:
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

def getBack(var_grad_fn):
    """
    Observe the backward graph functions of a tensor
    :param var_grad_fn: loss.grad_fn
    """
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def model_correlations(
    model: Model,
    bounds=Union[Tuple, List[Tuple]],
    num_iters: int = 1,
    num_inputs: int = 100,
):

    if isinstance(bounds, Tuple):
        assert len(bounds) == 2
        bounds = [bounds for i in range(model.in_size)]
    else:
        assert len(bounds[0]) == 2

    # Compute the outputs of reference
    correlations = np.zeros((model.out_size, model.in_size))
    for n in range(num_iters):
        for i in range(model.out_size):
            # input_ref = np.empty(model.in_size)
            # for idx, bound in enumerate(bounds):
            #     input_ref[idx] = np.random.uniform(bound[0], bound[1], 1)
            # input_ref = torch.tensor(input_ref).float()
            input_ref = torch.zeros(model.in_size).float()
            ref_outputs = model.forward(input_ref)
            if isinstance(ref_outputs, Tuple):
                ref_outputs = ref_outputs[0].squeeze()
            ref_outputs = ref_outputs.detach()
            ref_output = ref_outputs[i]
            for j in range(model.in_size):
                new_input = input_ref.clone()
                low = bounds[j][0]
                high = bounds[j][1]
                correlation = 0
                for k in range(num_inputs):
                    new_input[j] = low + (high - low) / (k + 1)
                    outputs = model.forward(new_input)
                    if isinstance(outputs, Tuple):
                        outputs = outputs[0].squeeze()
                    output = outputs.detach()[i]
                    correlation += abs(output - ref_output)
                correlations[i, j] += correlation

    max_corr = np.max(correlations, axis=1)
    max_corr = np.where(max_corr <= 0.0, 1, max_corr)
    max_corr = np.tile(np.expand_dims(max_corr, axis=-1), model.in_size)
    return np.round(correlations / max_corr, 3)


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
            elif "hidden_layers" in names or "layers" in names:
                if "bias" in names:
                    bias = torch.squeeze(param).numpy()
                    biases.append(bias)
                elif "weight" in names:
                    weight = torch.squeeze(param).numpy()
                    weights.append(weight)
                else:
                    raise ValueError(f"Weird hidden layer named {name}")

        # if isinstance(model, LassoNetGaussianMLP):
        #     all_weights = {
        #         "hidden_weights": weights,
        #         "hidden_biases": biases,
        #         "mean_and_logvar_weights": mean_and_logvar_weights,
        #         "mean_and_logvar_biases": mean_and_logvar_biases,
        #         "skip": skip,
        #     }
        #     if not deterministic:
        #         all_weights["min_logvar"] = min_logvar
        #         all_weights["max_logvar"] = max_logvar
        if isinstance(model, GaussianMLP):
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
