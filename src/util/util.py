import git
import logging
from typing import Dict, Union, Tuple, List
import yaml
import warnings

import omegaconf
import os
import numpy as np
import torch

import mbrl
from mbrl.models.model import Model
from mbrl.models.gaussian_mlp import GaussianMLP
from lassonet.model import LassoNet


def get_mapping_dict(dict_space):
    mapping = {}
    length = 0
    for key, value in dict_space.items():
        new_length = value.shape[0]
        mapping[key] = slice(length, length + new_length)
        length += new_length
    mapping["length"] = length
    return mapping


def get_base_dir_path():
    cwd = os.getcwd()
    path = os.path.normpath(cwd)
    chunks = path.split(os.sep)[1:]
    base_dir = ""
    for chunk in chunks:
        base_dir = base_dir + os.sep + chunk
        if chunk == "HUCRL_for_FMDP":  # TODO: HARD-CODED
            break
    base_dir += os.sep
    return base_dir


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


def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0,
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items
