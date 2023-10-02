from typing import Dict
import os
import yaml
import omegaconf


def convert_yaml_configs(
    overrides: Dict = {}, configs_directory_path="configs", filenames=[]
):

    # Gather each yaml dictionary in one
    configs = {}
    for filename in os.listdir(configs_directory_path):
        if filenames and filename in filenames:
            f = os.path.join(configs_directory_path, filename)
            with open(f, "r") as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            configs.update(config)

    # Update with overrides
    configs.update(overrides)

    # Create a DictConfig with omegaconf
    config_final = omegaconf.OmegaConf.create(configs)

    return config_final


def get_configs(overrides: Dict = {}, configs_directory_path="configs"):
    """_summary_

    :param overrides: should be of the form:
    {
        model: {
            key_to_override: value
        }
        agent: {
            key_to_override: value
        }
        env: {
            key_to_override: value
        }
        training: {
            key_to_override: value
        }
    }
    :param configs_directory_path: configs folder
    :return: config dictionary required to set up the model, agent and the environment
    """

    model_filenames = ["algorithm.yaml", "dynamics_model.yaml", "experiment.yaml"]
    env_filenames = ["env.yaml"]
    agent_filenames = ["agent.yaml"]
    training_filenames = ["training.yaml"]
    model_overrides = overrides["model"] if overrides else {}
    env_overrides = overrides["env"] if overrides else {}
    agent_overrides = overrides["agent"] if overrides else {}
    training_overrides = overrides["training"] if overrides else {}

    model_config = convert_yaml_configs(
        model_overrides, configs_directory_path, filenames=model_filenames
    )
    agent_config = convert_yaml_configs(
        agent_overrides, configs_directory_path, filenames=agent_filenames
    )
    env_config = convert_yaml_configs(
        env_overrides, configs_directory_path, filenames=env_filenames
    )
    training_config = convert_yaml_configs(
        training_overrides, configs_directory_path, filenames=training_filenames
    )

    model_config.overrides.num_steps = (
        model_config.overrides.trial_length * model_config.overrides.num_trials
    )

    model_config.dynamics_model.device = agent_config.optimizer_cfg.device

    return model_config, agent_config, env_config, training_config
