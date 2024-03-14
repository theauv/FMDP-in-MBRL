import wandb
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

NEURIPS_TEXT_WIDTH = 397.48499
NEURIPS_FONT_FAMILY = "Times New Roman"

#'episode_reward', 'trajectory_optimization_iteration', '_timestamp',
#'_step', 'reward_r2', 'eval_r2_score',
#'Mean relative error pred_reward', 'env_step', 'best_eval_score',
#'Mean relative error pred_obs', 'DBN graph', 'eval_score',
#'train_epoch', 'total_avg_loss', 'reward_eval_score', 'train_iteration',
#'env_episode', 'episode_steps', '_runtime', 'train_r2_score', 'Env',
#'dynamics_r2', 'dynamics_eval_score', 'best_trajectory_value'

FORMAT_HYPERGRID={
    "Mean_relative_error": {
        'x': 'env_step',
        'y': 'Mean relative error pred_obs',
        'name': 'Mean relative error on predicted next state',
        'x_label': 'Iteration',
        'y_label': 'Mean relative error',
        'x_lim': (0,25000),
        'y_lim': (0,2),
        'legend_position': 'upper right',
        'alpha': 0.8,
    },
    "episode_reward": {
        'x': 'env_episode',
        'y': 'episode_reward',
        'name': 'Cumulative reward',
        'x_label': 'Episode',
        'y_label': 'Reward',
        'x_lim': None,
        'y_lim': None,
        'legend_position': 'lower right',
        'alpha': 0.8,
    },
    # "env_episode": {
    #     'y': 'env_step',
    #     'x': 'env_episode',
    #     'name': 'Training speed in step',
    #     'y_label': 'Iteration',
    #     'x_label': 'Number of episodes',
    #     'x_lim': None,
    #     'y_lim': None,
    #     'legend_position': 'upper left',
    # },
    # "best_trajectory_value": {
    #     'x': 'trajectory_optimization_iteration',
    #     'y': 'best_trajectory_value',
    #     'name': 'Planning Cumulative Reward',
    #     'x_label': 'Cumulative planning step',
    #     'y_label': 'Reward',
    #     'x_lim': None,
    #     'y_lim': None,
    #     'legend_position': 'lower right',
    #     'alpha': 0.8,
    # }
    # "Training_MSE": {
    #     'x': 'train_epoch',
    #     'y': 'eval_score',
    #     'name': 'Training Loss',
    #     'x_label': 'Training epoch',
    #     'y_label': 'MSE',
    #     'x_lim': None,
    #     'y_lim': None,
    #     'legend_position': 'upper right',
    # },
}

FILTERS_HYPERGRID=[
    {
        "config.overrides.env_config.grid_dim": 7,
        "config.dynamics_model.model_trainer.optim_lr": 0.001,
    },
    {
        #"config.overrides.env_config.grid_dim": 3,
        "tags": {"$in": ["comparison"]},
    },
    # {
    #     #"config.overrides.env_config.grid_dim": 3,
    #     "config.dynamics_model.model_trainer.optim_lr": 0.001,
    #     "$or": [ 
    #         {"config.dynamics_model.model._target_": {"$eq": "src.model.gaussian_process.MultiOutputGP"}}, 
    #         {"config.dynamics_model.model._target_": {"$eq": "src.model.gaussian_process.FactoredMultiOutputGP"}},
    #     ],
    # },
    # {
    #     "$or": [ 
    #         {"config.overrides.env_config.grid_dim": 3},
    #         {"config.overrides.env_config.grid_dim": 7},
    #     ],
    #     "config.dynamics_model.model_trainer.optim_lr": 0.01,
    #     "$or": [
    #         {"config.dynamics_model.model._target_": {"$eq": "src.model.simple.Simple"}}, 
    #         {"config.dynamics_model.model._target_": {"$eq": "src.model.simple.FactoredSimple"}},
    #         #{"config.dynamics_model.model._target_": {"$eq": "src.model.lasso_net.LassoSimple"}},
    #     ],
    # },
]


def set_up_plt(font_family):

    plt.rcParams["text.usetex"] = True

    SMALL_SIZE = 12
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 24

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams["font.family"] = font_family


def set_size(width, fraction=1, subplots=(1, 1)):
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt - 1
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def get_runs_from_wandb(group_name, filters):
    api = wandb.Api()
    filters["group"]=group_name
    runs = api.runs("hucrl_fmdp/hucrl_fmdp", filters = filters)
    print(len(runs))
    return runs

def get_run_name(config):
    agent=config.get('agent')
    if agent is not None:
        if agent == 'random':
            agent = 'Random agent'
        elif agent == 'good_heuristic':
            agent = 'Heuristic'
        else:
            raise ValueError(f"No model named {agent}")
        return f'{agent}'
    else:
        overrides = config['overrides']
        model = config['dynamics_model']['model']
        model_name = model['_target_'].split('.')[-1]
        if model_name == 'Simple':
            model_name = 'NN'
        elif model_name == 'FactoredSimple':
            model_name = 'Factored NN'
        elif model_name == 'LassoSimple':
            model_name = 'Lassonets'
        elif model_name == 'MultiOutputGP':
            model_name = 'GP'
        elif model_name == 'FactoredMultiOutputGP':
            model_name = 'Factored GP'
        else:
            raise ValueError(f"No model named {model_name}")
        if overrides['env'] == 'hypergrid':
            grid_dim = overrides['env_config']['grid_dim']
            return f'{model_name} dim={grid_dim}'
        else:
            return f'{model_name}'

def get_df_from_runs(runs, formats):
    hist_dict = {key:[] for key in formats.keys()}
    print(hist_dict)
    for run in runs:
        print(run)
        for run_name, f in formats.items():
            hist = run.history(keys=[f['x'], f['y']]) #run.history(keys=['epoch', 'val/loss'])
            print(hist.keys())
            hist['name'] = get_run_name(run.config)
            for previous_hist in hist_dict[run_name]:
                if hist['name'][0] == previous_hist['name'][0]:
                    hist[f['x']]=previous_hist[f['x']]
            hist_dict[run_name].append(hist)
            print({key:len(h) for key, h in hist_dict.items()})
            print({key:[frame.shape for frame in h] for key, h in hist_dict.items()})
    df_dict = {key: pd.concat(value, ignore_index=True) for key, value in hist_dict.items()}
    print({key:h.shape for key, h in df_dict.items()})
    return df_dict

def plot_from_df(group_name, df_dict, formats, save):
    for run_name, df in df_dict.items():
        from time import time
        start = time()
        print(df)
        matplotlib.use("TkAgg")
        set_up_plt(NEURIPS_FONT_FAMILY)
        set_size(NEURIPS_TEXT_WIDTH)
        print(time()-start)
        f=formats[run_name]
        fig=sns.lineplot(x=f['x'], y=f['y'], hue='name', data=df)
        print(time()-start)
        #Additional formatting TODO
        fig.legend_.set_title(None)
        if f['x_lim']:
            plt.xlim(f['x_lim'])
        if f['y_lim']:
            plt.ylim(f['y_lim'])
        if f.get('alpha') is not None:
            plt.setp(fig.lines, alpha=f['alpha'])
        plt.legend(loc=f['legend_position'])
        plt.xlabel(f['x_label'])
        plt.ylabel(f['y_label'])
        plt.title(f['name'])
        print(time()-start)
        if save:
            my_dir=f"results/{group_name}"
            Path(my_dir).mkdir(parents=True, exist_ok=True)
            name=f"rename_{f['name']}"
            plt.savefig(f"{my_dir}/{name}.png")
            plt.show()
        else:
            plt.show()

def main(group_name, filters, formats, save):
    for filter in filters:
        runs = get_runs_from_wandb(group_name=group_name, filters=filter)
        df_dict = get_df_from_runs(runs, formats)
        plot_from_df(group_name, df_dict, formats, save)



if __name__=="__main__":
    group_name='hypergrid_58c15745c0d1c8d5dff046d96cccecfa4e6e3a0b'
    filters=FILTERS_HYPERGRID
    formats=FORMAT_HYPERGRID
    save=False

    main(group_name, filters, formats, save)
