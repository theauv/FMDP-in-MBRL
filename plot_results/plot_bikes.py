import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np

from plot_results.plots_util import (
    set_up_plt,
    set_size,
    get_runs_from_wandb,
    get_run_name,
)

NEURIPS_TEXT_WIDTH = 397.48499
NEURIPS_FONT_FAMILY = "Times New Roman"

#'episode_reward', 'trajectory_optimization_iteration', '_timestamp',
#'_step', 'reward_r2', 'eval_r2_score',
#'Mean relative error pred_reward', 'env_step', 'best_eval_score',
#'Mean relative error pred_obs', 'DBN graph', 'eval_score',
#'train_epoch', 'total_avg_loss', 'reward_eval_score', 'train_iteration',
#'env_episode', 'episode_steps', '_runtime', 'train_r2_score', 'Env',
#'dynamics_r2', 'dynamics_eval_score', 'best_trajectory_value'

FORMAT_BIKES = {
    "Mean_relative_error": {
        "x": "env_step",
        "y": "Mean relative error pred_reward",
        "name": "Mean relative error on predicted reward",
        "x_label": "Iteration",
        "y_label": "Mean relative error",
        "x_lim": (0, 600),
        "y_lim": (0, 1),
        "legend_position": "upper right",
        "alpha": 0.8,
        "m": 0.1,
        "max": 1,
    },
    "Missclassified_bikes": {
        "x": "env_step",
        "y": "Missclassified pred_bike_allocations ratio",
        "name": "Missclassified predicted bike allocations ratio",
        "x_label": "Iteration",
        "y_label": "Missclassification ratio",
        "x_lim": (0, 600),
        "y_lim": (0, 1),
        "legend_position": "upper right",
        "alpha": 0.8,
        "m": 0.1,
    },
    "episode_reward": {
        "x": "env_episode",
        "y": "episode_reward",
        "name": "Cumulative reward",
        "x_label": "Episode",
        "y_label": "Reward",
        "x_lim": (0, 150),
        "y_lim": (0, 4),
        "legend_position": "lower right",
        "alpha": 0.8,
        "m": 0.1,
    },
}

FILTERS_BIKES = [
    {
        "$or": [
            {"config.algorithm.rescale_output": {"$eq": True}},
            {"config.agent": {"$eq": "random"}},
            {"config.agent": {"$eq": "good_heuristic"}},
        ]
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
    #         {"config.dynamics_model.model._target_": {"$eq": "src.model.simple.FFNN"}},
    #         {"config.dynamics_model.model._target_": {"$eq": "src.model.simple.FactoredFFNN"}},
    #         #{"config.dynamics_model.model._target_": {"$eq": "src.model.lasso_net.LassoFFNN"}},
    #     ],
    # },
]


def get_df_from_runs(runs, formats):
    hist_dict = {key: [] for key in formats.keys()}
    print(hist_dict)
    for run in runs:
        print(run)
        for run_name, f in formats.items():
            hist = run.history(keys=[f["x"], f["y"]])
            print(hist)
            if not hist.empty:
                rescale = f.get("max")
                col = hist[f["y"]]
                if rescale is not None:
                    col[col > float(rescale)] = float(rescale)
                hist[f["y"]] = col
                alpha = f.get("m")
                if alpha is not None:
                    col = col.ewm(alpha=f["m"]).mean()
                hist["smooth"] = col
                x_max = f["x_lim"]
                if x_max is not None:
                    x_max = int(x_max[1])
                    hist = hist.loc[hist[f["x"]] <= x_max]
                hist["name"] = get_run_name(run.config)
                # for previous_hist in hist_dict[run_name]:
                #     if hist['name'][0] == previous_hist['name'][0]:
                #         hist[f['x']]=previous_hist[f['x']]
                hist_dict[run_name].append(hist)
    df_dict = {
        key: pd.concat(value, ignore_index=True) for key, value in hist_dict.items()
    }
    print(df_dict)
    print("yeaaaah", {key: h.shape for key, h in df_dict.items()})
    return df_dict


def plot_from_df(group_name, df_dict, formats, save):
    for run_name, df in df_dict.items():
        from time import time

        start = time()
        print(df)
        matplotlib.use("TkAgg")
        set_up_plt(NEURIPS_FONT_FAMILY)
        set_size(NEURIPS_TEXT_WIDTH)
        print(time() - start)
        f = formats[run_name]

        df_random = df[df.name == "Random agent"]
        df_heuristic = df[df.name == "Heuristic"]
        random_mean = df_random[f["y"]].mean()
        random_std = df_random[f["y"]].std()
        heuristic_mean = df_heuristic[f["y"]].mean()
        heuristic_std = df_heuristic[f["y"]].std()

        print("YEAAAAAAAAH")
        print(random_mean, random_std, heuristic_mean, heuristic_std)

        df = df[df.name != "Random agent"]
        df = df[df.name != "Heuristic"]
        fig = sns.lineplot(x=f["x"], y="smooth", hue="name", data=df, errorbar=None)
        # Additional formatting TODO
        fig.legend_.set_title(None)
        if f["x_lim"]:
            plt.xlim(f["x_lim"])
        if f["y_lim"]:
            plt.ylim(f["y_lim"])
        if f.get("alpha") is not None:
            plt.setp(fig.lines, alpha=f["alpha"])
        sns.lineplot(
            x=f["x"],
            y=f["y"],
            hue="name",
            data=df,
            estimator=None,
            alpha=0.3,
            legend=False,
        )

        if not np.isnan(random_mean):
            n_steps = f["x_lim"][1]
            data = np.random.normal(random_mean, random_std, n_steps)
            # Calculate mean and standard deviation
            soft_red = (0.9, 0.7, 0.7)
            mean_data = np.full_like(data, random_mean)
            std_data = np.full_like(data, random_std)
            sns.lineplot(
                x=np.arange(n_steps),
                y=mean_data,
                color=soft_red,
                label="Random Agent",
                alpha=0.7,
            )
            plt.fill_between(
                np.arange(n_steps),
                mean_data - std_data,
                mean_data + std_data,
                color=soft_red,
                alpha=0.2,
            )

            data = np.random.normal(heuristic_mean, heuristic_std, n_steps)
            # Calculate mean and standard deviation
            soft_green = (0.7, 0.9, 0.7)
            mean_data = np.full_like(data, heuristic_mean)
            std_data = np.full_like(data, heuristic_std)
            sns.lineplot(
                x=np.arange(n_steps),
                y=mean_data,
                color=soft_green,
                label="Heuristic",
                alpha=0.7,
            )
            plt.fill_between(
                np.arange(n_steps),
                mean_data - std_data,
                mean_data + std_data,
                color=soft_green,
                alpha=0.2,
            )
        plt.legend(loc=f["legend_position"])
        plt.xlabel(f["x_label"])
        plt.ylabel(f["y_label"])
        plt.title(f["name"])
        print(time() - start)
        if save:
            my_dir = f"results/{group_name}"
            Path(my_dir).mkdir(parents=True, exist_ok=True)
            name = f"rename_{f['name']}"
            plt.savefig(f"{my_dir}/{name}.png")
            plt.show()
        else:
            plt.show()


def main(group_name, filters, formats, save):
    for filter in filters:
        runs = get_runs_from_wandb(group_name=group_name, filters=filter)
        df_dict = get_df_from_runs(runs, formats)
        plot_from_df(group_name, df_dict, formats, save)


if __name__ == "__main__":
    # real_4step_21030a797c1fe09b091f6ce98644cdd97362c20b
    group_names = [
        #'real_4step_21030a797c1fe09b091f6ce98644cdd97362c20b',
        "art_4step_94812f872e89705bfbf3a64702608ffac2e35615",
        #'real_4step_20centroid_21030a797c1fe09b091f6ce98644cdd97362c20b',
        #'art_4step_20centroid_94812f872e89705bfbf3a64702608ffac2e35615',
    ]

    for group_name in group_names:
        filters = FILTERS_BIKES
        formats = FORMAT_BIKES
        save = True

        if group_name in [
            "real_4step_20centroid_21030a797c1fe09b091f6ce98644cdd97362c20b",
            "art_4step_20centroid_94812f872e89705bfbf3a64702608ffac2e35615",
        ]:
            for key, format in formats.items():
                if key == "episode_reward":
                    format["x_lim"] = (0, 300)
                else:
                    format["x_lim"] = (0, 1200)

        main(group_name, filters, formats, save)
