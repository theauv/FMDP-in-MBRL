import wandb
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np
from plots_util import set_up_plt, set_size, NEURIPS_TEXT_WIDTH, NEURIPS_FONT_FAMILY

def load_all_files(directory):
    dirpath = Path(directory)
    print(directory)
    assert dirpath.is_dir()
    assert directory.split('/')[-1].split('_')[0]=='factor'
    print(directory.split('/')[-1])
    factor=int(directory.split('/')[-1].split('_')[1])
    print(factor)
    theta_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            array=np.load(x)
            lambdas=np.arange(0, 50, 0.1)
            lambda_max=20
            lambda_max_idx=np.where(lambdas==lambda_max)[0][0]
            array=np.abs(array[:, :lambda_max_idx])
            array[array>10]=0
            col = [f'{i}' for i in range(array.shape[0])]
            df=pd.DataFrame(array.T, columns = col)
            df['lambda']=lambdas[:lambda_max_idx]
            theta_list.append(df)
        else:
            raise ValueError
    df = pd.concat(theta_list)
    return df

def get_dfs_factors(directory):
    dirpath = Path(directory)
    assert dirpath.is_dir()
    print(directory.split('_')[0])
    assert directory.split('/')[-1].split('_')[0]=='dim'
    dim=int(directory.split('/')[-1].split('_')[1])
    print(dim)
    #col.append('factor')
    df_dict = {}
    for path in dirpath.iterdir():
        print(directory.split('/')[-1])
        filename = str(path)
        factor=int(filename.split('/')[-1].split('_')[1])
        df=load_all_files(filename)
        print(df)
        df=df.melt('lambda', var_name='theta', value_name='value')
        print("Melted", df)
        df_dict[f"LassoNet {factor}"]=df
        print(df_dict)
    return df_dict


def plot_from_dim_dir(directory, save):
    df_dict=get_dfs_factors(directory)
    matplotlib.use("TkAgg")
    for lasso, df in df_dict.items():
        set_up_plt(NEURIPS_FONT_FAMILY)
        set_size(NEURIPS_TEXT_WIDTH)
        fig=sns.lineplot(x='lambda', y='value', hue='theta', data=df)
        #Additional formatting TODO
        fig.legend_.set_title(None)
        #plt.xlim(f['x_lim'])
        plt.ylim((0,2))
        #plt.setp(fig.lines, alpha=f['alpha'])
        #plt.legend(loc=??)
        plt.xlabel('$\\lambda$')
        plt.ylabel('$\\theta$')
        plt.title(lasso)
        if save:
            my_dir=f"results/"
            Path(my_dir).mkdir(parents=True, exist_ok=True)
            name=f"rename_thetas_rename"
            plt.savefig(f"{my_dir}/{name}.png")
        plt.show()




if __name__=="__main__":
    save=True
    plot_from_dim_dir("thetas/dim_2", save)

