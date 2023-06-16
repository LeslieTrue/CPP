import matplotlib.pyplot as plt
import seaborn as sns

def update_pi_from_z(net):
    import copy
    model_dict = net.state_dict()
    save_dict = copy.deepcopy(model_dict)
    to_rename_keys = []
    for key in save_dict:
        if 'subspace' in key:
            to_rename_keys.append(key)
    for key in to_rename_keys:
        print(f'renamed key {key}')
        pre, post = key.split('subspace')
        save_dict[pre + 'cluster' + post] = save_dict.pop(key)

    model_dict.update(save_dict)
    log = net.load_state_dict(model_dict)
    print(log)
    return net

def plot_codinglength(y, x, filename):
    plt.plot(x, y)
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=4)
    ax.set_xlabel('Number of Clusters', fontdict={'fontsize': 15, 'fontweight': 'bold', 'fontfamily': 'serif'})
    ax.set_ylabel('Coding Bits', fontdict={'fontsize': 15, 'fontweight': 'bold', 'fontfamily': 'serif'})
    # ax.set_title('Measure Optimal Number of Clusters', fontdict={'fontsize': 20, 'fontweight': 'bold', 'fontfamily': 'serif'})

    # Show grid
    ax.grid(True)

    # Display the plot
    plt.savefig(filename, format='pdf',  dpi=600)
    