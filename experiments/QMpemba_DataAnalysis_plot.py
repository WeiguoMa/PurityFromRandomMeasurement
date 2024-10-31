import pickle
from itertools import product
from typing import Dict, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter


def data_preparation_EA(results: np.ndarray, **kwargs) -> Dict:
    """
    Args:
        results: Shape - (epoches, thetas, times, 6);
    """
    _RESULT = {
        key: value for key, value in kwargs.items()
    }
    epoches = results.shape[0]
    renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS = results[:, :, :, 0], results[:, :, :, 1], results[:, :, :, 2]
    renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS = results[:, :, :, 3], results[:, :, :, 4], results[:, :, :,
                                                                                                       5]

    _RESULT['AQ_A_IDEAL'] = np.mean(renyi_rhoAQ_ideal - renyi_rhoA_ideal, axis=0)

    _RESULT['AQ_A_HAMMING'] = np.mean(renyi_rhoAQ_hamming - renyi_rhoA_hamming, axis=0)
    _RESULT['AQ_A_CS'] = np.mean(renyi_rhoAQ_CS - renyi_rhoA_CS, axis=0)

    _RESULT['AQ_A_HAMMING_ERRORBAR'] = np.std(np.abs(renyi_rhoAQ_hamming - renyi_rhoA_hamming), axis=0) / np.sqrt(
        epoches)
    _RESULT['AQ_A_CS_ERRORBAR'] = np.std(np.abs(renyi_rhoAQ_CS - renyi_rhoA_CS), axis=0) / np.sqrt(epoches)

    _RESULT['A_IDEAL'] = np.mean(renyi_rhoA_ideal, axis=0)
    _RESULT['AQ_IDEAL'] = np.mean(renyi_rhoAQ_ideal, axis=0)
    _RESULT['A_CS_IDEAL'] = np.mean(np.abs(renyi_rhoA_CS - renyi_rhoA_ideal), axis=0)
    _RESULT['A_HAMMING_IDEAL'] = np.mean(np.abs(renyi_rhoA_hamming - renyi_rhoA_ideal), axis=0)
    _RESULT['AQ_CS_IDEAL'] = np.mean(np.abs(renyi_rhoAQ_CS - renyi_rhoAQ_ideal), axis=0)
    _RESULT['AQ_HAMMING_IDEAL'] = np.mean(np.abs(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal), axis=0)

    _RESULT['A_CS_IDEAL_ERRORBAR'] = np.std(np.abs(renyi_rhoA_CS - renyi_rhoA_ideal), axis=0) / np.sqrt(epoches)
    _RESULT['A_HAMMING_IDEAL_ERRORBAR'] = np.std(np.abs(renyi_rhoA_hamming - renyi_rhoA_ideal), axis=0) / np.sqrt(
        epoches)
    _RESULT['AQ_CS_IDEAL_ERRORBAR'] = np.std(np.abs(renyi_rhoAQ_CS - renyi_rhoAQ_ideal), axis=0) / np.sqrt(epoches)
    _RESULT['AQ_HAMMING_IDEAL_ERRORBAR'] = np.std(np.abs(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal), axis=0) / np.sqrt(
        epoches)

    return _RESULT


def process_Errors(file_name):
    """
    Errors with ErrorBar. This function does not distinguish the errors in THETAS and TIMES.
    """
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    AQ_A_HAMMING_IDEAL = np.nanmean(np.abs(data['AQ_A_HAMMING'] - data['AQ_A_IDEAL']) / data['AQ_A_IDEAL'])
    AQ_A_CS_IDEAL = np.nanmean(np.abs(data['AQ_A_CS'] - data['AQ_A_IDEAL']) / data['AQ_A_IDEAL'])
    AQ_A_HAMMING_IDEAL_ERRORBAR = np.nanmean(data['AQ_A_HAMMING_ERRORBAR'])
    AQ_A_CS_IDEAL_ERRORBAR = np.nanmean(data['AQ_A_CS_ERRORBAR'])

    A_CS_IDEAL = np.nanmean(data['A_CS_IDEAL'] / data['A_IDEAL'])
    A_HAMMING_IDEAL = np.nanmean(data['A_HAMMING_IDEAL'] / data['A_IDEAL'])
    AQ_CS_IDEAL = np.nanmean(data['AQ_CS_IDEAL'] / data['AQ_IDEAL'])
    AQ_HAMMING_IDEAL = np.nanmean(data['AQ_HAMMING_IDEAL'] / data['AQ_IDEAL'])

    A_CS_ERROR = np.nanmean(data['A_CS_IDEAL_ERRORBAR'])
    A_HAMMING_ERROR = np.nanmean(data['A_HAMMING_IDEAL_ERRORBAR'])
    AQ_CS_ERROR = np.nanmean(data['AQ_CS_IDEAL_ERRORBAR'])
    AQ_HAMMING_ERROR = np.nanmean(data['AQ_HAMMING_IDEAL_ERRORBAR'])

    return [
        A_CS_IDEAL, A_HAMMING_IDEAL,
        A_CS_ERROR, A_HAMMING_ERROR,
        AQ_CS_IDEAL, AQ_HAMMING_IDEAL,
        AQ_CS_ERROR, AQ_HAMMING_ERROR,
        AQ_A_CS_IDEAL, AQ_A_HAMMING_IDEAL,
        AQ_A_CS_IDEAL_ERRORBAR, AQ_A_HAMMING_IDEAL_ERRORBAR
    ]


def data_preparation_M_K_error(MLIST, KLIST, subA: List):
    _RESULT = np.array([
        [
            process_Errors(f'QMpemba/QMpemba_M{M}_K{K}_N10_A{subA}_EP50.pkl')
            for K in KLIST
        ]
        for M in MLIST
    ])

    return _RESULT


def prepare_data4MKs(MKCombination: Dict, subA):
    _dataMK = {
        key: np.mean(np.array(
            [
                process_Errors(file_name=f'../data/QMpemba/QMpemba_M{para[0]}_K{para[1]}_N10_A{subA}_EP50.pkl')
                for para in value
            ]
        ), axis=0)
        for key, value in MKCombination.items()
    }

    return np.array([
        [_data[idx] for _data in _dataMK.values()]
        for idx in range(12)
    ])


def MKsData_subA(MLIST, KLIST, subAList):
    _combinations_dict = {}
    for _m, _k in product(MLIST, KLIST):
        _product_value = _m * _k
        if _product_value not in _combinations_dict:
            _combinations_dict[_product_value] = []
        _combinations_dict[_product_value].append((_m, _k))
    MKDict = dict(sorted(_combinations_dict.items()))
    return MKDict, np.array([
        prepare_data4MKs(MKDict, subA) for subA in subAList
    ])


def plot_Errors_Hamming_CS(MLIST, KLIST, Z, Zerror, save: bool = False, **kwargs):
    if kwargs.get('info') == 'A':
        title = r'Errors of $S(\rho_{A})$'
        _norm = True
    elif kwargs.get('info') == 'AQ':
        title = r'Errors of $S(\rho_{AQ})$'
        _norm = True
    elif kwargs.get('info') == 'AQ_A':
        title = r'Errors of $S(\rho_{AQ}) - S(\rho_{A})$'
        _norm = False
    else:
        raise ValueError(f'Unknown info: {kwargs.get("info")}')

    x_indices, y_indices = np.arange(len(KLIST)), np.arange(len(MLIST))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    plt.rcParams['font.family'] = 'Arial'

    cmap = 'coolwarm'
    if not _norm:
        norm = Normalize(vmin=min(Z[0].min(), Z[1].min()), vmax=max(Z[0].max(), Z[1].max()))
    else:
        norm = Normalize(vmin=0., vmax=0.7)
    # norm = Normalize(vmin=min(Z[0].min(), Z[1].min()), vmax=max(Z[0].max(), Z[1].max()))
    for i in range(len(MLIST)):
        for j in range(len(KLIST)):
            top_triangle = Polygon(np.array([(x_indices[j] - 0.5, y_indices[i] - 0.5),
                                             (x_indices[j] + 0.5, y_indices[i] - 0.5),
                                             (x_indices[j], y_indices[i])]),
                                   color=plt.get_cmap(cmap)(norm(Z[0][i, j])), ec='black')
            ax.add_patch(top_triangle)

            bottom_triangle = Polygon(np.array([(x_indices[j] - 0.5, y_indices[i] + 0.5),
                                                (x_indices[j] + 0.5, y_indices[i] + 0.5),
                                                (x_indices[j], y_indices[i])]),
                                      color=plt.get_cmap(cmap)(norm(Z[1][i, j])), ec='black')
            ax.add_patch(bottom_triangle)

            ax.errorbar(x_indices[j], y_indices[i] - 0.25, yerr=Zerror[0][i, j], fmt='.', color='black', capsize=6)
            ax.errorbar(x_indices[j], y_indices[i] + 0.25, yerr=Zerror[1][i, j], fmt='.', color='#FF7F0E', capsize=6)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.037, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r'Error ($\%$)', fontsize=16)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.0f}"))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(x_indices)
    ax.set_yticks(y_indices)
    ax.set_xticklabels(KLIST, fontsize=16)
    ax.set_yticklabels(MLIST, fontsize=16)

    ax.set_xlabel(r'$K$', fontsize=20)
    ax.set_ylabel(r'$M$', fontsize=20)

    hamming_error_legend = mlines.Line2D([], [],
                                         color='#FF7F0E', marker='.',
                                         markersize=10, linestyle='None', label='Hamming')

    cs_error_legend = mlines.Line2D([], [],
                                    color='black', marker='.',
                                    markersize=10, linestyle='None', label='Classical Shadow')

    ax.legend(handles=[hamming_error_legend, cs_error_legend],
              loc='upper center', bbox_to_anchor=(0.55, -0.065),
              ncol=2, frameon=False, handletextpad=0.5, columnspacing=10)

    for spine in ax.spines.values():
        spine.set_color('black')

    plt.grid(False)
    plt.title(title, fontsize=20)
    plt.tight_layout()

    if save:
        save_location = f"../figures/QMpembaErrors_{kwargs.get('info')}_N{kwargs.get('N')}_A{kwargs.get('subA')}.pdf"
        plt.savefig(save_location, bbox_inches='tight')


def plot_EA_Hamming_CS_THETAS(result, save: bool = False):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams['font.family'] = 'Arial'

    theta_list, time_list = result.get('THETA_LIST'), [t * 1000 for t in result.get('TIME_LIST')]
    aq_a_ideal, aq_a_hamming, aq_a_cs = result.get('AQ_A_IDEAL'), result.get('AQ_A_HAMMING'), result.get('AQ_A_CS')
    aq_a_hamming_error, aq_a_cs_error = result.get('AQ_A_HAMMING_ERRORBAR'), result.get('AQ_A_CS_ERRORBAR')

    for i, theta in enumerate(theta_list):
        plt.errorbar(time_list, aq_a_ideal[i], label=r'$\theta=$' + result["THETA_LABELS"][i], fmt='o', capsize=1,
                     linestyle='-')

        plt.errorbar(time_list, aq_a_hamming[i], yerr=aq_a_hamming_error[i], fmt='^', capsize=1, linestyle='--',
                     alpha=0.5)
        plt.errorbar(time_list, aq_a_cs[i], yerr=aq_a_cs_error[i], fmt='*', capsize=1, linestyle='-.', alpha=0.5)

    _custom_lines = [
        Line2D([0], [0], linestyle='--', marker='^', label='Hamming'),
        Line2D([0], [0], linestyle='-.', marker='*', label='Classical Shadow')
    ]

    plt.xlabel("Time ($ns$)", fontsize=22)
    plt.ylabel("$\\Delta S$", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(
        f"Quench EA for N{result.get('QNUMBER_TOTAL')}-A{result.get('SUBSYSTEM_A')}, M={result['M']}, K={result['K']}",
        fontsize=24)
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + _custom_lines, fontsize=20)
    plt.grid(False)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
    plt.tight_layout()

    if save:
        save_location = f"../figures/QMpemba_N{result.get('QNUMBER_TOTAL')}_A{result.get('SUBSYSTEM_A')}_M{result['M']}_N{result['K']}.pdf"
        plt.savefig(save_location, bbox_inches='tight')


def plot_error_3AQ_A(MLIST, KLIST, subA, save: bool = False):
    _data = data_preparation_M_K_error(MLIST, KLIST, subA)
    error_combinations = [
        ((0, 1), (2, 3), 'A'),
        ((4, 5), (6, 7), 'AQ'),
        ((-4, -3), (-2, -1), 'AQ_A')
    ]

    for (i, j, info) in error_combinations:
        z_data = (_data[:, :, i[0]], _data[:, :, i[1]])
        zError_data = (_data[:, :, j[0]], _data[:, :, j[1]])

        plot_Errors_Hamming_CS(MLIST, KLIST, z_data, zError_data, N=10, subA=subA, save=save, info=info)

    return _data


def plot_error_AQ_A_inDetail_combined(MLIST, KLIST, data, save: bool = False, **kwargs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=300)
    plt.rcParams['font.family'] = 'Arial'

    ax1.errorbar(MLIST, data[:, 1, -3] * 100, yerr=data[:, 1, -1] * 100, fmt='-o', capsize=6, linewidth=2,
                 label=r'Hamming $K=100$')
    ax1.errorbar(MLIST, data[:, 1, -4] * 100, yerr=data[:, 1, -2] * 100, fmt='-^', capsize=6, linewidth=2,
                 label=r'Classical Shadow $K=100$')
    ax1.errorbar(MLIST, data[:, 4, -3] * 100, yerr=data[:, 4, -1] * 100, fmt='--o', capsize=6, linewidth=2,
                 label=r'Hamming $K=1000$')
    ax1.errorbar(MLIST, data[:, 4, -4] * 100, yerr=data[:, 4, -2] * 100, fmt='--^', capsize=6, linewidth=2,
                 label=r'Classical Shadow $K=1000$')
    ax1.set_xlabel(r'$M$', fontsize=20)
    ax1.set_xticks(MLIST)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_ylabel(r'Error ($\%$)', fontsize=20)
    ax1.legend(fontsize=16)
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_color('black')
    ax1.set_title(r'Errors of $S(\rho_AQ) - S(\rho_A)$', fontsize=20)

    ax2.errorbar(KLIST, data[2, :, -3] * 100, yerr=data[2, :, -1] * 100, fmt='-o', capsize=6, label=r'Hamming $M=100$')
    ax2.errorbar(KLIST, data[2, :, -4] * 100, yerr=data[2, :, -2] * 100, fmt='-^', capsize=6,
                 label=r'Classical Shadow $M=100$')
    ax2.errorbar(KLIST, data[-1, :, -3] * 100, yerr=data[-1, :, -1] * 100, fmt='--o', capsize=6,
                 label=r'Hamming $M=1000$')
    ax2.errorbar(KLIST, data[-1, :, -4] * 100, yerr=data[-1, :, -2] * 100, fmt='--^', capsize=6,
                 label=r'Classical Shadow $M=1000$')
    ax2.set_xlabel(r'$K$', fontsize=20)
    ax2.set_xticks(KLIST)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_ylabel(r'Error ($\%$)', fontsize=20)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_color('black')

    plt.tight_layout()

    if save:
        plt.savefig(
            f"../figures/error_inDetail_combined_N{kwargs.get('N')}_A{kwargs.get('subA')}.pdf",
            bbox_inches='tight'
        )


def plot_MK_N(MK_values, MKs_data, N_values):
    plt.rcParams['font.family'] = 'Arial'
    if MKs_data.shape[1] < 12:
        raise ValueError("MKs_data must have at least 12 variables in the second dimension.")

    # Define indices for data extraction and configuration
    data_indices = [(0, 2), (4, 6), (8, 10)]
    method_names = ['Classical Shadow', 'Hamming']
    colors = ['blue', 'green', 'red']  # Colors for each N_values
    linestyles = ['-', '--']
    markers = {'Classical Shadow': '^', 'Hamming': 'o'}
    subplot_titles = [r'$S(\rho_A)$', r'$S(\rho_{A,Q})$', r'$S(\rho_{A,Q}) - S(\rho_{A})$']

    # Initialize figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

    for ax_idx, (cs_idx, err_idx) in enumerate(data_indices):
        for method_idx, method in enumerate(method_names):
            linestyle = linestyles[method_idx]
            for color_idx, N in enumerate(N_values):
                data = MKs_data[:, cs_idx + method_idx, :][color_idx, 1:] * 100  # Convert to percentage
                error = MKs_data[:, err_idx + method_idx, :][color_idx, 1:]

                axs[ax_idx].errorbar(
                    MK_values[1:], data, yerr=error,
                    label=f'{method}, l={N}', color=colors[color_idx],
                    linestyle=linestyle, marker=markers[method], markersize=7, linewidth=2
                )

        # Labeling each subplot
        axs[ax_idx].text(0.55, 0.65, subplot_titles[ax_idx], transform=axs[ax_idx].transAxes, fontweight='bold',
                         fontsize=20)
        axs[ax_idx].set_ylabel(r'Error ($\%$)', fontsize=20)
        axs[ax_idx].tick_params(axis='both', labelsize=18)

    # Set x-axis to logarithmic scale
    axs[-1].set_xscale('log')
    axs[-1].set_xlabel(r'$M \times K$', fontsize=20)

    # Unified legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=15, ncol=1, bbox_to_anchor=(0.98, 0.96))

    # Super title and layout adjustments
    fig.suptitle(r"Performance Analysis for Different $l$ and $MK$ Values", fontsize=24, y=0.99, x=0.55)
    plt.tight_layout()
    plt.savefig('../figures/MKsData_subA_3.pdf', bbox_inches='tight')


def call_plot_MK_N(MLIST, KLIST, subAList: List):
    MK_Dict, MKs_data = MKsData_subA(MLIST, KLIST, subAList)
    NValues = [len(subA) for subA in subAList]

    plot_MK_N(list(MK_Dict.keys()), MKs_data, NValues)


if __name__ == '__main__':
    MLIST = [10, 50, 100, 200, 500, 1000]
    KLIST = [50, 100, 200, 500, 1000]
    subA = [1, 2, 3, 4]

    data = plot_error_3AQ_A(MLIST, KLIST, subA=subA, save=True)

    plot_error_AQ_A_inDetail_combined(MLIST, KLIST, data, save=True, N=10, subA=subA)
