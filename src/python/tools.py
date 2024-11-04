import pickle
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt


def extract_data(data: Dict, save_location: Optional[str] = None) -> Dict:
    """
    Structure:
    KRepeat -> [DM0_Relating: List, DM1_Relating: List, DM2_Relating: List, DM3_Relating: List]
    For DM0:
        Convert2ndarray - > KRepeat[:, 0]
    """
    avg_CS_Ks_DMs, std_CS_Ks_DMs, avg_hamming_Ks_DMs, std_hamming_Ks_DMs = [], [], [], []
    avgTime_CS_Ks_DMs, stdTime_CS_Ks_DMs, avgTime_Hamming_Ks_DMs, stdTime_Hamming_Ks_DMs = [], [], [], []
    for item in data.values():
        avg_CS_Ks_DMs.append(item['avg_CS_DMs'])
        std_CS_Ks_DMs.append(item['std_CS_DMs'])
        avg_hamming_Ks_DMs.append(item['avg_hamming_DMs'])
        std_hamming_Ks_DMs.append(item['std_hamming_DMs'])
        avgTime_CS_Ks_DMs.append(item['avg_time_cs_DMs'])
        stdTime_CS_Ks_DMs.append(item['std_time_cs_DMs'])
        avgTime_Hamming_Ks_DMs.append(item['avg_time_hamming_DMs'])
        stdTime_Hamming_Ks_DMs.append(item['std_time_hamming_DMs'])

    dictDMs = {
        idx: {
            'avg_CS_Ks': np.array(avg_CS_Ks_DMs)[:, idx],
            'std_CS_Ks': np.array(std_CS_Ks_DMs)[:, idx],
            'avg_hamming_Ks': np.array(avg_hamming_Ks_DMs)[:, idx],
            'std_hamming_Ks': np.array(std_hamming_Ks_DMs)[:, idx],
            'avgTime_CS_Ks': np.array(avgTime_CS_Ks_DMs)[:, idx],
            'stdTime_CS_Ks': np.array(stdTime_CS_Ks_DMs)[:, idx],
            'avgTime_Hamming_Ks': np.array(avgTime_Hamming_Ks_DMs)[:, idx],
            'stdTime_Hamming_Ks': np.array(stdTime_Hamming_Ks_DMs)[:, idx],
        }
        for idx in range(len(avg_CS_Ks_DMs[0]))
    }

    if save_location is not None:
        with open(save_location, 'wb') as outfile:
            pickle.dump(dictDMs, outfile)

    return dictDMs


def combine_datasets(x_axis: List,
                     datasets: List[Dict],
                     dataset_location: Optional[str] = None):
    combine_data = {
        x_axis[i]: datasets[i]
        for i in range(len(x_axis))
    }
    extractedData = extract_data(combine_data, dataset_location)
    return extractedData


def plot_figures(x_values, data: Dict, **kwargs):
    print('Plotting Figures...')
    for idxDM, value in enumerate(data.values()):
        if not isinstance(value, Dict):
            continue

        print(f"------------------- CURRENT DM: {idxDM} -------------------")
        plt.figure(figsize=(10, 6), dpi=300)

        plt.errorbar(x_values, value['avg_CS_Ks'], yerr=value['std_CS_Ks'], fmt='o-',
                     label='CS Avg.', capsize=5)
        plt.errorbar(x_values, value['avg_hamming_Ks'], yerr=value['std_hamming_Ks'],
                     fmt='s-', label='Hamming Avg.', capsize=5)

        if kwargs.get('STANDARD_PURITY') is not None:
            plt.axhline(y=kwargs.get('STANDARD_PURITY')[idxDM], color='g', linestyle='--', label='Standard Purity')

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        if kwargs.get('entropy_ylim') is not None:
            plt.ylim(kwargs.get('entropy_ylim'))

        if kwargs.get('entropy_xlabel') is not None:
            plt.xlabel(kwargs.get('entropy_xlabel'), fontsize=20)

        if kwargs.get('entropy_ylabel') is not None:
            plt.ylabel(kwargs.get('entropy_ylabel'), fontsize=20)

        if kwargs.get('entropy_title') is not None:
            plt.title(kwargs.get('entropy_title'), fontsize=22)

        plt.legend(fontsize=17)
        plt.tight_layout()
        if kwargs.get('entropy_plot_location') is not None:
            plt.savefig(kwargs.get('entropy_plot_location'))
        else:
            plt.show()

        # Plot Time consumption with error bars
        plt.figure(figsize=(10, 6), dpi=300)
        plt.errorbar(x_values, value['avgTime_CS_Ks'], yerr=value['stdTime_CS_Ks'], fmt='o-',
                     label='Classical Shadow', capsize=5)
        plt.errorbar(x_values, value['avgTime_Hamming_Ks'], yerr=value['stdTime_Hamming_Ks'],
                     fmt='s-', label='Hamming', capsize=5)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        if kwargs.get('time_consumption_ylim') is not None:
            plt.ylim(kwargs.get('time_consumption_ylim'))

        if kwargs.get('time_consumption_xlabel') is not None:
            plt.xlabel(kwargs.get('time_consumption_xlabel'), fontsize=20)

        if kwargs.get('time_consumption_ylabel') is not None:
            plt.ylabel(kwargs.get('time_consumption_ylabel'), fontsize=20)

        if kwargs.get('time_consumption_title') is not None:
            plt.title(kwargs.get('time_consumption_title'), fontsize=22)

        plt.legend(fontsize=17)
        plt.tight_layout()

        if kwargs.get('time_consumption_location') is not None:
            plt.savefig(kwargs.get('time_consumption_location'))
        else:
            plt.show()


def analyze_results_EA(results_epoches, theta_values, fig1_loc: str, fig2_loc: str, **kwargs):
    epoches = results_epoches.shape[0]

    renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS = results_epoches[:, :, 0], results_epoches[:, :, 1], results_epoches[:, :, 2]
    renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS = results_epoches[:, :, 3], results_epoches[:, :, 4], results_epoches[:, :, 5]

    v1 = np.mean(renyi_rhoAQ_ideal - renyi_rhoA_ideal, axis=0)
    v6 = np.mean(renyi_rhoAQ_hamming - renyi_rhoA_hamming, axis=0)
    v7 = np.mean(renyi_rhoAQ_CS - renyi_rhoA_CS, axis=0)

    v2 = np.mean(np.abs(renyi_rhoA_CS - renyi_rhoA_ideal), axis=0)
    v3 = np.mean(np.abs(renyi_rhoA_hamming - renyi_rhoA_ideal), axis=0)
    v4 = np.mean(np.abs(renyi_rhoAQ_CS - renyi_rhoAQ_ideal), axis=0)
    v5 = np.mean(np.abs(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal), axis=0)

    v2_std = np.std(np.abs(renyi_rhoA_CS - renyi_rhoA_ideal), axis=0) / np.sqrt(epoches)
    v3_std = np.std(np.abs(renyi_rhoA_hamming - renyi_rhoA_ideal), axis=0) / np.sqrt(epoches)
    v4_std = np.std(np.abs(renyi_rhoAQ_CS - renyi_rhoAQ_ideal), axis=0) / np.sqrt(epoches)
    v5_std = np.std(np.abs(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal), axis=0) / np.sqrt(epoches)
    v6_std = np.std(np.abs(renyi_rhoAQ_hamming - renyi_rhoA_hamming), axis=0) / np.sqrt(epoches)
    v7_std = np.std(np.abs(renyi_rhoAQ_CS - renyi_rhoA_CS), axis=0) / np.sqrt(epoches)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.errorbar(theta_values, v1, yerr=0, label='Ideal', fmt='o', capsize=1, linestyle='-')
    plt.errorbar(theta_values, v6, yerr=v6_std, label='Hamming', fmt='^', capsize=1, linestyle='-')
    plt.errorbar(theta_values, v7, yerr=v7_std, label='Classical Shadow', fmt='*', capsize=1, linestyle='-')
    plt.xlabel(r"$\theta$", fontsize=20)
    plt.ylabel(r"$\Delta S$", fontsize=20)
    plt.xticks([0, np.pi / 2, np.pi], ['0', r'$\pi/2$', r'$\pi$'], fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"Entanglement Asymmetry with Methods for M={kwargs.get('M')}, K={kwargs.get('K')}, L={kwargs.get('L')}", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig1_loc)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.errorbar(theta_values, v2, yerr=v2_std, label='A - Classical Shadow', fmt='o', capsize=1, linestyle='-')
    plt.errorbar(theta_values, v3, yerr=v3_std, label='A - Hamming', fmt='^', capsize=1, linestyle='-')
    plt.errorbar(theta_values, v4, yerr=v4_std, label='AQ - Classical Shadow', fmt='*', capsize=1, linestyle='--')
    plt.errorbar(theta_values, v5, yerr=v5_std, label='AQ - Hamming', fmt='+', capsize=1, linestyle='--')
    plt.xlabel(r"$\theta$", fontsize=20)
    plt.ylabel(r"Difference of $S$", fontsize=20)
    plt.xticks([0, np.pi / 2, np.pi], ['0', r'$\pi/2$', r'$\pi$'], fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"Difference of R\'enyi Entropy with Methods for M={kwargs.get('M')}, K={kwargs.get('K')}, L={kwargs.get('L')}", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig2_loc)