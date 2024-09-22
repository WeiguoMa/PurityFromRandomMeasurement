import os
import pickle
import sys
import time
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.python.Physics.generate_TEST_DM import pseudo_random_DM
from src.python.RenyiEntropy import RenyiEntropy
from src.python.fake_sampler import FakeSampler, random_measurementScheme


def calculate_renyi2_IDEAL(dm):
    return -np.log2(np.trace(dm @ dm).real)


def generate_fakeMeasurements4DMs(DMs: List, K: int, random_measurementSchemes):
    return [
        [
            FAKESAMPLER.fake_sampling_dm(DM, measure_times=K, measurement_orientation=measurement_orientation)
            for measurement_orientation in random_measurementSchemes
        ]
        for DM in DMs
    ]


def calculateRenyi2(measurementResults4DMs: List, random_measurementSchemes) -> (List, List, List, List):
    time_cs, time_hamming = [], []  # FOR DM_LIST
    entropy_cs, entropy_hamming = [], []  # FOR DM_LIST
    for measurementDM in measurementResults4DMs:
        renyiCalculator = RenyiEntropy(random_measurementSchemes, measurementDM)

        start_time = time.time()
        entropy_cs.append(renyiCalculator.calculateRenyiEntropy(classical_shadow=True))
        time_cs.append(time.time() - start_time)

        start_time = time.time()
        entropy_hamming.append(renyiCalculator.calculateRenyiEntropy(classical_shadow=False))
        time_hamming.append(time.time() - start_time)

    return entropy_cs, entropy_hamming, time_cs, time_hamming


def calculate_entropy_with_errorBars(DMs4Gauging: List, M, K, repeats: int = 10) -> Dict:
    time_cs_results_repeats = []
    time_random_results_repeats = []
    renyiEntropy_CS_results_repeats = []
    renyiEntropy_Hamming_results_repeats = []
    STANDARD_PURITY = np.array([calculate_renyi2_IDEAL(dm) for dm in TEST_DM])

    for _ in tqdm(range(repeats)):
        random_measurementSchemes = random_measurementScheme(QNUMBER, amount=M)
        measurementResults4DMs = generate_fakeMeasurements4DMs(DMs4Gauging, K, random_measurementSchemes)

        renyiEntropy_CS, renyiEntropy_Hamming, time_cs, time_hamming \
            = calculateRenyi2(measurementResults4DMs, random_measurementSchemes)

        time_cs_results_repeats.append(time_cs)
        time_random_results_repeats.append(time_hamming)
        renyiEntropy_CS_results_repeats.append(renyiEntropy_CS)
        renyiEntropy_Hamming_results_repeats.append(renyiEntropy_Hamming)

    return {
        'avg_CS_DMs': np.abs(np.mean(renyiEntropy_CS_results_repeats, axis=0) - STANDARD_PURITY),
        'std_CS_DMs': np.std(renyiEntropy_CS_results_repeats, axis=0) / np.sqrt(repeats),
        'avg_hamming_DMs': np.abs(np.mean(renyiEntropy_Hamming_results_repeats, axis=0) - STANDARD_PURITY),
        'std_hamming_DMs': np.std(renyiEntropy_Hamming_results_repeats, axis=0) / np.sqrt(repeats),
        'avg_time_cs_DMs': np.mean(time_cs_results_repeats, axis=0),
        'std_time_cs_DMs': np.std(time_cs_results_repeats, axis=0) / np.sqrt(repeats),
        'avg_time_hamming_DMs': np.mean(time_random_results_repeats, axis=0),
        'std_time_hamming_DMs': np.std(time_random_results_repeats, axis=0) / np.sqrt(repeats),
    }


def extract_data(data: Dict) -> Dict:
    """
    Structure:
    KRepeat -> [DM0_Relating: List, DM1_Relating: List, DM2_Relating: List, DM3_Relating: List]
    For DM0:
        Convert2ndarray - > KRepeat[:, 0]
    """
    avg_CS_Ns_DMs, std_CS_Ns_DMs, avg_hamming_Ns_DMs, std_hamming_Ns_DMs = [], [], [], []
    avgTime_CS_Ns_DMs, stdTime_CS_Ns_DMs, avgTime_Hamming_Ns_DMs, stdTime_Hamming_Ns_DMs = [], [], [], []
    for item in data.values():
        avg_CS_Ns_DMs.append(item['avg_CS_DMs'])
        std_CS_Ns_DMs.append(item['std_CS_DMs'])
        avg_hamming_Ns_DMs.append(item['avg_hamming_DMs'])
        std_hamming_Ns_DMs.append(item['std_hamming_DMs'])
        avgTime_CS_Ns_DMs.append(item['avg_time_cs_DMs'])
        stdTime_CS_Ns_DMs.append(item['std_time_cs_DMs'])
        avgTime_Hamming_Ns_DMs.append(item['avg_time_hamming_DMs'])
        stdTime_Hamming_Ns_DMs.append(item['std_time_hamming_DMs'])

    dictDMs = {
        idx: {
            'avg_CS_Ns': np.array(avg_CS_Ns_DMs)[:, idx],
            'std_CS_Ns': np.array(std_CS_Ns_DMs)[:, idx],
            'avg_hamming_Ns': np.array(avg_hamming_Ns_DMs)[:, idx],
            'std_hamming_Ns': np.array(std_hamming_Ns_DMs)[:, idx],
            'avgTime_CS_Ns': np.array(avgTime_CS_Ns_DMs)[:, idx],
            'stdTime_CS_Ns': np.array(stdTime_CS_Ns_DMs)[:, idx],
            'avgTime_Hamming_Ns': np.array(avgTime_Hamming_Ns_DMs)[:, idx],
            'stdTime_Hamming_Ns': np.array(stdTime_Hamming_Ns_DMs)[:, idx],
        }
        for idx in range(len(avg_CS_Ns_DMs[0]))
    }

    return dictDMs


def plot_figures(K_values, data: Dict, saveLocation: Optional[str] = None):
    print('Plotting Figures...')
    for idxDM, value in enumerate(data.values()):
        if not isinstance(value, Dict):
            continue

        print(f"------------------- CURRENT DM: {idxDM} -------------------")
        plt.figure(figsize=(10, 6), dpi=300)

        plt.errorbar(K_values, value['avg_CS_Ns'], yerr=value['std_CS_Ns'], fmt='o-',
                     label='CS Avg.', capsize=5)
        plt.errorbar(K_values, value['avg_hamming_Ns'], yerr=value['std_hamming_Ns'],
                     fmt='s-', label='Hamming Avg.', capsize=5)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('N (Number of Qubits)', fontsize=20)
        plt.ylabel('Renyi Entropy', fontsize=20)
        plt.title('Error of Renyi Entropy vs N with Error Bars', fontsize=22)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"../figures/N_holdK_{K}_M_{M}_rp_{repeats}_RenyiEntropy_DM_{idxDM}.pdf")

        # Plot Time consumption with error bars
        plt.figure(figsize=(10, 6))
        plt.errorbar(K_values, value['avgTime_CS_Ns'], yerr=value['stdTime_CS_Ns'], fmt='o-',
                     label='Classical Shadow', capsize=5)
        plt.errorbar(K_values, value['avgTime_Hamming_Ns'], yerr=value['stdTime_Hamming_Ns'],
                     fmt='s-', label='Hamming', capsize=5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('N (Number of Qubits)', fontsize=20)
        plt.ylabel('Time (seconds)', fontsize=20)
        plt.title('Time Consumption vs N with Error Bars', fontsize=22)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"../figures/N_holdK_{K}_M_{M}_rp_{repeats}_TimeConsumption_DM_{idxDM}.pdf")


if __name__ == '__main__':
    K = 1000
    M = 1000
    repeats = 10
    QNUMBER_LIST = [3, 4, 5, 6]

    dataNs = {}
    for QNUMBER in QNUMBER_LIST:
        print('------------------- CURRENT QNUMBER: ', QNUMBER, ' -------------------')
        FAKESAMPLER = FakeSampler(QNUMBER)
        TEST_DM = pseudo_random_DM(QNUMBER, numPure=0, numMixed=1)

        dataNs[QNUMBER] = calculate_entropy_with_errorBars(DMs4Gauging=TEST_DM, M=M, K=K, repeats=repeats)

    extractedData: Dict = extract_data(dataNs)

    extractedData['QNUMBER_LIST'] = QNUMBER_LIST
    extractedData['M'] = M
    extractedData['K'] = K

    with open(f"../data/N_holdK_{K}_M_{M}_rp_{repeats}.mat", 'wb') as f:
        pickle.dump(extractedData, f)

    plot_figures(QNUMBER_LIST, extractedData)
