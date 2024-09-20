import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
import pickle

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

    print(f'-------- CURRENT K = {K} --------')
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
        'avg_CS_DMs': np.mean(renyiEntropy_CS_results_repeats, axis=0),
        'std_CS_DMs': np.std(renyiEntropy_CS_results_repeats, axis=0) / np.sqrt(repeats),
        'avg_hamming_DMs': np.mean(renyiEntropy_Hamming_results_repeats, axis=0),
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

    return dictDMs


def plot_figures(K_values, data: Dict, saveLocation: Optional[str] = None):
    print('Plotting Figures...')
    for idxDM, value in enumerate(data.values()):
        if not isinstance(value, Dict):
            continue

        print(f"------------------- CURRENT DM: {idxDM} -------------------")
        plt.figure(figsize=(10, 6), dpi=300)

        plt.errorbar(K_values, value['avg_CS_Ks'], yerr=value['std_CS_Ks'], fmt='o-',
                     label='CS Avg.', capsize=5)
        plt.errorbar(K_values, value['avg_hamming_Ks'], yerr=value['std_hamming_Ks'],
                     fmt='s-', label='Hamming Avg.', capsize=5)

        plt.axhline(y=STANDARD_PURITY[idxDM], color='g', linestyle='--', label='Standard Purity')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('K (Number of Samples)', fontsize=20)
        plt.ylabel('Renyi Entropy', fontsize=20)
        plt.title('Renyi Entropy vs K with Error Bars', fontsize=22)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"./figures/K_holdM_{M}_qn_{QNUMBER}_rp_{repeats}_RenyiEntropy_DM_{idxDM}.pdf")

        # Plot Time consumption with error bars
        plt.figure(figsize=(10, 6))
        plt.errorbar(K_values, value['avgTime_CS_Ks'], yerr=value['stdTime_CS_Ks'], fmt='o-',
                     label='Classical Shadow', capsize=5)
        plt.errorbar(K_values, value['avgTime_Hamming_Ks'], yerr=value['stdTime_Hamming_Ks'],
                     fmt='s-', label='Hamming', capsize=5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('K (Number of Samples)', fontsize=20)
        plt.ylabel('Time (seconds)', fontsize=20)
        plt.title('Time Consumption vs K with Error Bars', fontsize=22)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"./figures/K_holdM_{M}_qn_{QNUMBER}_rp_{repeats}_TimeConsumption_DM_{idxDM}.pdf")


if __name__ == '__main__':
    QNUMBER = 4
    FAKESAMPLER = FakeSampler(QNUMBER)

    # Fixed TEST_DM
    TEST_DM = pseudo_random_DM(QNUMBER, numPure=0, numMixed=1)
    STANDARD_PURITY = [calculate_renyi2_IDEAL(dm) for dm in TEST_DM]

    M = 1000
    repeats = 50
    K_values = [50, 100, 300, 500, 1000, 2000, 4000]

    dataKs = {K: calculate_entropy_with_errorBars(DMs4Gauging=TEST_DM, M=M, K=K, repeats=repeats) for K in K_values}

    extractedData: Dict = extract_data(dataKs)

    extractedData['QNUMBER'] = QNUMBER
    extractedData['M'] = M
    extractedData['K_values'] = K_values
    extractedData['STANDARD_PURITY'] = STANDARD_PURITY

    with open(f"./data/K_holdM_{M}_qn_{QNUMBER}_rp_{repeats}.mat", 'wb') as f:
        pickle.dump(extractedData, f)

    plot_figures(K_values, extractedData)
