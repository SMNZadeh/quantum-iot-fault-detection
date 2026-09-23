Grover-Based Fault Detection on Real IoT Data
=============================================
This script implements a hybrid quantum-classical framework for fault detection
in IoT networks using Grover's algorithm. It loads real IoT sensor data,
defines fault criteria based on measurable metrics, constructs a Grover circuit
with a real-fault oracle, and evaluates performance under ideal and noisy
conditions. It also generates figures for the paper.

Requirements:
pip install qiskit qiskit-aer pandas numpy matplotlib
import pandas as pd
import numpy as np
import math
import random
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ============================================================
# CONFIGURATION
# ============================================================

SHOTS = 1024                    # Number of measurement shots
MAX_QUBITS = 22                 # Maximum qubits for statevector simulation
NUM_ERRORS = 3                  # Number of faulty nodes to inject
RANDOM_SEED = 42                # Seed for reproducibility
CSV_PATH = 'iot_sensor_data.csv'  # Path to real IoT dataset

# Output figure filenames
FIG1 = 'figure1_complexity.png'
FIG2 = 'figure2_faults.png'
FIG3 = 'figure3_noise.png'


# ============================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================

def load_real_iot_data(csv_path, num_items=128, seed=RANDOM_SEED):
    """Load and preprocess a real IoT sensor dataset.

    Supports two dataset layouts:
      1) "Wide" format: one column per sensor feature
         (e.g. temperature_celsius, humidity_percent, ...).
      2) "Long" format: one row per individual sensor reading, with a
         Sensor_Type/Value pair (and optionally a ground-truth
         Oracle_Target/Anomaly column).

    Returns:
        normalized      : normalized feature dataframe
        feature_cols    : list of feature column names used
        oracle_labels   : boolean array of ground-truth fault labels if the
                           dataset provides them, otherwise None
    """
    df = pd.read_csv(csv_path)

    # --- Layout 1: wide format --------------------------------------
    feature_cols = []
    for col in ['temperature_celsius', 'humidity_percent', 'noise_level_db',
                'temperature', 'humidity', 'voltage', 'RSSI', 'SNR']:
        if col in df.columns:
            feature_cols.append(col)

    if len(feature_cols) >= 2:
        df_features = df[feature_cols].dropna()

        if len(df_features) > num_items:
            df_subset = df_features.sample(n=num_items, random_state=seed)
        else:
            df_subset = df_features

        normalized = (df_subset - df_subset.min()) / (df_subset.max() - df_subset.min())
        normalized = normalized.reset_index(drop=True)
        return normalized, feature_cols, None

    # --- Layout 2: long format (Sensor_Type / Value [/ Oracle_Target]) --
    if {'Sensor_Type', 'Value'}.issubset(df.columns):
        df = df.dropna(subset=['Sensor_ID', 'Value']).reset_index(drop=True)

        if len(df) > num_items:
            df_subset = df.sample(n=num_items, random_state=seed).reset_index(drop=True)
        else:
            df_subset = df.reset_index(drop=True)

        value_col = df_subset[['Value']].rename(columns={'Value': 'sensor_value'})
        normalized = (value_col - value_col.min()) / (value_col.max() - value_col.min())

        oracle_labels = None
        if 'Oracle_Target' in df_subset.columns:
            oracle_labels = df_subset['Oracle_Target'].fillna(0).astype(bool).values

        return normalized, ['sensor_value'], oracle_labels

    raise ValueError(
        "Could not find at least 2 recognized sensor feature columns, and no "
        "Sensor_Type/Value long-format columns were found either."
    )


# ============================================================
# 2. FAULT CRITERION DEFINITION
# ============================================================

def define_fault_criterion(df_normalized, feature_cols, oracle_labels=None):
    """Define fault criteria.

    If the dataset already provides ground-truth fault labels
    (oracle_labels, from an Oracle_Target/Anomaly column), use them
    directly. Otherwise, fall back to the IQR method on a temperature
    column (or a percentile threshold if no temperature column exists).
    """
    if oracle_labels is not None:
        thresholds = {'method': 'oracle_ground_truth'}
        return oracle_labels, thresholds

    temp_col = None
    for col in feature_cols:
        if 'temp' in col.lower():
            temp_col = col
            break

    if temp_col:
        q1 = df_normalized[temp_col].quantile(0.25)
        q3 = df_normalized[temp_col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr

        fault_labels = (
            (df_normalized[temp_col] > upper_bound) |
            (df_normalized[temp_col] < lower_bound)
        )

        thresholds = {
            'temperature_upper': upper_bound,
            'temperature_lower': lower_bound,
            'method': 'IQR'
        }
    else:
        col = feature_cols[0]
        threshold = df_normalized[col].quantile(0.9)
        fault_labels = df_normalized[col] > threshold
        thresholds = {f'{col}_threshold': threshold, 'method': 'percentile'}

    return fault_labels.values, thresholds


# ============================================================
# 3. ORACLE CREATION
# ============================================================

def create_real_fault_oracle(fault_labels, num_qubits):
    """Create a quantum oracle based on real fault labels."""
    qc = QuantumCircuit(num_qubits, name="RealFaultOracle")

    for idx, is_faulty in enumerate(fault_labels):
        if is_faulty and idx < 2**num_qubits:
            binary = format(idx, f'0{num_qubits}b')

            for qubit, bit in enumerate(binary):
                if bit == '0':
                    qc.x(qubit)

            if num_qubits > 1:
                qc.h(num_qubits - 1)
                qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
                qc.h(num_qubits - 1)
            else:
                qc.z(0)

            for qubit, bit in enumerate(binary):
                if bit == '0':
                    qc.x(qubit)

    return qc


# ============================================================
# 4. GROVER CIRCUIT CONSTRUCTION
# ============================================================

def build_grover_circuit_with_real_oracle(fault_labels, num_qubits):
    """Build the complete Grover circuit using the real fault oracle."""
    num_marked = sum(fault_labels)
    total_states = 2 ** num_qubits

    if num_marked > 0:
        theta = math.asin(math.sqrt(num_marked / total_states))
        iterations = max(1, int(round(math.pi / (4 * theta) - 0.5)))
    else:
        iterations = 0

    oracle = create_real_fault_oracle(fault_labels, num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits, name="GroverRealFault")
    qc.h(range(num_qubits))

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)

        qc.h(range(num_qubits))
        qc.x(range(num_qubits))

        if num_qubits > 1:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        else:
            qc.z(0)

        qc.x(range(num_qubits))
        qc.h(range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    return qc, iterations, num_marked


# ============================================================
# 5. NOISE SIMULATION
# ============================================================

def run_noisy_simulation(circuit, shots=SHOTS, noise_level=0.01):
    """Run Grover circuit on a noisy Aer simulator."""
    noise_model = NoiseModel()

    error_1q = depolarizing_error(noise_level, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'sx', 'x'])

    error_2q = depolarizing_error(noise_level * 5, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    simulator = AerSimulator(
        method='density_matrix',
        noise_model=noise_model
    )

    transpiled = transpile(circuit, simulator, optimization_level=1)

    job = simulator.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts(transpiled)

    return dict(counts)


# ============================================================
# 6. RESULTS ANALYSIS
# ============================================================

def analyze_results(counts, fault_labels, num_qubits):
    """Analyze measurement results against ground truth fault labels."""
    total_shots = sum(counts.values())
    marked_indices = [i for i, f in enumerate(fault_labels) if f]
    marked_set = set(marked_indices)

    marked_hits = 0
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx in marked_set:
            marked_hits += count

    success_probability = marked_hits / total_shots if total_shots > 0 else 0

    best_bitstring = max(counts, key=counts.get)
    best_index = int(best_bitstring, 2)

    return {
        'success_probability': success_probability,
        'best_index': best_index,
        'best_is_faulty': best_index in marked_set,
        'num_marked': len(marked_indices),
        'marked_indices': marked_indices
    }


# ============================================================
# 7. PLOTTING FUNCTIONS
# ============================================================

def plot_complexity_comparison(N, M, iterations, filename=FIG1):
    """Figure 1: Query complexity comparison (classical vs Grover)."""
    classical_queries = N
    grover_queries = iterations

    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ['Classical\nSearch', 'Grover\nAlgorithm']
    queries = [classical_queries, grover_queries]
    colors = ['#c0392b', '#2980b9']

    bars = ax.bar(methods, queries, color=colors, alpha=0.85, width=0.5)

    for bar, value in zip(bars, queries):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(queries) * 0.02,
                f'{value}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Queries', fontsize=11)
    ax.set_title(f'Query Complexity: Classical O(N) vs Grover O(√N)\n(N = {N}, M = {M})',
                 fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 1 saved: {filename}")


def plot_fault_distribution(fault_labels, filename=FIG2):
    """Figure 2: Distribution of faulty nodes across the IoT network."""
    N = len(fault_labels)

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#e74c3c' if f else '#27ae60' for f in fault_labels]

    ax.bar(range(N), [1] * N, color=colors, width=1.0)

    faulty_indices = [i for i, f in enumerate(fault_labels) if f]
    for idx in faulty_indices:
        ax.annotate('Fault', xy=(idx, 1), xytext=(idx, 1.3),
                    ha='center', fontsize=8, color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

    ax.set_xlabel('Node Index', fontsize=11)
    ax.set_ylabel('Status', fontsize=11)
    ax.set_title(f'Faulty Node Distribution Across IoT Network '
                 f'(N = {N}, Faults = {len(faulty_indices)})',
                 fontsize=11)
    ax.set_yticks([])
    ax.set_xlim(-1, N)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 2 saved: {filename}")


def plot_noise_impact(noise_levels, success_probs, filename=FIG3):
    """Figure 3: Effect of depolarizing noise on success probability."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(noise_levels, success_probs, marker='o',
            color='#8e44ad', linewidth=2, markersize=8)

    for x, y in zip(noise_levels, success_probs):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(6, 6),
                    textcoords='offset points', fontsize=9)

    ax.set_xlabel('Noise Level (%)', fontsize=11)
    ax.set_ylabel('Success Probability (%)', fontsize=11)
    ax.set_title('Impact of Depolarizing Noise on Grover Success Probability',
                 fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 3 saved: {filename}")


# ============================================================
# 8. MAIN WORKFLOW
# ============================================================

def main():
    print("=" * 60)
    print("Grover-Based Fault Detection on Real IoT Data")
    print("=" * 60)

    # Step 1: Load real data (no synthetic fallback)
    df_normalized, feature_cols, oracle_labels = load_real_iot_data(CSV_PATH, num_items=128)
    print(f"\n[OK] Data loaded: {len(df_normalized)} records")
    print(f"     Features: {feature_cols}")
    if oracle_labels is not None:
        print(f"     Ground-truth Oracle_Target labels found: {oracle_labels.sum()} faults")

    # Step 2: Define fault criterion
    fault_labels, thresholds = define_fault_criterion(df_normalized, feature_cols, oracle_labels)
    num_faults = sum(fault_labels)
    print(f"\n[OK] Fault criterion: {thresholds['method']}")
    print(f"     Detected {num_faults} faulty nodes")

    # Step 3: Calculate required qubits
    N = len(df_normalized)
    num_qubits = math.ceil(math.log2(N))
    print(f"\n[OK] Qubits required: {num_qubits} (N = {N})")

    # Step 4: Build Grover circuit
    circuit, iterations, num_marked = build_grover_circuit_with_real_oracle(
        fault_labels, num_qubits
    )
    print(f"\n[OK] Grover iterations: {iterations}")
    print(f"     Circuit depth: {circuit.depth()}")
    print(f"     Gate count: {circuit.size()}")

    # Step 5: Ideal simulation
    print("\n" + "-" * 60)
    print("Ideal Simulation (No Noise)")
    print("-" * 60)

    ideal_simulator = AerSimulator(method='statevector')
    transpiled_ideal = transpile(circuit, ideal_simulator)
    job_ideal = ideal_simulator.run(transpiled_ideal, shots=SHOTS)
    counts_ideal = job_ideal.result().get_counts()

    result_ideal = analyze_results(counts_ideal, fault_labels, num_qubits)
    print(f"  Success probability: {result_ideal['success_probability'] * 100:.2f}%")
    print(f"  Most probable index: {result_ideal['best_index']}")
    print(f"  Is faulty: {result_ideal['best_is_faulty']}")

    # Step 6: Noisy simulation
    print("\n" + "-" * 60)
    print("Noisy Simulation (Depolarizing Noise)")
    print("-" * 60)

    noise_levels_pct = [0.1, 0.5, 1.0]
    success_probs = [result_ideal['success_probability'] * 100]

    for nl in noise_levels_pct:
        counts_noisy = run_noisy_simulation(circuit, shots=SHOTS, noise_level=nl / 100)
        result_noisy = analyze_results(counts_noisy, fault_labels, num_qubits)
        success_probs.append(result_noisy['success_probability'] * 100)
        print(f"  Noise {nl:.1f}%: Success = {result_noisy['success_probability'] * 100:.2f}%")

    # Step 7: Classical comparison
    print("\n" + "-" * 60)
    print("Classical Search Comparison")
    print("-" * 60)

    classical_queries = N
    grover_queries = iterations
    speedup = classical_queries / grover_queries if grover_queries > 0 else 0

    print(f"  Classical queries: {classical_queries}")
    print(f"  Grover queries: {grover_queries}")
    print(f"  Theoretical speedup: {speedup:.2f}x")

    # Step 8: Generate figures
    print("\n" + "-" * 60)
    print("Generating Figures")
    print("-" * 60)

    plot_complexity_comparison(N, num_marked, iterations, FIG1)
    plot_fault_distribution(fault_labels, FIG2)
    plot_noise_impact([0.0] + noise_levels_pct, success_probs, FIG3)

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
