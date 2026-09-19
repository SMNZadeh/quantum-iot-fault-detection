"""
Grover-Based Fault Detection on Real IoT Data
=============================================
This script implements a hybrid quantum-classical framework for fault detection
in IoT networks using Grover's algorithm. It loads real IoT sensor data,
defines fault criteria based on measurable metrics, constructs a Grover circuit
with a real-fault oracle, and evaluates performance under ideal and noisy
conditions.

Requirements:
    pip install qiskit qiskit-aer pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import math
import random
import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

SHOTS = 1024                    # Number of measurement shots
MAX_QUBITS = 22                 # Maximum qubits for statevector simulation
NUM_ERRORS = 3                  # Number of faulty nodes to inject
RANDOM_SEED = 42                # Seed for reproducibility
CSV_PATH = 'iot_sensor_data.csv'  # Path to real IoT dataset


# ============================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================

def load_real_iot_data(csv_path, num_items=128, seed=RANDOM_SEED):
    """
    Load and preprocess a real IoT sensor dataset.

    Recommended datasets:
        - Intel Lab Data: http://db.csail.mit.edu/labdata/labdata.html
        - Multisensor WSN: https://zenodo.org/records/18183640

    Args:
        csv_path: Path to the CSV file
        num_items: Number of records to select (128 for quantum feasibility)
        seed: Random seed for reproducibility

    Returns:
        normalized_df: Normalized feature DataFrame
        feature_cols: List of feature column names
    """
    df = pd.read_csv(csv_path)

    # Identify relevant feature columns
    feature_cols = []
    for col in ['temperature_celsius', 'humidity_percent', 'noise_level_db',
                'temperature', 'humidity', 'voltage', 'RSSI', 'SNR']:
        if col in df.columns:
            feature_cols.append(col)

    if len(feature_cols) < 2:
        raise ValueError("At least 2 sensor feature columns are required.")

    # Select features and remove missing values
    df_features = df[feature_cols].dropna()

    # Select subset for quantum simulation feasibility
    if len(df_features) > num_items:
        df_subset = df_features.sample(n=num_items, random_state=seed)
    else:
        df_subset = df_features

    # Normalize to [0, 1] range
    normalized = (df_subset - df_subset.min()) / (df_subset.max() - df_subset.min())

    return normalized, feature_cols


# ============================================================
# 2. FAULT CRITERION DEFINITION
# ============================================================

def define_fault_criterion(df_normalized, feature_cols):
    """
    Define fault criteria based on real IoT metrics using the IQR method.

    A node is marked faulty if its temperature reading falls outside
    the interquartile range (1.5 × IQR).

    Args:
        df_normalized: Normalized feature DataFrame
        feature_cols: List of feature column names

    Returns:
        fault_labels: Boolean array (True = faulty)
        thresholds: Dictionary of thresholds used
    """
    # Find temperature column
    temp_col = None
    for col in feature_cols:
        if 'temp' in col.lower():
            temp_col = col
            break

    if temp_col:
        # IQR-based outlier detection
        q1 = df_normalized[temp_col].quantile(0.25)
        q3 = df_normalized[temp_col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr

        # Fault condition: temperature outside IQR range
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
        # Fallback: use 90th percentile of first feature
        col = feature_cols[0]
        threshold = df_normalized[col].quantile(0.9)
        fault_labels = df_normalized[col] > threshold
        thresholds = {f'{col}_threshold': threshold, 'method': 'percentile'}

    return fault_labels.values, thresholds


# ============================================================
# 3. ORACLE CREATION
# ============================================================

def create_real_fault_oracle(fault_labels, num_qubits):
    """
    Create a quantum oracle based on real fault labels.

    The oracle marks faulty node states by applying a phase flip
    (via multi-controlled Z) to the corresponding basis states.

    Args:
        fault_labels: Boolean array indicating faulty nodes
        num_qubits: Number of index qubits

    Returns:
        oracle: QuantumCircuit implementing the oracle
    """
    qc = QuantumCircuit(num_qubits, name="RealFaultOracle")

    for idx, is_faulty in enumerate(fault_labels):
        if is_faulty and idx < 2**num_qubits:
            # Convert index to binary string
            binary = format(idx, f'0{num_qubits}b')

            # Apply X gates to qubits where bit is '0'
            for qubit, bit in enumerate(binary):
                if bit == '0':
                    qc.x(qubit)

            # Multi-controlled Z (phase flip on |11...1>)
            if num_qubits > 1:
                qc.h(num_qubits - 1)
                qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
                qc.h(num_qubits - 1)
            else:
                qc.z(0)

            # Restore original state
            for qubit, bit in enumerate(binary):
                if bit == '0':
                    qc.x(qubit)

    return qc


# ============================================================
# 4. GROVER CIRCUIT CONSTRUCTION
# ============================================================

def build_grover_circuit_with_real_oracle(fault_labels, num_qubits):
    """
    Build the complete Grover circuit using the real fault oracle.

    Steps:
        1. Initialize uniform superposition (H gates)
        2. Apply Oracle + Diffuser for k iterations
        3. Measure index qubits

    Args:
        fault_labels: Boolean array of faulty nodes
        num_qubits: Number of index qubits

    Returns:
        qc: Complete Grover circuit
        iterations: Number of Grover iterations
        num_marked: Number of marked (faulty) states
    """
    num_marked = sum(fault_labels)
    total_states = 2 ** num_qubits

    # Calculate optimal iterations: k ≈ π/4 × √(N/M)
    if num_marked > 0:
        theta = math.asin(math.sqrt(num_marked / total_states))
        iterations = max(1, int(round(math.pi / (4 * theta) - 0.5)))
    else:
        iterations = 0

    # Create oracle
    oracle = create_real_fault_oracle(fault_labels, num_qubits)

    # Create complete circuit
    qc = QuantumCircuit(num_qubits, num_qubits, name="GroverRealFault")

    # Step 1: Uniform superposition
    qc.h(range(num_qubits))

    # Step 2-3: Iterative Oracle + Diffuser
    for _ in range(iterations):
        # Apply Oracle
        qc.compose(oracle, inplace=True)

        # Diffuser: H, X, MCZ, X, H
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

    # Step 4: Measurement
    qc.measure(range(num_qubits), range(num_qubits))

    return qc, iterations, num_marked


# ============================================================
# 5. NOISE SIMULATION
# ============================================================

def run_noisy_simulation(circuit, shots=SHOTS, noise_level=0.01):
    """
    Run Grover circuit on a noisy Aer simulator.

    Args:
        circuit: Grover quantum circuit
        shots: Number of measurement shots
        noise_level: Single-qubit depolarizing error rate

    Returns:
        counts: Measurement counts dictionary
    """
    # Create noise model
    noise_model = NoiseModel()

    # Single-qubit depolarizing noise
    error_1q = depolarizing_error(noise_level, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'sx', 'x'])

    # Two-qubit depolarizing noise (higher error rate)
    error_2q = depolarizing_error(noise_level * 5, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    # Create noisy simulator
    simulator = AerSimulator(
        method='density_matrix',
        noise_model=noise_model
    )

    # Transpile circuit for noisy backend
    transpiled = transpile(circuit, simulator, optimization_level=1)

    # Run simulation
    job = simulator.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts(transpiled)

    return dict(counts)


# ============================================================
# 6. RESULTS ANALYSIS
# ============================================================

def analyze_results(counts, fault_labels, num_qubits):
    """
    Analyze measurement results against ground truth fault labels.

    Args:
        counts: Measurement counts from simulation
        fault_labels: Boolean array of true faulty nodes
        num_qubits: Number of index qubits

    Returns:
        Dictionary with success probability, best index, etc.
    """
    total_shots = sum(counts.values())
    marked_indices = [i for i, f in enumerate(fault_labels) if f]
    marked_set = set(marked_indices)

    # Count hits on marked states
    marked_hits = 0
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx in marked_set:
            marked_hits += count

    success_probability = marked_hits / total_shots if total_shots > 0 else 0

    # Most probable measurement
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
# 7. MAIN WORKFLOW
# ============================================================

def main():
    print("=" * 60)
    print("Grover-Based Fault Detection on Real IoT Data")
    print("=" * 60)

    # Step 1: Load real data
    try:
        df_normalized, feature_cols = load_real_iot_data(CSV_PATH, num_items=128)
        print(f"\n[OK] Data loaded: {len(df_normalized)} records")
        print(f"     Features: {feature_cols}")
    except FileNotFoundError:
        print(f"\n[WARN] File {CSV_PATH} not found. Using simulated real data...")
        np.random.seed(RANDOM_SEED)
        df_normalized = pd.DataFrame({
            'temperature_celsius': np.random.normal(25, 5, 128),
            'humidity_percent': np.random.normal(60, 15, 128),
            'noise_level_db': np.random.normal(50, 10, 128)
        })
        feature_cols = ['temperature_celsius', 'humidity_percent', 'noise_level_db']

    # Step 2: Define fault criterion
    fault_labels, thresholds = define_fault_criterion(df_normalized, feature_cols)
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
    print(f"  Success probability: {result_ideal['success_probability']*100:.2f}%")
    print(f"  Most probable index: {result_ideal['best_index']}")
    print(f"  Is faulty: {result_ideal['best_is_faulty']}")

    # Step 6: Noisy simulation
    print("\n" + "-" * 60)
    print("Noisy Simulation (Depolarizing Noise)")
    print("-" * 60)

    for noise_level in [0.001, 0.005, 0.01]:
        counts_noisy = run_noisy_simulation(circuit, shots=SHOTS, noise_level=noise_level)
        result_noisy = analyze_results(counts_noisy, fault_labels, num_qubits)
        print(f"  Noise {noise_level*100:.1f}%: Success = {result_noisy['success_probability']*100:.2f}%")

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

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()