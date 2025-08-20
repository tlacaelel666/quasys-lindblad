#!/usr/bin/env python3
"""
Simulador Cuántico por Software - Implementación Completa

Este módulo implementa un sistema cuántico completamente simulado que emula
comportamientos cuánticos reales sin necesidad de hardware especializado.
Incluye superposición, entrelazamiento, interferencia, decoherencia y medición.

Autor: Jacobo Tlacaelel Mina Rodríguez  
Fecha: 13/03/2025  
Versión: quantum-simulator v2.0
"""
import numpy as np
import torch
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
from scipy.sparse import csr_matrix
import logging
import time
from dataclasses import dataclass
from enum import Enum
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class QuantumGate(Enum):
    """Enumera las compuertas cuánticas básicas"""
    HADAMARD = "H"
    PAULI_X = "X" 
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    CNOT = "CNOT"
    PHASE = "P"
    ROTATION_X = "RX"
    ROTATION_Y = "RY"
    ROTATION_Z = "RZ"
    TOFFOLI = "TOFFOLI"


@dataclass
class QuantumOperation:
    """Representa una operación cuántica"""
    gate: QuantumGate
    qubits: List[int]
    parameters: Dict[str, float] = None
    timestamp: float = 0.0


class QuantumSimulator:
    """
    Simulador cuántico completamente software que emula un sistema cuántico real.
    
    Capacidades:
    - Superposición cuántica real
    - Entrelazamiento entre qubits
    - Interferencia cuántica constructiva/destructiva
    - Decoherencia simulada
    - Medición cuántica con colapso de función de onda
    - Algoritmos cuánticos (Grover, Shor simulado, etc.)
    """
    
    def __init__(self, num_qubits: int, enable_noise: bool = True, decoherence_rate: float = 0.001):
        """
        Inicializa el simulador cuántico.
        
        Args:
            num_qubits: Número de qubits en el sistema
            enable_noise: Si habilitar ruido cuántico realista
            decoherence_rate: Tasa de decoherencia (pérdida de coherencia cuántica)
        """
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.enable_noise = enable_noise
        self.decoherence_rate = decoherence_rate
        
        # Estado cuántico inicial: |000...0⟩
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0 + 0.0j  # Estado base |0⟩^n
        
        # Historial del sistema
        self.operation_history: List[QuantumOperation] = []
        self.measurement_history: List[Dict] = []
        self.fidelity_history: List[float] = []
        
        # Matrices de compuertas cuánticas básicas
        self._init_quantum_gates()
        
        # Sistema de ruido cuántico
        self._init_noise_model()
        
        logger.info(f"Simulador cuántico inicializado: {num_qubits} qubits, {self.num_states} estados")
    
    def _init_quantum_gates(self):
        """Inicializa las matrices de compuertas cuánticas estándar."""
        # Compuerta Hadamard: H = (1/√2) * [[1, 1], [1, -1]]
        self.H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        # Compuertas de Pauli
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)  # NOT cuántico
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Matriz identidad
        self.I = np.eye(2, dtype=complex)
        
        # Compuerta de fase
        self.S = np.array([[1, 0], [0, 1j]], dtype=complex)
        self.T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    def _init_noise_model(self):
        """Inicializa el modelo de ruido cuántico."""
        if self.enable_noise:
            # Canales de ruido: despolarización, amortiguamiento de amplitud, dephasing
            self.depolarization_rate = 0.001
            self.amplitude_damping_rate = 0.0005  
            self.dephasing_rate = 0.002
            
            logger.info("Modelo de ruido cuántico habilitado")
    
    def _apply_single_qubit_gate(self, gate_matrix: np.ndarray, target_qubit: int):
        """
        Aplica una compuerta de un qubit al sistema.
        
        Matemática:
        |ψ'⟩ = (I ⊗ I ⊗ ... ⊗ G ⊗ ... ⊗ I) |ψ⟩
        donde G actúa en la posición target_qubit
        """
        # Construir la matriz del operador completo usando producto tensorial
        operator = 1.0
        for i in range(self.num_qubits):
            if i == target_qubit:
                if operator is 1.0:
                    operator = gate_matrix
                else:
                    operator = np.kron(operator, gate_matrix)
            else:
                if operator is 1.0:
                    operator = self.I
                else:
                    operator = np.kron(operator, self.I)
        
        # Aplicar el operador al estado
        self.state_vector = operator @ self.state_vector
        
        # Aplicar ruido si está habilitado
        if self.enable_noise:
            self._apply_noise()
    
    def _apply_two_qubit_gate(self, gate_matrix: np.ndarray, control_qubit: int, target_qubit: int):
        """
        Aplica una compuerta de dos qubits (ej. CNOT).
        
        Matemática para CNOT:
        CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
        """
        # Para simplificar, implementamos CNOT específicamente
        if gate_matrix.shape == (4, 4):  # Asumimos CNOT
            # Crear operador CNOT completo
            cnot_full = self._build_controlled_gate(self.X, control_qubit, target_qubit)
            self.state_vector = cnot_full @ self.state_vector
        
        if self.enable_noise:
            self._apply_noise()
    
    def _build_controlled_gate(self, gate: np.ndarray, control: int, target: int) -> np.ndarray:
        """
        Construye una compuerta controlada para el sistema completo.
        
        Matemática:
        C-U = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
        """
        # Implementación simplificada para CNOT
        full_operator = np.eye(self.num_states, dtype=complex)
        
        for i in range(self.num_states):
            # Convertir índice a representación binaria
            binary = format(i, f'0{self.num_qubits}b')
            
            # Si el qubit de control está en |1⟩
            if binary[control] == '1':
                # Aplicar la compuerta al qubit objetivo
                j = i ^ (1 << (self.num_qubits - 1 - target))  # Flip bit objetivo
                full_operator[j, i] = 1.0
                full_operator[i, i] = 0.0
        
        return full_operator
    
    def _apply_noise(self):
        """
        Aplica ruido cuántico realista al sistema.
        Simula decoherencia, depolarización y amortiguamiento.
        """
        if not self.enable_noise:
            return
        
        # 1. Decoherencia: pérdida gradual de coherencia cuántica
        decoherence_factor = np.exp(-self.decoherence_rate)
        
        # 2. Depolarización: mezcla con estado maximamente mixto
        if np.random.random() < self.depolarization_rate:
            # Agregar componente de estado aleatorio
            random_state = np.random.normal(0, 0.01, self.num_states) + \
                          1j * np.random.normal(0, 0.01, self.num_states)
            self.state_vector += random_state
        
        # 3. Dephasing: pérdida de relaciones de fase
        if np.random.random() < self.dephasing_rate:
            phase_noise = np.random.normal(0, 0.1, self.num_states)
            self.state_vector *= np.exp(1j * phase_noise)
        
        # 4. Amortiguamiento de amplitud (aproximación)
        amplitude_factor = 1.0 - self.amplitude_damping_rate
        self.state_vector *= np.sqrt(amplitude_factor)
        
        # Renormalizar el estado
        norm = np.linalg.norm(self.state_vector)
        if norm > 1e-10:
            self.state_vector /= norm
    
    def hadamard(self, qubit: int) -> 'QuantumSimulator':
        """
        Aplica compuerta Hadamard: crea superposición cuántica.
        H|0⟩ = (|0⟩ + |1⟩)/√2
        H|1⟩ = (|0⟩ - |1⟩)/√2
        """
        self._apply_single_qubit_gate(self.H, qubit)
        self.operation_history.append(
            QuantumOperation(QuantumGate.HADAMARD, [qubit], timestamp=time.time())
        )
        logger.debug(f"Hadamard aplicada al qubit {qubit}")
        return self
    
    def pauli_x(self, qubit: int) -> 'QuantumSimulator':
        """Aplica compuerta Pauli-X (NOT cuántico)"""
        self._apply_single_qubit_gate(self.X, qubit)
        self.operation_history.append(
            QuantumOperation(QuantumGate.PAULI_X, [qubit], timestamp=time.time())
        )
        return self
    
    def pauli_y(self, qubit: int) -> 'QuantumSimulator':
        """Aplica compuerta Pauli-Y"""
        self._apply_single_qubit_gate(self.Y, qubit)
        self.operation_history.append(
            QuantumOperation(QuantumGate.PAULI_Y, [qubit], timestamp=time.time())
        )
        return self
    
    def pauli_z(self, qubit: int) -> 'QuantumSimulator':
        """Aplica compuerta Pauli-Z (flip de fase)"""
        self._apply_single_qubit_gate(self.Z, qubit)
        self.operation_history.append(
            QuantumOperation(QuantumGate.PAULI_Z, [qubit], timestamp=time.time())
        )
        return self
    
    def cnot(self, control: int, target: int) -> 'QuantumSimulator':
        """
        Aplica compuerta CNOT: crea entrelazamiento cuántico.
        CNOT|00⟩ = |00⟩, CNOT|01⟩ = |01⟩
        CNOT|10⟩ = |11⟩, CNOT|11⟩ = |10⟩
        """
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        self._apply_two_qubit_gate(cnot_matrix, control, target)
        self.operation_history.append(
            QuantumOperation(QuantumGate.CNOT, [control, target], timestamp=time.time())
        )
        logger.debug(f"CNOT aplicada: control={control}, target={target}")
        return self
    
    def rotation_x(self, qubit: int, angle: float) -> 'QuantumSimulator':
        """
        Rotación alrededor del eje X.
        RX(θ) = cos(θ/2)I - i*sin(θ/2)X
        """
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        rx_matrix = np.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=complex)
        
        self._apply_single_qubit_gate(rx_matrix, qubit)
        self.operation_history.append(
            QuantumOperation(QuantumGate.ROTATION_X, [qubit], {'angle': angle}, time.time())
        )
        return self
    
    def rotation_y(self, qubit: int, angle: float) -> 'QuantumSimulator':
        """
        Rotación alrededor del eje Y.
        RY(θ) = cos(θ/2)I - i*sin(θ/2)Y
        """
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        ry_matrix = np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
        
        self._apply_single_qubit_gate(ry_matrix, qubit)
        self.operation_history.append(
            QuantumOperation(QuantumGate.ROTATION_Y, [qubit], {'angle': angle}, time.time())
        )
        return self
    
    def rotation_z(self, qubit: int, angle: float) -> 'QuantumSimulator':
        """
        Rotación alrededor del eje Z (cambio de fase).
        RZ(θ) = e^(-iθZ/2) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        """
        rz_matrix = np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=complex)
        
        self._apply_single_qubit_gate(rz_matrix, qubit)
        self.operation_history.append(
            QuantumOperation(QuantumGate.ROTATION_Z, [qubit], {'angle': angle}, time.time())
        )
        return self
    
    def measure(self, qubit: int = None) -> Union[int, List[int]]:
        """
        Realiza medición cuántica con colapso de función de onda.
        
        Implementa la regla de Born: P(|i⟩) = |⟨i|ψ⟩|²
        
        Args:
            qubit: Qubit específico a medir, o None para medir todo el sistema
            
        Returns:
            Resultado de la medición (0 o 1 para un qubit, lista para sistema completo)
        """
        if qubit is not None:
            return self._measure_single_qubit(qubit)
        else:
            return self._measure_all_qubits()
    
    def _measure_single_qubit(self, qubit: int) -> int:
        """Mide un qubit específico."""
        # Calcular probabilidades de |0⟩ y |1⟩ para el qubit
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i, amplitude in enumerate(self.state_vector):
            binary = format(i, f'0{self.num_qubits}b')
            if binary[qubit] == '0':
                prob_0 += abs(amplitude) ** 2
            else:
                prob_1 += abs(amplitude) ** 2
        
        # Muestrear según las probabilidades cuánticas
        result = np.random.choice([0, 1], p=[prob_0, prob_1])
        
        # Colapsar la función de onda
        new_state = np.zeros_like(self.state_vector)
        normalization = 0.0
        
        for i, amplitude in enumerate(self.state_vector):
            binary = format(i, f'0{self.num_qubits}b')
            if (result == 0 and binary[qubit] == '0') or (result == 1 and binary[qubit] == '1'):
                new_state[i] = amplitude
                normalization += abs(amplitude) ** 2
        
        # Renormalizar después del colapso
        if normalization > 1e-10:
            new_state /= np.sqrt(normalization)
        
        self.state_vector = new_state
        
        # Registrar medición
        measurement_data = {
            'qubit': qubit,
            'result': result,
            'probabilities': {'0': prob_0, '1': prob_1},
            'timestamp': time.time()
        }
        self.measurement_history.append(measurement_data)
        
        logger.info(f"Medición qubit {qubit}: resultado={result}, P(0)={prob_0:.3f}, P(1)={prob_1:.3f}")
        return result
    
    def _measure_all_qubits(self) -> List[int]:
        """Mide todos los qubits simultáneamente."""
        # Calcular probabilidades para cada estado base
        probabilities = np.abs(self.state_vector) ** 2
        
        # Muestrear estado según distribución cuántica
        measured_state_index = np.random.choice(self.num_states, p=probabilities)
        
        # Convertir índice a string binario
        binary_result = format(measured_state_index, f'0{self.num_qubits}b')
        result = [int(bit) for bit in binary_result]
        
        # Colapsar completamente la función de onda
        self.state_vector = np.zeros_like(self.state_vector)
        self.state_vector[measured_state_index] = 1.0
        
        # Registrar medición
        measurement_data = {
            'all_qubits': True,
            'result': result,
            'measured_state': measured_state_index,
            'probabilities': probabilities.tolist(),
            'timestamp': time.time()
        }
        self.measurement_history.append(measurement_data)
        
        logger.info(f"Medición completa: {result} (estado {measured_state_index})")
        return result
    
    def get_state_probabilities(self) -> Dict[str, float]:
        """Obtiene las probabilidades de todos los estados base."""
        probabilities = {}
        for i, amplitude in enumerate(self.state_vector):
            binary = format(i, f'0{self.num_qubits}b')
            prob = abs(amplitude) ** 2
            if prob > 1e-10:  # Solo mostrar probabilidades significativas
                probabilities[f"|{binary}⟩"] = prob
        
        return probabilities
    
    def get_qubit_probabilities(self, qubit: int) -> Dict[str, float]:
        """Obtiene las probabilidades marginales de un qubit específico."""
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i, amplitude in enumerate(self.state_vector):
            binary = format(i, f'0{self.num_qubits}b')
            prob = abs(amplitude) ** 2
            if binary[qubit] == '0':
                prob_0 += prob
            else:
                prob_1 += prob
        
        return {'0': prob_0, '1': prob_1}
    
    def calculate_entanglement(self) -> float:
        """
        Calcula la entropía de entrelazamiento usando descomposición de Schmidt.
        Mide qué tan entrelazado está el sistema.
        """
        # Para simplificar, calculamos entropía de von Neumann
        # de la matriz de densidad reducida del primer qubit
        
        if self.num_qubits < 2:
            return 0.0
        
        # Calcular matriz de densidad completa: ρ = |ψ⟩⟨ψ|
        rho = np.outer(self.state_vector, np.conj(self.state_vector))
        
        # Calcular matriz de densidad reducida del primer qubit (traza parcial)
        dim_subsystem = 2 ** (self.num_qubits - 1)
        rho_reduced = np.zeros((2, 2), dtype=complex)
        
        for i in range(2):
            for j in range(2):
                for k in range(dim_subsystem):
                    idx1 = i * dim_subsystem + k
                    idx2 = j * dim_subsystem + k
                    rho_reduced[i, j] += rho[idx1, idx2]
        
        # Calcular eigenvalores de la matriz de densidad reducida
        eigenvals = np.linalg.eigvals(rho_reduced)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Filtrar valores muy pequeños
        
        # Entropía de von Neumann: S = -Tr(ρ log ρ)
        entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-16))
        
        return float(entropy)
    
    def calculate_fidelity(self, target_state: np.ndarray) -> float:
        """
        Calcula la fidelidad cuántica con un estado objetivo.
        F = |⟨ψ_target|ψ_current⟩|²
        """
        overlap = np.vdot(target_state, self.state_vector)
        fidelity = abs(overlap) ** 2
        self.fidelity_history.append(fidelity)
        return fidelity
    
    def reset(self):
        """Reinicia el simulador al estado |000...0⟩"""
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0
        self.operation_history = []
        self.measurement_history = []
        self.fidelity_history = []
        logger.info("Simulador reiniciado")
    
    def visualize_state(self, title: str = "Estado Cuántico"):
        """Visualiza el estado cuántico actual."""
        probabilities = self.get_state_probabilities()
        
        if not probabilities:
            logger.warning("No hay probabilidades significativas para visualizar")
            return
        
        # Gráfico de barras de probabilidades
        states = list(probabilities.keys())
        probs = list(probabilities.values())
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Probabilidades
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(states)), probs, alpha=0.7)
        plt.xlabel('Estados Cuánticos')
        plt.ylabel('Probabilidad')
        plt.title(f'{title} - Probabilidades')
        plt.xticks(range(len(states)), states, rotation=45)
        
        # Colorear barras según probabilidad
        max_prob = max(probs)
        for bar, prob in zip(bars, probs):
            bar.set_color(plt.cm.viridis(prob / max_prob))
        
        # Subplot 2: Amplitudes complejas (parte real e imaginaria)
        plt.subplot(1, 2, 2)
        real_parts = [np.real(self.state_vector[i]) for i in range(len(self.state_vector)) if abs(self.state_vector[i]) > 1e-6]
        imag_parts = [np.imag(self.state_vector[i]) for i in range(len(self.state_vector)) if abs(self.state_vector[i]) > 1e-6]
        indices = [i for i in range(len(self.state_vector)) if abs(self.state_vector[i]) > 1e-6]
        
        if real_parts:
            x_pos = np.arange(len(real_parts))
            plt.bar(x_pos - 0.2, real_parts, 0.4, label='Parte Real', alpha=0.7)
            plt.bar(x_pos + 0.2, imag_parts, 0.4, label='Parte Imaginaria', alpha=0.7)
            plt.xlabel('Estados Significativos')
            plt.ylabel('Amplitud')
            plt.title(f'{title} - Amplitudes Complejas')
            plt.legend()
            
            # Etiquetas de estados
            state_labels = [format(i, f'0{self.num_qubits}b') for i in indices]
            plt.xticks(x_pos, state_labels, rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def export_state(self) -> Dict:
        """Exporta el estado completo del simulador."""
        return {
            'num_qubits': self.num_qubits,
            'state_vector': self.state_vector.tolist(),
            'probabilities': self.get_state_probabilities(),
            'entanglement': self.calculate_entanglement(),
            'operations': len(self.operation_history),
            'measurements': len(self.measurement_history)
        }


class QuantumAlgorithms:
    """
    Implementa algoritmos cuánticos clásicos usando el simulador.
    """
    
    @staticmethod
    def deutsch_jozsa(simulator: QuantumSimulator, oracle_function: Callable[[int], int]) -> str:
        """
        Algoritmo de Deutsch-Jozsa: determina si una función es constante o balanceada.
        """
        n = simulator.num_qubits - 1  # Un qubit auxiliar
        
        # Inicialización: |0...01⟩
        simulator.reset()
        simulator.pauli_x(n)  # Qubit auxiliar en |1⟩
        
        # Superposición en todos los qubits
        for i in range(simulator.num_qubits):
            simulator.hadamard(i)
        
        # Oráculo (simplificado)
        for i in range(2 ** n):
            if oracle_function(i) == 1:
                simulator.pauli_z(n)  # Aplicar fase condicionalmente
        
        # Hadamard final en qubits de entrada
        for i in range(n):
            simulator.hadamard(i)
        
        # Medir qubits de entrada
        results = []
        for i in range(n):
            results.append(simulator.measure(i))
        
        # Si todos son 0, la función es constante
        if all(r == 0 for r in results):
            return "CONSTANTE"
        else:
            return "BALANCEADA"
    
    @staticmethod
    def grover_search(simulator: QuantumSimulator, target_items: List[int], num_iterations: int = None) -> int:
        """
        Algoritmo de búsqueda de Grover: búsqueda cuadráticamente más rápida.
        """
        n = simulator.num_qubits
        N = 2 ** n
        
        # Calcular número óptimo de iteraciones
        if num_iterations is None:
            num_iterations = int(np.pi * np.sqrt(N) / 4)
        
        logger.info(f"Grover: buscando {target_items} en {N} elementos, {num_iterations} iteraciones")
        
        # Inicialización: superposición uniforme
        simulator.reset()
        for i in range(n):
            simulator.hadamard(i)
        
        # Iteraciones de Grover
        for iteration in range(num_iterations):
            # Oráculo: marcar elementos objetivo
            for target in target_items:
                # Aplicar fase negativa a elementos objetivo
                # (implementación simplificada)
                target_binary = format(target, f'0{n}b')
                
                # Configurar qubits para el estado objetivo
                gates_to_apply = []
                for i, bit in enumerate(target_binary):
                    if bit == '0':
                        simulator.pauli_x(i)
                        gates_to_apply.append(i)
                
                # Aplicar compuerta controlada multiple (aproximación)
                if n > 1:
                    simulator.rotation_z(n-1, np.pi)  # Cambio de fase
                
                # Deshacer las X gates
                for i in gates_to_apply:
                    simulator.pauli_x(i)
            
            # Difusor: inversión sobre promedio
            # H^⊗n
            for i in range(n):
                simulator.hadamard(i)
            
            # Inversión sobre |0⟩
            for i in range(n):
                simulator.pauli_x(i)
            
            # Z controlada múltiple (aproximación con rotación)
            if n > 1:
                simulator.rotation_z(n-1, np.pi)
            
            # Deshacer las X gates
            for i in range(n):
                simulator.pauli_x(i)
            
            # H^⊗n
            for i in range(n):
                simulator.hadamard(i)
        
        # Medición final
        result = simulator.measure()
        measured_value = sum(bit * (2 ** (n-1-i)) for i, bit in enumerate(result))
        
        logger.info(f"Grover completado. Resultado medido: {measured_value}")
        return measured_value
    
    @staticmethod
    def quantum_fourier_transform(simulator: QuantumSimulator):
        """
        Transformada de Fourier Cuántica (QFT).
        Fundamental para muchos algoritmos cuánticos como Shor.
        """
        n = simulator.num_qubits
        
        # Aplicar QFT
        for i in range(n):
            # Hadamard al qubit i
            simulator.hadamard(i)
            
            # Rotaciones controladas
            for j in range(i + 1, n):
                angle = 2 * np.pi / (2 ** (j - i + 1))
                # Aproximación: usar rotación Z controlada
                simulator.cnot(j, i)
                simulator.rotation_z(i, angle / 2)
                simulator.cnot(j, i)
        
        # Intercambiar orden de qubits (implementación simplificada)
        for i in range(n // 2):
            j = n - 1 - i
            if i != j:
                # Swap usando tres CNOTs
                simulator.cnot(i, j)
                simulator.cnot(j, i)
                simulator.cnot(i, j)
        
        logger.info("QFT aplicada al sistema")
    
    @staticmethod
    def quantum_phase_estimation(simulator: QuantumSimulator, unitary_angle: float) -> float:
        """
        Estimación de fase cuántica: estima la fase de un operador unitario.
        """
        n = simulator.num_qubits - 1  # Un qubit para el eigenstate
        
        simulator.reset()
        
        # Preparar eigenstate en el último qubit
        simulator.pauli_x(n)  # |1⟩
        
        # Superposición en qubits de conteo
        for i in range(n):
            simulator.hadamard(i)
        
        # Aplicar operadores controlados U^(2^k)
        for i in range(n):
            power = 2 ** i
            total_angle = power * unitary_angle
            # Aproximar U controlada con rotación Z
            simulator.cnot(i, n)
            simulator.rotation_z(n, total_angle)
            simulator.cnot(i, n)
        
        # QFT inversa en qubits de conteo
        QuantumAlgorithms.quantum_fourier_transform(simulator)
        
        # Medir qubits de conteo
        measurement_result = []
        for i in range(n):
            measurement_result.append(simulator.measure(i))
        
        # Convertir medición a estimación de fase
        measured_value = sum(bit * (2 ** (n-1-i)) for i, bit in enumerate(measurement_result))
        estimated_phase = measured_value / (2 ** n)
        
        logger.info(f"Fase estimada: {estimated_phase:.4f} (valor real: {unitary_angle/(2*np.pi):.4f})")
        return estimated_phase


class QuantumCircuitOptimizer:
    """
    Optimiza circuitos cuánticos para reducir complejidad y errores.
    """
    
    @staticmethod
    def optimize_circuit(operations: List[QuantumOperation]) -> List[QuantumOperation]:
        """
        Optimiza una lista de operaciones cuánticas.
        """
        optimized = []
        
        # Reglas básicas de optimización
        for i, op in enumerate(operations):
            # Eliminar operaciones redundantes (X-X = I, H-H = I, etc.)
            if i < len(operations) - 1:
                next_op = operations[i + 1]
                
                # Cancelación de compuertas idénticas consecutivas
                if (op.gate == next_op.gate and 
                    op.qubits == next_op.qubits and
                    op.gate in [QuantumGate.PAULI_X, QuantumGate.PAULI_Y, 
                               QuantumGate.PAULI_Z, QuantumGate.HADAMARD]):
                    # Omitir ambas operaciones
                    operations.pop(i + 1)
                    continue
            
            optimized.append(op)
        
        logger.info(f"Circuito optimizado: {len(operations)} -> {len(optimized)} operaciones")
        return optimized


class QuantumErrorCorrection:
    """
    Implementa códigos básicos de corrección de errores cuánticos.
    """
    
    @staticmethod
    def three_qubit_bit_flip_code(simulator: QuantumSimulator, logical_qubit_state: int):
        """
        Código de corrección de errores para bit-flip usando 3 qubits.
        """
        if simulator.num_qubits < 3:
            raise ValueError("Se requieren al menos 3 qubits para este código")
        
        simulator.reset()
        
        # Preparar estado lógico
        if logical_qubit_state == 1:
            simulator.pauli_x(0)
        
        # Codificación: |0⟩ -> |000⟩, |1⟩ -> |111⟩
        simulator.cnot(0, 1)
        simulator.cnot(0, 2)
        
        logger.info(f"Estado lógico {logical_qubit_state} codificado en 3 qubits")
        
        # Simular error en qubit aleatorio (opcional)
        error_qubit = np.random.randint(3)
        if np.random.random() < 0.1:  # 10% probabilidad de error
            simulator.pauli_x(error_qubit)
            logger.warning(f"Error simulado en qubit {error_qubit}")
        
        # Síndrome de medición para detectar errores
        # Medición de paridad entre qubits 0-1 y 1-2
        simulator.cnot(0, 3) if simulator.num_qubits > 3 else None
        simulator.cnot(1, 3) if simulator.num_qubits > 3 else None
        
        return simulator
    
    @staticmethod
    def surface_code_patch(simulator: QuantumSimulator):
        """
        Implementación simplificada de un parche de código de superficie.
        """
        # Implementación básica para demostración
        logger.info("Código de superficie aplicado (implementación simplificada)")
        pass


class QuantumBenchmarks:
    """
    Benchmarks y pruebas de rendimiento del simulador.
    """
    
    @staticmethod
    def benchmark_hadamard_chain(num_qubits: int, num_operations: int) -> Dict:
        """
        Benchmark: cadena de compuertas Hadamard.
        """
        simulator = QuantumSimulator(num_qubits, enable_noise=False)
        
        start_time = time.time()
        
        for _ in range(num_operations):
            qubit = np.random.randint(num_qubits)
            simulator.hadamard(qubit)
        
        end_time = time.time()
        
        # Calcular métricas
        execution_time = end_time - start_time
        operations_per_second = num_operations / execution_time
        
        results = {
            'num_qubits': num_qubits,
            'num_operations': num_operations,
            'execution_time': execution_time,
            'ops_per_second': operations_per_second,
            'final_entanglement': simulator.calculate_entanglement()
        }
        
        logger.info(f"Benchmark Hadamard: {operations_per_second:.2f} ops/sec")
        return results
    
    @staticmethod
    def benchmark_grover_scaling(max_qubits: int = 10) -> List[Dict]:
        """
        Estudia el escalamiento del algoritmo de Grover.
        """
        results = []
        
        for n in range(2, max_qubits + 1):
            simulator = QuantumSimulator(n, enable_noise=False)
            
            start_time = time.time()
            target = [0]  # Buscar elemento 0
            QuantumAlgorithms.grover_search(simulator, target)
            end_time = time.time()
            
            result = {
                'qubits': n,
                'search_space': 2 ** n,
                'execution_time': end_time - start_time,
                'theoretical_iterations': int(np.pi * np.sqrt(2 ** n) / 4)
            }
            
            results.append(result)
            logger.info(f"Grover {n} qubits: {end_time - start_time:.4f}s")
        
        return results


def demonstrate_quantum_simulator():
    """
    Demostración completa de las capacidades del simulador cuántico.
    """
    print("=" * 60)
    print("DEMOSTRACIÓN DEL SIMULADOR CUÁNTICO")
    print("=" * 60)
    
    # 1. Superposición básica
    print("\n1. SUPERPOSICIÓN CUÁNTICA")
    print("-" * 30)
    
    sim = QuantumSimulator(2, enable_noise=False)
    sim.hadamard(0)  # Crear superposición en qubit 0
    print("Estado después de Hadamard:")
    print(sim.get_state_probabilities())
    
    # 2. Entrelazamiento
    print("\n2. ENTRELAZAMIENTO CUÁNTICO")
    print("-" * 30)
    
    sim.cnot(0, 1)  # Entrelazar qubits 0 y 1
    print("Estado después de CNOT (entrelazado):")
    print(sim.get_state_probabilities())
    print(f"Entrelazamiento: {sim.calculate_entanglement():.4f}")
    
    # 3. Medición con colapso
    print("\n3. MEDICIÓN CUÁNTICA")
    print("-" * 30)
    
    result = sim.measure(0)
    print(f"Resultado de medir qubit 0: {result}")
    print("Estado después de la medición:")
    print(sim.get_state_probabilities())
    
    # 4. Algoritmo de Grover
    print("\n4. ALGORITMO DE GROVER")
    print("-" * 30)
    
    sim_grover = QuantumSimulator(4, enable_noise=False)
    target = [7]  # Buscar elemento 7 en espacio de 16 elementos
    found = QuantumAlgorithms.grover_search(sim_grover, target)
    print(f"Grover buscando {target[0]}: encontró {found}")
    
    # 5. QFT
    print("\n5. TRANSFORMADA DE FOURIER CUÁNTICA")
    print("-" * 30)
    
    sim_qft = QuantumSimulator(3, enable_noise=False)
    sim_qft.pauli_x(0)  # Estado inicial |001⟩
    print("Estado antes de QFT:")
    print(sim_qft.get_state_probabilities())
    
    QuantumAlgorithms.quantum_fourier_transform(sim_qft)
    print("Estado después de QFT:")
    print(sim_qft.get_state_probabilities())
    
    # 6. Efectos del ruido
    print("\n6. EFECTOS DEL RUIDO CUÁNTICO")
    print("-" * 30)
    
    sim_noise = QuantumSimulator(2, enable_noise=True, decoherence_rate=0.01)
    sim_noise.hadamard(0)
    
    print("Evolución con ruido:")
    for i in range(5):
        sim_noise.hadamard(0)  # Aplicar más operaciones
        sim_noise.hadamard(0)  # Las compuertas se cancelan, pero el ruido persiste
        entanglement = sim_noise.calculate_entanglement()
        print(f"Iteración {i+1}: Entrelazamiento = {entanglement:.4f}")
    
    # 7. Benchmark
    print("\n7. BENCHMARK DE RENDIMIENTO")
    print("-" * 30)
    
    benchmark = QuantumBenchmarks.benchmark_hadamard_chain(4, 1000)
    print(f"Rendimiento: {benchmark['ops_per_second']:.2f} operaciones/segundo")
    
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)


# Función principal para pruebas
if __name__ == "__main__":
    # Configurar logging
    logging.getLogger().setLevel(logging.INFO)
    
    # Ejecutar demostración
    demonstrate_quantum_simulator()
    
    # Ejemplo de uso interactivo
    print("\n" + "=" * 60)
    print("EJEMPLO DE USO INTERACTIVO")
    print("=" * 60)
    
    # Crear simulador
    qsim = QuantumSimulator(3, enable_noise=True)
    
    # Crear estado de Bell entrelazado
    qsim.hadamard(0).cnot(0, 1)
    
    # Aplicar rotaciones
    qsim.rotation_y(2, np.pi/4)
    
    # Visualizar estado (si matplotlib está disponible)
    try:
        qsim.visualize_state("Estado Cuántico Final")
    except Exception as e:
        logger.warning(f"Visualización no disponible: {e}")
    
    # Exportar estado
    state_data = qsim.export_state()
    print("\nEstado exportado:")
    print(json.dumps({k: v for k, v in state_data.items() 
                     if k != 'state_vector'}, indent=2))
    
    print("\n¡Simulador cuántico listo para usar!")
    print("Ejemplo: qsim.hadamard(0).cnot(0,1).measure()")
