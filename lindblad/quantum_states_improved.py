class QuantumVisualizer:
    """Clase para visualización de estados cuánticos"""
    
    @staticmethod
    def plot_probability_distribution(state: QuantumState, title: str = "Distribución de Probabilidades"):
        """
        Grafica la distribución de probabilidades del estado
        
        Args:
            state: Estado cuántico
            title: Título del gráfico
        """
        try:
            probs = QuantumOperations.measure_probability_distribution(state)
            
            states = list(probs.keys())
            probabilities = list(probs.values())
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(states, probabilities, alpha=0.7, 
                          color=['skyblue' if p > 0.01 else 'lightgray' for p in probabilities])
            
            # Añadir valores sobre las barras
            for bar, prob in zip(bars, probabilities):
                if prob > 0.001:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{prob:.3f}', ha='center', va='bottom')
            
            plt.xlabel('Estados Base')
            plt.ylabel('Probabilidad')
            plt.title(title)
            plt.xticks(rotation=45 if len(states) > 8 else 0)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib no disponible. Instalarlo para visualizaciones.")
            print("Distribución de probabilidades:")
            for state, prob in probs.items():
                if prob > 0.001:
                    print(f"  |{state}⟩: {prob:.4f}")
    
    @staticmethod
    def plot_bloch_sphere_2d(state: QuantumState):
        """
        Representa un estado de 1 qubit en el círculo de Bloch (proyección 2D)
        
        Args:
            state: Estado de un solo qubit
        """
        if state.n_qubits != 1:
            raise ValueError("Solo se puede representar estados de 1 qubit en la esfera de Bloch")
        
        try:
            # Extraer amplitudes
            alpha, beta = state.amplitudes[0], state.amplitudes[1]
            
            # Calcular coordenadas de Bloch
            x = 2 * np.real(np.conj(alpha) * beta)
            y = 2 * np.imag(np.conj(alpha) * beta)
            z = abs(alpha)**2 - abs(beta)**2
            
            # Gráfico 2D (proyecciones)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Proyección XY
            circle1 = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
            ax1.add_patch(circle1)
            ax1.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='red', ec='red')
            ax1.scatter([x], [y], c='red', s=100)
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('Proyección XY')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Proyección XZ
            circle2 = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
            ax2.add_patch(circle2)
            ax2.arrow(0, 0, x, z, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
            ax2.scatter([x], [z], c='blue', s=100)
            ax2.set_xlim(-1.2, 1.2)
            ax2.set_ylim(-1.2, 1.2)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Z')
            ax2.set_title('Proyección XZ')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            
            plt.suptitle(f'Representación de Bloch: {state}')
            plt.tight_layout()
            plt.show()
            
            print(f"Coordenadas de Bloch: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
            
        except ImportError:
            print("Matplotlib no disponible.")
            print(f"Estado: {state}")

class QuantumCircuit:
    """Clase para representar y simular circuitos cuánticos"""
    
    def __init__(self, n_qubits: int):
        """
        Inicializa un circuito cuántico
        
        Args:
            n_qubits: Número de qubits del circuito
        """
        self.n_qubits = n_qubits
        self.operations = []
        self.state = QuantumStateManager.create_computational_state("0" * n_qubits)
    
    def add_gate(self, gate: str, qubit: int, **kwargs):
        """
        Añade una compuerta al circuito
        
        Args:
            gate: Tipo de compuerta ("H", "X", "Y", "Z", "RZ", etc.)
            qubit: Qubit al que aplicar la compuerta
            **kwargs: Parámetros adicionales (ej. theta para rotaciones)
        """
        self.operations.append({"gate": gate, "qubit": qubit, "params": kwargs})
        return self
    
    def add_cnot(self, control: int, target: int):
        """
        Añade una compuerta CNOT al circuito
        
        Args:
            control: Qubit de control
            target: Qubit objetivo
        """
        self.operations.append({"gate": "CNOT", "control": control, "target": target})
        return self
    

    def execute(self) -> QuantumState:
        """
        Ejecuta el circuito y retorna el estado final
        
        Returns:
            Estado cuántico final después de aplicar todas las operaciones
        """
        current_state = self.state.copy()
        
        for op in self.operations:
            if op["gate"] == "H":
                gate_matrix = QuantumGates.hadamard()
                current_state = QuantumOperations.apply_single_qubit_gate(
                    current_state, gate_matrix, op["qubit"])
                    
            elif op["gate"] == "X":
                gate_matrix = QuantumGates.pauli_x()
                current_state = QuantumOperations.apply_single_qubit_gate(
                    current_state, gate_matrix, op["qubit"])
                    
            elif op["gate"] == "Y":
                gate_matrix = QuantumGates.pauli_y()
                current_state = QuantumOperations.apply_single_qubit_gate(
                    current_state, gate_matrix, op["qubit"])
                    
            elif op["gate"] == "Z":
                gate_matrix = QuantumGates.pauli_z()
                current_state = QuantumOperations.apply_single_qubit_gate(
                    current_state, gate_matrix, op["qubit"])
                    
            elif op["gate"] == "RX":
                theta = op["params"].get("theta", 0)
                gate_matrix = QuantumGates.rotation_x(theta)
                current_state = QuantumOperations.apply_single_qubit_gate(
                    current_state, gate_matrix, op["qubit"])
                    
            elif op["gate"] == "RY":
                theta = op["params"].get("theta", 0)
                gate_matrix = QuantumGates.rotation_y(theta)
                current_state = QuantumOperations.apply_single_qubit_gate(
                    current_state, gate_matrix, op["qubit"])
                    
            elif op["gate"] == "RZ":
                theta = op["params"].get("theta", 0)
                gate_matrix = QuantumGates.rotation_z(theta)
                current_state = QuantumOperations.apply_single_qubit_gate(
                    current_state, gate_matrix, op["qubit"])
                    
            elif op["gate"] == "CNOT":
                if current_state.n_qubits == 2:
                    gate_matrix = QuantumGates.cnot()
                    current_state = QuantumOperations.apply_two_qubit_gate(
                        current_state, gate_matrix, op["control"], op["target"])
                else:
                    warnings.warn("CNOT solo implementado para sistemas de 2 qubits")
            
            else:
                raise ValueError(f"Compuerta no reconocida: {op['gate']}")
        
        return current_state
    
    def measure(self, qubits: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Simula medición del circuito
        
        Args:
            qubits: Lista de qubits a medir (None = todos)
            
        Returns:
            Diccionario con resultados de la medición
        """
        final_state = self.execute()
        
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        result, post_measurement_state = QuantumOperations.measure_state(final_state)
        
        # Convertir resultado a string binario
        binary_result = format(result, f'0{final_state.n_qubits}b')
        
        return {
            "measurement_result": binary_result,
            "probability": final_state.probability(result),
            "post_measurement_state": post_measurement_state,
            "final_state": final_state
        }
    
    def __str__(self) -> str:
        """Representación string del circuito"""
        circuit_str = f"Circuito Cuántico ({self.n_qubits} qubits):\n"
        for i, op in enumerate(self.operations):
            if op["gate"] == "CNOT":
                circuit_str += f"  {i+1}. CNOT: control={op['control']}, target={op['target']}\n"
            else:
                params_str = ", ".join([f"{k}={v}" for k, v in op.get("params", {}).items()])
                if params_str:
                    circuit_str += f"  {i+1}. {op['gate']}(qubit={op['qubit']}, {params_str})\n"
                else:
                    circuit_str += f"  {i+1}. {op['gate']}(qubit={op['qubit']})\n"
        return circuit_str

class QuantumEntanglement:
    """Clase para análisis de entrelazamiento cuántico"""
    
    @staticmethod
    def von_neumann_entropy(density_matrix: np.ndarray) -> float:
        """
        Calcula la entropía de von Neumann
        
        Args:
            density_matrix: Matriz de densidad
            
        Returns:
            Entropía de von Neumann
        """
        eigenvals = np.linalg.eigvals(density_matrix)
        # Filtrar valores propios positivos
        positive_eigenvals = eigenvals[eigenvals > 1e-12]
        
        if len(positive_eigenvals) == 0:
            return 0.0
        
        return -np.sum(positive_eigenvals * np.log2(positive_eigenvals))
    
    @staticmethod
    def entanglement_entropy(state: QuantumState, subsystem_qubits: List[int]) -> float:
        """
        Calcula la entropía de entrelazamiento
        
        Args:
            state: Estado cuántico total
            subsystem_qubits: Lista de qubits del subsistema A
            
        Returns:
            Entropía de entrelazamiento entre A y el resto
        """
        if state.n_qubits <= 1:
            return 0.0
        
        try:
            # Calcular traza parcial sobre el complemento
            complement_qubits = [q for q in range(state.n_qubits) if q not in subsystem_qubits]
            
            if len(complement_qubits) == 1 and state.n_qubits == 2:
                reduced_density = QuantumOperations.partial_trace(state, complement_qubits)
                return QuantumEntanglement.von_neumann_entropy(reduced_density)
            else:
                # Para casos más complejos, usar aproximación
                warnings.warn("Cálculo de entrelazamiento limitado a casos simples")
                return 0.0
                
        except Exception as e:
            warnings.warn(f"Error calculando entrelazamiento: {e}")
            return 0.0
    
    @staticmethod
    def is_entangled(state: QuantumState, tolerance: float = 1e-10) -> bool:
        """
        Determina si un estado está entrelazado (implementación básica)
        
        Args:
            state: Estado cuántico
            tolerance: Tolerancia para comparaciones numéricas
            
        Returns:
            True si el estado está entrelazado
        """
        if state.n_qubits <= 1:
            return False
        
        # Para estados de 2 qubits, verificar si es separable
        if state.n_qubits == 2:
            # Un estado |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩ es separable
            # si y solo si αδ - βγ = 0
            alpha, beta, gamma, delta = state.amplitudes
            determinant = abs(alpha * delta - beta * gamma)
            return determinant > tolerance
        else:
            # Para más qubits, usar entropía de entrelazamiento
            entropy = QuantumEntanglement.entanglement_entropy(state, [0])
            return entropy > toleranceclass QuantumStateManager:import numpy as np
import sympy as sp
from typing import List, Tuple, Union, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from scipy.linalg import expm
import warnings

class QubitState(Enum):
    """Enumeración para los estados base de un qubit"""
    ZERO = "0"
    ONE = "1"

class PauliGate(Enum):
    """Enumeración para las compuertas de Pauli"""
    I = "I"  # Identidad
    X = "X"  # Pauli-X (NOT)
    Y = "Y"  # Pauli-Y
    Z = "Z"  # Pauli-Z

class CommonGates(Enum):
    """Enumeración para compuertas comunes"""
    H = "H"      # Hadamard
    CNOT = "CNOT"  # Controlled-NOT
    T = "T"      # T gate
    S = "S"      # S gate

@dataclass
class QuantumState:
    """Clase para representar un estado cuántico"""
    amplitudes: np.ndarray
    basis_states: List[str]
    n_qubits: Optional[int] = field(init=False)
    
    def __post_init__(self):
        """Validación después de inicialización"""
        if len(self.amplitudes) != len(self.basis_states):
            raise ValueError("El número de amplitudes debe coincidir con el número de estados base")
        
        # Calcular número de qubits
        self.n_qubits = int(np.log2(len(self.basis_states)))
        
        # Validar que sea una potencia de 2
        if 2**self.n_qubits != len(self.basis_states):
            raise ValueError("El número de estados base debe ser una potencia de 2")
        
        # Normalización automática
        norm = np.linalg.norm(self.amplitudes)
        if not np.isclose(norm, 1.0, atol=1e-10):
            if norm > 1e-10:  # Evitar división por cero
                print(f"Advertencia: Estado no normalizado (norma = {norm:.6f}). Normalizando...")
                self.amplitudes = self.amplitudes / norm
            else:
                raise ValueError("Vector de amplitudes nulo")
    
    def probability(self, state_index: int) -> float:
        """Calcula la probabilidad de medir un estado específico"""
        if not 0 <= state_index < len(self.amplitudes):
            raise IndexError("Índice de estado fuera de rango")
        return abs(self.amplitudes[state_index])**2
    
    def expectation_value(self, observable: np.ndarray) -> complex:
        """Calcula el valor esperado de un observable"""
        if observable.shape != (len(self.amplitudes), len(self.amplitudes)):
            raise ValueError("Las dimensiones del observable no coinciden con el estado")
        return np.conj(self.amplitudes).T @ observable @ self.amplitudes
    
    def fidelity(self, other: 'QuantumState') -> float:
        """Calcula la fidelidad con otro estado cuántico"""
        if len(self.amplitudes) != len(other.amplitudes):
            raise ValueError("Los estados deben tener la misma dimensión")
        return abs(np.conj(self.amplitudes).T @ other.amplitudes)**2
    
    def trace_distance(self, other: 'QuantumState') -> float:
        """Calcula la distancia de traza entre dos estados puros"""
        return np.sqrt(1 - self.fidelity(other))
    
    def copy(self) -> 'QuantumState':
        """Crea una copia profunda del estado"""
        return QuantumState(self.amplitudes.copy(), self.basis_states.copy())
    
    def __str__(self) -> str:
        terms = []
        for i, (amp, basis) in enumerate(zip(self.amplitudes, self.basis_states)):
            if abs(amp) > 1e-10:  # Evitar mostrar términos muy pequeños
                if np.isreal(amp):
                    if amp.real >= 0 and len(terms) > 0:
                        terms.append(f"+ {amp.real:.4f}|{basis}⟩")
                    else:
                        terms.append(f"{amp.real:.4f}|{basis}⟩")
                else:
                    if len(terms) > 0:
                        terms.append(f"+ ({amp:.4f})|{basis}⟩")
                    else:
                        terms.append(f"({amp:.4f})|{basis}⟩")
        return " ".join(terms) if terms else "0"
    
    def __repr__(self) -> str:
        return f"QuantumState({self.n_qubits} qubits, {len(self.amplitudes)} estados)"

class QuantumGates:
    """Clase para generar matrices de compuertas cuánticas"""
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Compuerta Pauli-X (NOT)"""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Compuerta Pauli-Y"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Compuerta Pauli-Z"""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Compuerta Hadamard"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def phase_gate(theta: float) -> np.ndarray:
        """Compuerta de fase arbitraria"""
        return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """Rotación alrededor del eje X"""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -1j * sin_half],
                         [-1j * sin_half, cos_half]], dtype=complex)
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """Rotación alrededor del eje Y"""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -sin_half],
                         [sin_half, cos_half]], dtype=complex)
    
    @staticmethod
    def rotation_z(theta: float) -> np.ndarray:
        """Rotación alrededor del eje Z"""
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)
    
    @staticmethod
    def cnot() -> np.ndarray:
        """Compuerta CNOT (Controlled-NOT)"""
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
    
    @staticmethod
    def controlled_gate(gate: np.ndarray) -> np.ndarray:
        """Crea una versión controlada de cualquier compuerta de 1 qubit"""
        if gate.shape != (2, 2):
            raise ValueError("La compuerta debe ser de 2x2 (1 qubit)")
        
        controlled = np.eye(4, dtype=complex)
        controlled[2:4, 2:4] = gate
        return controlled
    """Clase para manejar y crear estados cuánticos comunes"""
    
    @staticmethod
    def create_basis_states(n_qubits: int) -> Tuple[np.ndarray, List[str]]:
        """
        Crea todos los estados base para n qubits
        
        Args:
            n_qubits: Número de qubits
            
        Returns:
            Tupla con array de estados y lista de etiquetas
        """
        n_states = 2**n_qubits
        states = np.zeros((n_states, n_qubits), dtype=int)
        labels = []
        
        for i in range(n_states):
            # Convertir a binario y rellenar con ceros
            binary = format(i, f'0{n_qubits}b')
            states[i] = [int(bit) for bit in binary]
            labels.append(binary)
        
        return states, labels
    
    @staticmethod
    def create_superposition(n_qubits: int, equal_weights: bool = True) -> QuantumState:
        """
        Crea un estado de superposición uniforme
        
        Args:
            n_qubits: Número de qubits
            equal_weights: Si True, todos los estados tienen igual probabilidad
            
        Returns:
            Estado cuántico en superposición
        """
        states, labels = QuantumStateManager.create_basis_states(n_qubits)
        n_states = len(labels)
        
        if equal_weights:
            amplitudes = np.ones(n_states) / np.sqrt(n_states)
        else:
            # Amplitudes aleatorias normalizadas
            amplitudes = np.random.random(n_states) + 1j * np.random.random(n_states)
            amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
    @staticmethod
    def create_ghz_state(n_qubits: int) -> QuantumState:
        """
        Crea un estado GHZ (Greenberger-Horne-Zeilinger)
        
        Args:
            n_qubits: Número de qubits
            
        Returns:
            Estado GHZ |00...0⟩ + |11...1⟩
        """
        n_states = 2**n_qubits
        amplitudes = np.zeros(n_states, dtype=complex)
        
        # Estado |00...0⟩
        amplitudes[0] = 1/np.sqrt(2)
        # Estado |11...1⟩
        amplitudes[-1] = 1/np.sqrt(2)
        
        _, labels = QuantumStateManager.create_basis_states(n_qubits)
        return QuantumState(amplitudes, labels)
    
    @staticmethod
    def create_w_state(n_qubits: int) -> QuantumState:
        """
        Crea un estado W simétrico
        
        Args:
            n_qubits: Número de qubits
            
        Returns:
            Estado W con un solo 1 en cada término
        """
        if n_qubits < 2:
            raise ValueError("El estado W requiere al menos 2 qubits")
        
        n_states = 2**n_qubits
        amplitudes = np.zeros(n_states, dtype=complex)
        
        # Añadir amplitud a estados con exactamente un 1
        for i in range(n_qubits):
            state_index = 2**i
            amplitudes[state_index] = 1/np.sqrt(n_qubits)
        
        _, labels = QuantumStateManager.create_basis_states(n_qubits)
        return QuantumState(amplitudes, labels)
    
    @staticmethod
    def create_computational_state(binary_string: str) -> QuantumState:
        """
        Crea un estado computacional específico
        
        Args:
            binary_string: String binario como "101"
            
        Returns:
            Estado computacional correspondiente
        """
        n_qubits = len(binary_string)
        if not all(bit in '01' for bit in binary_string):
            raise ValueError("El string debe contener solo '0' y '1'")
        
        n_states = 2**n_qubits
        amplitudes = np.zeros(n_states, dtype=complex)
        
        # Convertir string binario a índice
        state_index = int(binary_string, 2)
        amplitudes[state_index] = 1.0
        
        _, labels = QuantumStateManager.create_basis_states(n_qubits)
        return QuantumState(amplitudes, labels)
    
    @staticmethod
    def create_bell_state(state_type: str = "phi_plus") -> QuantumState:
        """
        Crea estados de Bell
        
        Args:
            state_type: Tipo de estado Bell ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
            
        Returns:
            Estado de Bell correspondiente
        """
        sqrt2_inv = 1/np.sqrt(2)
        
        bell_states = {
            "phi_plus": (np.array([sqrt2_inv, 0, 0, sqrt2_inv]), ["00", "01", "10", "11"]),
            "phi_minus": (np.array([sqrt2_inv, 0, 0, -sqrt2_inv]), ["00", "01", "10", "11"]),
            "psi_plus": (np.array([0, sqrt2_inv, sqrt2_inv, 0]), ["00", "01", "10", "11"]),
            "psi_minus": (np.array([0, sqrt2_inv, -sqrt2_inv, 0]), ["00", "01", "10", "11"])
        }
        
        if state_type not in bell_states:
            raise ValueError(f"Tipo de estado Bell no válido: {state_type}")
        
        amplitudes, labels = bell_states[state_type]
        return QuantumState(amplitudes, labels)

class QuantumOperations:
    """Clase para operaciones con estados cuánticos"""
    
    @staticmethod
    def apply_single_qubit_gate(state: QuantumState, gate: np.ndarray, 
                               qubit_index: int) -> QuantumState:
        """
        Aplica una compuerta de un solo qubit
        
        Args:
            state: Estado cuántico original
            gate: Matriz de la compuerta (2x2)
            qubit_index: Índice del qubit (0 = más significativo)
            
        Returns:
            Nuevo estado después de aplicar la compuerta
        """
        if gate.shape != (2, 2):
            raise ValueError("La compuerta debe ser de 2x2")
        
        if not 0 <= qubit_index < state.n_qubits:
            raise ValueError(f"Índice de qubit inválido: {qubit_index}")
        
        # Crear compuerta completa usando producto tensorial
        gates = []
        for i in range(state.n_qubits):
            if i == qubit_index:
                gates.append(gate)
            else:
                gates.append(np.eye(2, dtype=complex))
        
        # Producto tensorial de todas las compuertas
        full_gate = gates[0]
        for g in gates[1:]:
            full_gate = np.kron(full_gate, g)
        
        new_amplitudes = full_gate @ state.amplitudes
        return QuantumState(new_amplitudes, state.basis_states.copy())
    
    @staticmethod
    def apply_two_qubit_gate(state: QuantumState, gate: np.ndarray,
                            control_qubit: int, target_qubit: int) -> QuantumState:
        """
        Aplica una compuerta de dos qubits
        
        Args:
            state: Estado cuántico original
            gate: Matriz de la compuerta (4x4)
            control_qubit: Índice del qubit de control
            target_qubit: Índice del qubit objetivo
            
        Returns:
            Nuevo estado después de aplicar la compuerta
        """
        if gate.shape != (4, 4):
            raise ValueError("La compuerta debe ser de 4x4")
        
        if state.n_qubits < 2:
            raise ValueError("Se necesitan al menos 2 qubits")
        
        if control_qubit == target_qubit:
            raise ValueError("Los qubits de control y objetivo deben ser diferentes")
        
        # Para simplificar, implementamos solo para sistemas de 2 qubits
        # Una implementación completa requeriría manejo más sofisticado
        if state.n_qubits == 2:
            new_amplitudes = gate @ state.amplitudes
            return QuantumState(new_amplitudes, state.basis_states.copy())
        else:
            raise NotImplementedError("Compuertas de 2 qubits solo implementadas para sistemas de 2 qubits")
    
    @staticmethod
    def tensor_product(state1: QuantumState, state2: QuantumState) -> QuantumState:
        """
        Calcula el producto tensorial de dos estados
        
        Args:
            state1: Primer estado cuántico
            state2: Segundo estado cuántico
            
        Returns:
            Estado producto tensorial
        """
        new_amplitudes = np.kron(state1.amplitudes, state2.amplitudes)
        
        # Crear nuevas etiquetas combinando las existentes
        new_labels = []
        for label1 in state1.basis_states:
            for label2 in state2.basis_states:
                new_labels.append(label1 + label2)
        
        return QuantumState(new_amplitudes, new_labels)
    
    @staticmethod
    def partial_trace(state: QuantumState, trace_qubits: List[int]) -> np.ndarray:
        """
        Calcula la traza parcial sobre los qubits especificados
        
        Args:
            state: Estado cuántico
            trace_qubits: Lista de índices de qubits a trazar
            
        Returns:
            Matriz de densidad reducida
        """
        # Crear matriz de densidad del estado puro
        rho = np.outer(state.amplitudes, np.conj(state.amplitudes))
        
        # Para simplicidad, implementamos solo casos básicos
        if len(trace_qubits) == 1 and state.n_qubits == 2:
            if trace_qubits[0] == 0:
                # Trazar el primer qubit
                reduced = np.array([[rho[0,0] + rho[1,1], rho[0,2] + rho[1,3]],
                                   [rho[2,0] + rho[3,1], rho[2,2] + rho[3,3]]])
            else:
                # Trazar el segundo qubit
                reduced = np.array([[rho[0,0] + rho[2,2], rho[0,1] + rho[2,3]],
                                   [rho[1,0] + rho[3,2], rho[1,1] + rho[3,3]]])
            return reduced
        else:
            raise NotImplementedError("Traza parcial solo implementada para casos básicos")
    
    @staticmethod
    def measure_state(state: QuantumState, measurement_basis: Optional[List[np.ndarray]] = None) -> Tuple[int, QuantumState]:
        """
        Simula una medición del estado cuántico
        
        Args:
            state: Estado a medir
            measurement_basis: Base de medición (por defecto: base computacional)
            
        Returns:
            Tupla con (resultado_medición, estado_post_medición)
        """
        if measurement_basis is None:
            # Medición en base computacional
            probabilities = [state.probability(i) for i in range(len(state.amplitudes))]
            
            # Elegir resultado basado en probabilidades
            result = np.random.choice(len(probabilities), p=probabilities)
            
            # Estado post-medición (colapso)
            new_amplitudes = np.zeros_like(state.amplitudes)
            new_amplitudes[result] = 1.0
            
            return result, QuantumState(new_amplitudes, state.basis_states.copy())
        else:
            raise NotImplementedError("Medición en bases arbitrarias no implementada")
    
    @staticmethod
    def apply_phase(state: QuantumState, phase: float, state_index: int) -> QuantumState:
        """
        Aplica una fase a un estado específico
        
        Args:
            state: Estado cuántico original
            phase: Fase a aplicar (en radianes)
            state_index: Índice del estado al que aplicar la fase
            
        Returns:
            Nuevo estado con la fase aplicada
        """
        if not 0 <= state_index < len(state.amplitudes):
            raise IndexError("Índice de estado fuera de rango")
        
        new_amplitudes = state.amplitudes.copy()
        new_amplitudes[state_index] *= np.exp(1j * phase)
        
        return QuantumState(new_amplitudes, state.basis_states.copy())
    
    @staticmethod
    def measure_probability_distribution(state: QuantumState) -> Dict[str, float]:
        """
        Calcula la distribución de probabilidades para todos los estados
        
        Args:
            state: Estado cuántico
            
        Returns:
            Diccionario con probabilidades para cada estado base
        """
        return {
            basis: abs(amp)**2 
            for basis, amp in zip(state.basis_states, state.amplitudes)
        }

# Ejemplo de uso y demostración
def main():
    """Función principal de demostración extendida"""
    print("=== Sistema Cuántico Avanzado ===\n")
    
    # 1. Estados básicos
    print("1. Estados base para 2 qubits:")
    states, labels = QuantumStateManager.create_basis_states(2)
    for state, label in zip(states, labels):
        print(f"   |{label}⟩ = {state}")
    
    # 2. Superposición
    print("\n2. Estado de superposición uniforme:")
    superposition = QuantumStateManager.create_superposition(2)
    print(f"   {superposition}")
    print(f"   Representación: {repr(superposition)}")
    
    # 3. Estados de Bell
    print("\n3. Estados de Bell:")
    bell_types = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
    for bell_type in bell_types:
        bell = QuantumStateManager.create_bell_state(bell_type)
        print(f"   |{bell_type}⟩: {bell}")
    
    # 4. Estados GHZ y W
    print("\n4. Estados multipartitos:")
    ghz = QuantumStateManager.create_ghz_state(3)
    print(f"   GHZ(3): {ghz}")
    
    w_state = QuantumStateManager.create_w_state(3)
    print(f"   W(3): {w_state}")
    
    # 5. Análisis de entrelazamiento
    print("\n5. Análisis de entrelazamiento:")
    bell_phi = QuantumStateManager.create_bell_state("phi_plus")
    product_state = QuantumOperations.tensor_product(
        QuantumStateManager.create_computational_state("0"),
        QuantumStateManager.create_computational_state("0")
    )
    
    print(f"   Bell |Φ⁺⟩ entrelazado: {QuantumEntanglement.is_entangled(bell_phi)}")
    print(f"   Estado |00⟩ entrelazado: {QuantumEntanglement.is_entangled(product_state)}")
    
    try:
        entropy_bell = QuantumEntanglement.entanglement_entropy(bell_phi, [0])
        entropy_product = QuantumEntanglement.entanglement_entropy(product_state, [0])
        print(f"   Entropía Bell: {entropy_bell:.3f}")
        print(f"   Entropía producto: {entropy_product:.3f}")
    except:
        print("   (Cálculo de entropía no disponible)")
    
    # 6. Operaciones con compuertas
    print("\n6. Aplicación de compuertas:")
    psi = QuantumStateManager.create_computational_state("0")
    print(f"   Estado inicial: {psi}")
    
    # Aplicar Hadamard
    h_gate = QuantumGates.hadamard()
    psi_h = QuantumOperations.apply_single_qubit_gate(psi, h_gate, 0)
    print(f"   Después de H: {psi_h}")
    
    # Aplicar rotación
    ry_gate = QuantumGates.rotation_y(np.pi/3)
    psi_ry = QuantumOperations.apply_single_qubit_gate(psi, ry_gate, 0)
    print(f"   Después de RY(π/3): {psi_ry}")
    
    # 7. Circuito cuántico
    print("\n7. Circuito cuántico:")
    circuit = QuantumCircuit(2)
    circuit.add_gate("H", 0).add_cnot(0, 1)
    print(circuit)
    
    result = circuit.execute()
    print(f"   Estado final: {result}")
    
    measurement = circuit.measure()
    print(f"   Medición: {measurement['measurement_result']} "
          f"(P = {measurement['probability']:.3f})")
    
    # 8. Métricas de estado
    print("\n8. Métricas entre estados:")
    state1 = QuantumStateManager.create_bell_state("phi_plus")
    state2 = QuantumStateManager.create_bell_state("psi_plus")
    
    fidelity = state1.fidelity(state2)
    trace_dist = state1.trace_distance(state2)
    
    print(f"   Fidelidad entre |Φ⁺⟩ y |Ψ⁺⟩: {fidelity:.3f}")
    print(f"   Distancia de traza: {trace_dist:.3f}")
    
    # 9. Visualización (si matplotlib está disponible)
    print("\n9. Visualización:")
    try:
        single_qubit = QuantumStateManager.create_computational_state("0")
        h_applied = QuantumOperations.apply_single_qubit_gate(single_qubit, h_gate, 0)
        
        print("   Generando visualizaciones...")
        QuantumVisualizer.plot_bloch_sphere_2d(h_applied)
        QuantumVisualizer.plot_probability_distribution(bell_phi, "Estado de Bell |Φ⁺⟩")
        
    except ImportError:
        print("   (Matplotlib no disponible para visualizaciones)")
    except Exception as e:
        print(f"   (Error en visualización: {e})")
    
    # 10. Ejemplo avanzado: Algoritmo de Deutsch-Jozsa simplificado
    print("\n10. Ejemplo: Circuito Deutsch-Jozsa (1 qubit):")
    
    # Función constante f(x) = 0
    dj_circuit = QuantumCircuit(1)
    dj_circuit.add_gate("H", 0)  # Superposición
    # No aplicamos nada (función constante 0)
    dj_circuit.add_gate("H", 0)  # Interferencia
    
    dj_result = dj_circuit.execute()
    print(f"   Estado final (f constante): {dj_result}")
    
    # La probabilidad de medir |0⟩ debería ser 1 para función constante
    prob_zero = dj_result.probability(0)
    print(f"   P(|0⟩) = {prob_zero:.3f} → función {'constante' if prob_zero > 0.9 else 'balanceada'}")

if __name__ == "__main__":
    main()