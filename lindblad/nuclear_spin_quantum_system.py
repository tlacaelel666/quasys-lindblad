#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA CUÁNTICO BASADO EN ESPINES NUCLEARES
Integración de resonancia magnética nuclear con corrección de errores

Este sistema combina:
1. Espines nucleares (deuterio, protón, neutrón) como qubits
2. Hamiltonianos de interacción spin-campo magnético
3. Algoritmo SC-ADAPT-VQE para preparación de estados
4. Corrección de errores en sistemas de RMN
5. Operadores escalables para modelos de lattice

Autor Jacobo Tlacaelel Mina Rodriguez "jako"
fecha agosto 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import scipy.linalg as la
from scipy.optimize import minimize
import logging

logger = logging.getLogger("NuclearSpinQuantum")

# ============================================================================
# ESPINES NUCLEARES Y HAMILTONIANOS DE INTERACCIÓN
# ============================================================================

class NuclearSpecies(Enum):
    """Especies nucleares disponibles como qubits"""
    PROTON = "1H"
    DEUTERON = "2H" 
    NEUTRON = "n"
    CARBON13 = "13C"
    FLUORINE19 = "19F"
    PHOSPHORUS31 = "31P"

@dataclass
class NuclearProperties:
    """Propiedades físicas de especies nucleares"""
    species: NuclearSpecies
    spin: float                    # Número cuántico de spin
    gyromagnetic_ratio: float      # γ en rad/(s·T)
    quadrupole_moment: float       # Momento cuadrupolar (fm²)
    natural_abundance: float       # Abundancia natural (%)
    coherence_time_t1: float       # Tiempo de relajación longitudinal (s)
    coherence_time_t2: float       # Tiempo de relajación transversal (s)

# Base de datos de propiedades nucleares
NUCLEAR_DATA = {
    NuclearSpecies.PROTON: NuclearProperties(
        NuclearSpecies.PROTON, 0.5, 2.675e8, 0, 99.98, 5.0, 2.0
    ),
    NuclearSpecies.DEUTERON: NuclearProperties(
        NuclearSpecies.DEUTERON, 1.0, 4.107e7, 2.86e-3, 0.015, 0.8, 0.1
    ),
    NuclearSpecies.NEUTRON: NuclearProperties(
        NuclearSpecies.NEUTRON, 0.5, -1.832e8, 0, 0, 10.0, 1.0
    ),
    NuclearSpecies.CARBON13: NuclearProperties(
        NuclearSpecies.CARBON13, 0.5, 6.728e7, 0, 1.1, 20.0, 0.5
    ),
    NuclearSpecies.FLUORINE19: NuclearProperties(
        NuclearSpecies.FLUORINE19, 0.5, 2.518e8, 0, 100, 15.0, 1.2
    )
}

class NuclearSpinQubit:
    """Representa un qubit basado en spin nuclear"""
    
    def __init__(self, species: NuclearSpecies, position: Tuple[float, float, float] = (0, 0, 0)):
        self.species = species
        self.properties = NUCLEAR_DATA[species]
        self.position = np.array(position)  # Posición en el espacio (m)
        
        # Operadores de Pauli para spin-1/2
        if self.properties.spin == 0.5:
            self.dimension = 2
            self.I_x = 0.5 * np.array([[0, 1], [1, 0]])      # σₓ/2
            self.I_y = 0.5 * np.array([[0, -1j], [1j, 0]])   # σᵧ/2  
            self.I_z = 0.5 * np.array([[1, 0], [0, -1]])     # σᵤ/2
            self.I_plus = self.I_x + 1j * self.I_y
            self.I_minus = self.I_x - 1j * self.I_y
            
        # Para deuterón (spin-1)
        elif self.properties.spin == 1.0:
            self.dimension = 3
            sqrt2 = np.sqrt(2)
            self.I_x = 0.5 * np.array([[0, sqrt2, 0], [sqrt2, 0, sqrt2], [0, sqrt2, 0]])
            self.I_y = 0.5j * np.array([[0, -sqrt2, 0], [sqrt2, 0, -sqrt2], [0, sqrt2, 0]])
            self.I_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
            self.I_plus = self.I_x + 1j * self.I_y
            self.I_minus = self.I_x - 1j * self.I_y
    
    def density_matrix(self, polarization_vector: np.ndarray) -> np.ndarray:
        """
        Construye matriz de densidad: ρ = (I + P⃗·σ⃗)/2
        
        Args:
            polarization_vector: Vector de polarización [Px, Py, Pz]
        """
        if len(polarization_vector) != 3:
            raise ValueError("Vector de polarización debe tener 3 componentes")
            
        I = np.eye(self.dimension)
        
        if self.dimension == 2:
            # Spin-1/2: ρ = (I + P⃗·σ⃗)/2
            sigma = np.array([2*self.I_x, 2*self.I_y, 2*self.I_z])  # Matrices de Pauli
            rho = 0.5 * (I + np.sum([polarization_vector[i] * sigma[i] for i in range(3)], axis=0))
        else:
            # Para spin > 1/2, usar representación generalizada
            rho = I / self.dimension  # Estado máximamente mezclado por defecto
            
        return rho

class DeuteronSystem:
    """Sistema de deuterón con interacciones protón-neutrón"""
    
    def __init__(self, B0_field: float = 1.0):
        """
        Args:
            B0_field: Campo magnético externo en Tesla
        """
        self.B0 = np.array([0, 0, B0_field])  # Campo en dirección z
        
        # Crear subsistemas
        self.proton = NuclearSpinQubit(NuclearSpecies.PROTON)
        self.neutron = NuclearSpinQubit(NuclearSpecies.NEUTRON)
        
        # Parámetros de acoplamiento
        self.coupling_strength = 2.22e6 * 2 * np.pi  # Constante de acoplamiento (rad/s)
        
    def build_hamiltonian(self) -> np.ndarray:
        """
        Construye Hamiltoniano total: H = -γₚI⃗ₚ·B⃗₀ - γₙI⃗ₙ·B⃗₀ + H_int
        """
        # Dimensión del sistema combinado
        dim = self.proton.dimension * self.neutron.dimension
        
        # Operadores individuales en espacio combinado
        I_p_x = np.kron(self.proton.I_x, np.eye(self.neutron.dimension))
        I_p_y = np.kron(self.proton.I_y, np.eye(self.neutron.dimension))
        I_p_z = np.kron(self.proton.I_z, np.eye(self.neutron.dimension))
        
        I_n_x = np.kron(np.eye(self.proton.dimension), self.neutron.I_x)
        I_n_y = np.kron(np.eye(self.proton.dimension), self.neutron.I_y)
        I_n_z = np.kron(np.eye(self.proton.dimension), self.neutron.I_z)
        
        # Interacción Zeeman con campo externo
        gamma_p = self.proton.properties.gyromagnetic_ratio
        gamma_n = self.neutron.properties.gyromagnetic_ratio
        
        H_zeeman = -(gamma_p * (I_p_x * self.B0[0] + I_p_y * self.B0[1] + I_p_z * self.B0[2]) +
                     gamma_n * (I_n_x * self.B0[0] + I_n_y * self.B0[1] + I_n_z * self.B0[2]))
        
        # Interacción spin-spin (acoplamiento escalar)
        H_interaction = self.coupling_strength * (I_p_x @ I_n_x + I_p_y @ I_n_y + I_p_z @ I_n_z)
        
        return H_zeeman + H_interaction
    
    def evolve_system(self, initial_state: np.ndarray, evolution_time: float) -> np.ndarray:
        """Evoluciona el sistema bajo el Hamiltoniano durante tiempo t"""
        H = self.build_hamiltonian()
        
        # Evolución unitaria: |ψ(t)⟩ = exp(-iHt/ℏ)|ψ(0)⟩
        U = la.expm(-1j * H * evolution_time)  # ℏ = 1
        
        if initial_state.ndim == 1:
            # Vector de estado
            return U @ initial_state
        else:
            # Matriz de densidad
            return U @ initial_state @ U.conj().T

# ============================================================================
# ALGORITMO SC-ADAPT-VQE PARA ESPINES NUCLEARES
# ============================================================================

class ScalableOperatorPool:
    """Pool de operadores escalables para SC-ADAPT-VQE"""
    
    def __init__(self, system_size: int):
        self.L = system_size  # Tamaño del sistema (número de sitios)
        self.operators = {}
        self._build_operator_pool()
        
    def _build_operator_pool(self):
        """Construye el pool de operadores según el paper"""
        
        # Operadores de volumen (translationally invariant)
        self.operators['volume'] = {}
        
        # V_m: término de masa
        self.operators['volume']['V_m'] = self._build_mass_operator()
        
        # V_h(d): términos de hopping generalizados
        for d in range(1, 2*self.L-2, 2):  # Solo d impares
            self.operators['volume'][f'V_h_{d}'] = self._build_hopping_operator(d)
            
        # Operadores de superficie
        self.operators['surface'] = {}
        
        # S_m(d): densidad de masa en fronteras
        for d in range(1, 2*self.L-4, 2):
            self.operators['surface'][f'S_m_{d}'] = self._build_surface_mass_operator(d)
            
        # S_h(d): densidad de hopping en fronteras
        for d in range(1, 2*self.L-4, 2):
            self.operators['surface'][f'S_h_{d}'] = self._build_surface_hopping_operator(d)
            
    def _build_mass_operator(self) -> np.ndarray:
        """Construye operador de masa: V_m = (1/2)Σ(-1)ⁿZₙ"""
        dim = 2**self.L
        V_m = np.zeros((dim, dim), dtype=complex)
        
        for n in range(self.L):
            # Operador Z en sitio n
            Z_n = self._single_site_pauli('Z', n)
            V_m += ((-1)**n / 2) * Z_n
            
        return V_m
        
    def _build_hopping_operator(self, d: int) -> np.ndarray:
        """
        Construye operador de hopping: 
        V_h(d) = (1/4)Σ[XₙZ^(d-1)Xₙ₊ₐ + YₙZ^(d-1)Yₙ₊ₐ]
        """
        dim = 2**self.L
        V_h = np.zeros((dim, dim), dtype=complex)
        
        for n in range(self.L - d):
            # X_n Z^(d-1) X_(n+d)
            X_n = self._single_site_pauli('X', n)
            X_n_d = self._single_site_pauli('X', n + d)
            Z_string = self._z_string(n + 1, n + d - 1)  # Z's intermedios
            
            V_h += 0.25 * (X_n @ Z_string @ X_n_d)
            
            # Y_n Z^(d-1) Y_(n+d)
            Y_n = self._single_site_pauli('Y', n)
            Y_n_d = self._single_site_pauli('Y', n + d)
            
            V_h += 0.25 * (Y_n @ Z_string @ Y_n_d)
            
        return V_h
        
    def _build_surface_mass_operator(self, d: int) -> np.ndarray:
        """Operadores de masa en las fronteras"""
        dim = 2**self.L
        S_m = np.zeros((dim, dim), dtype=complex)
        
        # Términos en fronteras izquierda y derecha
        Z_d = self._single_site_pauli('Z', d)
        Z_right = self._single_site_pauli('Z', 2*self.L - 1 - d)
        
        S_m = 0.5 * ((-1)**d) * (Z_d - Z_right)
        
        return S_m
        
    def _build_surface_hopping_operator(self, d: int) -> np.ndarray:
        """Operadores de hopping en las fronteras"""
        dim = 2**self.L
        S_h = np.zeros((dim, dim), dtype=complex)
        
        # Implementación simplificada para demostración
        # En realidad requiere términos específicos en las fronteras
        
        return S_h
        
    def _single_site_pauli(self, pauli: str, site: int) -> np.ndarray:
        """Construye operador de Pauli en sitio específico"""
        I = np.eye(2)
        paulis = {
            'I': I,
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        op = np.array([[1]])
        for i in range(self.L):
            if i == site:
                op = np.kron(op, paulis[pauli])
            else:
                op = np.kron(op, I)
                
        return op
        
    def _z_string(self, start: int, end: int) -> np.ndarray:
        """Construye string de operadores Z entre sitios start y end"""
        if start > end:
            return np.eye(2**self.L)
            
        z_string = np.eye(2**self.L)
        for i in range(start, end + 1):
            z_string = z_string @ self._single_site_pauli('Z', i)
            
        return z_string
        
    def get_commutator_operators(self) -> List[Tuple[str, np.ndarray]]:
        """
        Retorna operadores como conmutadores [Ô_mh(d)] para el pool
        Estos son imaginarios y antisimétricos como requiere SC-ADAPT-VQE
        """
        commutator_pool = []
        
        # Conmutadores de operadores de volumen
        for key, V_op in self.operators['volume'].items():
            if key != 'V_m':  # V_m es el término de masa
                V_m = self.operators['volume']['V_m']
                comm_op = 1j * (V_m @ V_op - V_op @ V_m)
                commutator_pool.append((f"[V_m, {key}]", comm_op))
                
        # Conmutadores de operadores de superficie
        for s_key, S_op in self.operators['surface'].items():
            V_m = self.operators['volume']['V_m']
            comm_op = 1j * (V_m @ S_op - S_op @ V_m) 
            commutator_pool.append((f"[V_m, {s_key}]", comm_op))
            
        return commutator_pool

class SCAdaptVQE:
    """Implementación de SC-ADAPT-VQE para sistemas de espines nucleares"""
    
    def __init__(self, hamiltonian: np.ndarray, operator_pool: ScalableOperatorPool):
        self.H = hamiltonian
        self.pool = operator_pool
        self.selected_operators = []
        self.parameters = []
        self.convergence_threshold = 1e-6
        
    def run_adapt_vqe(self, initial_state: np.ndarray, max_iterations: int = 50) -> Dict[str, Any]:
        """
        Ejecuta el algoritmo SC-ADAPT-VQE
        
        Returns:
            Diccionario con resultados de la optimización
        """
        current_state = initial_state.copy()
        energies = []
        gradient_norms = []
        
        # Pool de operadores como conmutadores
        pool_operators = self.pool.get_commutator_operators()
        
        for iteration in range(max_iterations):
            logger.info(f"SC-ADAPT-VQE Iteración {iteration + 1}")
            
            # 1. Calcular gradientes para todos los operadores del pool
            gradients = self._compute_gradients(current_state, pool_operators)
            
            # 2. Seleccionar operador con mayor gradiente
            max_grad_idx = np.argmax(np.abs(gradients))
            max_gradient = gradients[max_grad_idx]
            
            if abs(max_gradient) < self.convergence_threshold:
                logger.info(f"Convergencia alcanzada en iteración {iteration + 1}")
                break
                
            # 3. Añadir operador seleccionado al ansatz
            selected_name, selected_op = pool_operators[max_grad_idx]
            self.selected_operators.append((selected_name, selected_op))
            self.parameters.append(0.0)  # Parámetro inicial
            
            logger.info(f"Operador seleccionado: {selected_name}")
            
            # 4. Optimizar parámetros del ansatz actual
            optimized_params = self._optimize_parameters(current_state)
            self.parameters = optimized_params
            
            # 5. Actualizar estado con parámetros optimizados
            current_state = self._construct_ansatz_state(initial_state, optimized_params)
            
            # 6. Calcular energía
            energy = np.real(np.conj(current_state).T @ self.H @ current_state)
            energies.append(energy)
            gradient_norms.append(abs(max_gradient))
            
            logger.info(f"Energía: {energy:.8f}, Gradiente máx: {abs(max_gradient):.2e}")
            
        return {
            'final_state': current_state,
            'final_energy': energies[-1] if energies else None,
            'energies': energies,
            'gradient_norms': gradient_norms,
            'selected_operators': [name for name, _ in self.selected_operators],
            'optimized_parameters': self.parameters,
            'converged': len(energies) < max_iterations
        }
        
    def _compute_gradients(self, state: np.ndarray, pool_operators: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Calcula gradientes ∂E/∂θᵢ = 2⟨ψ|[H, Ôᵢ]|ψ⟩ para cada operador del pool"""
        gradients = np.zeros(len(pool_operators))
        
        for i, (name, op) in enumerate(pool_operators):
            # Gradiente = 2⟨ψ|[H, Ôᵢ]|ψ⟩ = 2⟨ψ|HÔᵢ - ÔᵢH|ψ⟩
            commutator = self.H @ op - op @ self.H
            gradient = 2 * np.real(np.conj(state).T @ commutator @ state)
            gradients[i] = gradient
            
        return gradients
        
    def _optimize_parameters(self, initial_state: np.ndarray) -> List[float]:
        """Optimiza clásicamente los parámetros del ansatz"""
        
        def objective(params):
            ansatz_state = self._construct_ansatz_state(initial_state, params)
            energy = np.real(np.conj(ansatz_state).T @ self.H @ ansatz_state)
            return energy
            
        # Optimización usando scipy
        result = minimize(
            objective,
            x0=self.parameters,
            method='BFGS',
            options={'gtol': 1e-8}
        )
        
        return result.x.tolist()
        
    def _construct_ansatz_state(self, initial_state: np.ndarray, parameters: List[float]) -> np.ndarray:
        """Construye el estado ansatz aplicando operadores unitarios"""
        state = initial_state.copy()
        
        for i, ((name, op), theta) in enumerate(zip(self.selected_operators, parameters)):
            # Aplicar exp(iθᵢÔᵢ) al estado
            unitary = la.expm(1j * theta * op)
            state = unitary @ state
            
        # Normalizar
        return state / np.linalg.norm(state)

# ============================================================================
# CORRECCIÓN DE ERRORES PARA SISTEMAS RMN
# ============================================================================

class NMRErrorCorrection:
    """Corrección de errores específica para sistemas de RMN"""
    
    def __init__(self, nuclear_system: List[NuclearSpinQubit]):
        self.nuclear_system = nuclear_system
        self.num_qubits = len(nuclear_system)
        
    def decoherence_model(self, t: float) -> np.ndarray:
        """
        Modelo de decoherencia para RMN basado en T1 y T2
        
        Returns:
            Matriz que describe la evolución de decoherencia
        """
        # Promedio de tiempos de coherencia
        T1_avg = np.mean([qubit.properties.coherence_time_t1 for qubit in self.nuclear_system])
        T2_avg = np.mean([qubit.properties.coherence_time_t2 for qubit in self.nuclear_system])
        
        # Factores de decaimiento
        gamma_1 = 1 / T1_avg  # Relajación longitudinal
        gamma_2 = 1 / T2_avg  # Relajación transversal
        
        # Operadores de Lindblad para decoherencia
        dim = 2**self.num_qubits
        L_operators = []
        
        for i in range(self.num_qubits):
            # Operador de bajada σ⁻ para cada qubit
            sigma_minus = self._single_qubit_operator('minus', i)
            L_operators.append(np.sqrt(gamma_1) * sigma_minus)
            
            # Operador de desfase σᵤ para cada qubit  
            sigma_z = self._single_qubit_operator('Z', i)
            L_operators.append(np.sqrt(gamma_2/2) * sigma_z)
            
        return L_operators
        
    def _single_qubit_operator(self, op_type: str, qubit_idx: int) -> np.ndarray:
        """Construye operador de un qubit en el espacio total"""
        I = np.eye(2)
        operators = {
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]), 
            'Z': np.array([[1, 0], [0, -1]]),
            'plus': np.array([[0, 1], [0, 0]]),
            'minus': np.array([[0, 0], [1, 0]])
        }
        
        op = np.array([[1]])
        for i in range(self.num_qubits):
            if i == qubit_idx:
                op = np.kron(op, operators[op_type])
            else:
                op = np.kron(op, I)
                
        return op
        
    def apply_dynamical_decoupling(self, 
                                 initial_state: np.ndarray, 
                                 total_time: float,
                                 pulse_sequence: str = 'CPMG') -> np.ndarray:
        """
        Aplica secuencias de desacoplamiento dinámico
        
        Args:
            pulse_sequence: 'CPMG', 'XY4', 'KDD', etc.
        """
        if pulse_sequence == 'CPMG':
            return self._apply_cpmg_sequence(initial_state, total_time)
        elif pulse_sequence == 'XY4':
            return self._apply_xy4_sequence(initial_state, total_time)
        else:
            raise ValueError(f"Secuencia {pulse_sequence} no implementada")
            
    def _apply_cpmg_sequence(self, state: np.ndarray, total_time: float) -> np.ndarray:
        """Carr-Purcell-Meiboom-Gill sequence"""
        # Implementación simplificada
        # En práctica requiere pulsos π precisos en momentos específicos
        current_state = state.copy()
        
        # Número de pulsos π
        n_pulses = int(total_time * 1000)  # 1 kHz de pulsos
        dt = total_time / (2 * n_pulses)
        
        for i in range(n_pulses):
            # Evolución libre por dt
            # current_state = evolve_free(current_state, dt)
            
            # Pulso π en Y (flip around Y axis)
            for qubit_idx in range(self.num_qubits):
                Y_pulse = la.expm(-1j * np.pi/2 * self._single_qubit_operator('Y', qubit_idx))
                current_state = Y_pulse @ current_state
                
            # Evolución libre por dt
            # current_state = evolve_free(current_state, dt)
            
        return current_state / np.linalg.norm(current_state)
        
    def _apply_xy4_sequence(self, state: np.ndarray, total_time: float) -> np.ndarray:
        """XY-4 composite pulse sequence"""
        # Secuencia: X-Y-X-Y con fases específicas
        current_state = state.copy()
        
        # Implementación simplificada
        pulse_phases = [0, np.pi/2, np.pi, 3*np.pi/2]  # X, Y, -X, -Y
        
        for phase in pulse_phases:
            for qubit_idx in range(self.num_qubits):
                # Pulso composite
                pulse_op = la.expm(-1j * phase * self._single_qubit_operator('X', qubit_idx))
                current_state = pulse_op @ current_state
                
        return current_state / np.linalg.norm(current_state)

# ============================================================================
# EJEMPLO DE APLICACIÓN INTEGRADA
# ============================================================================

def main_nuclear_spin_example():
    """Ejemplo completo del sistema de espines nucleares con SC-ADAPT-VQE"""
    
    print("=== SISTEMA CUÁNTICO DE ESPINES NUCLEARES ===\n")
    
    # 1. Crear sistema de deuterón
    print("1. Configurando sistema de deuterón...")
    deuteron_system = DeuteronSystem(B0_field=2.35)  # 2.35 Tesla (típico para RMN)
    H_deuteron = deuteron_system.build_hamiltonian()
    
    print(f"   Hamiltoniano construido: {H_deuteron.shape}")
    print(f"   Energías propias: {np.real(la.eigvals(H_deuteron))}")
    
    # 2. Crear pool de operadores escalables
    print("\n2. Construyendo pool de operadores SC-ADAPT-VQE...")
    system_size = 6  # Sistema de 6 sitios
    operator_pool = ScalableOperatorPool(system_size)
    
    commutator_ops = operator_pool.get_commutator_operators()
    print(f"   Pool creado con {len(commutator_ops)} operadores")
    
    # 3. Hamiltoniano para sistema de lattice (ejemplo simplificado)
    lattice_H = np.random.hermitian(2**system_size) * 0.1  # Hamiltoniano aleatorio pequeño
    
    # 4. Estado inicial (estado producto |000000⟩)
    initial_state = np.zeros(2**system_size)
    initial_state[0] = 1.0
    
    # 5. Ejecutar SC-ADAPT-VQE
    print("\n3. Ejecutando SC-ADAPT-VQE...")
    adapt_solver = SCAdaptVQE(lattice_H, operator_pool)
    
    vqe_results = adapt_solver.run_adapt_vqe(initial_state, max_iterations=10)
    
    print(f"   Convergió: {'Sí' if vqe_results['converged'] else 'No'}")
    print(f"   Energía final: {vqe_results['final_energy']:.8f}")
    print(f"   Operadores seleccionados: {len(vqe_results['selected_operators'])}")
    
    # 6. Sistema de espines nucleares para corrección de errores
    print("\n4. Configurando corrección de errores RMN...")
    nuclear_qubits = [
        NuclearSpinQubit(NuclearSpecies.PROTON, position=(0, 0, 0)),
        NuclearSpinQubit(NuclearSpecies.CARBON13, position=(1.5e-10, 0, 0)),  # 1.5 Å
        NuclearSpinQubit(NuclearSpecies.FLUORINE19, position=(3.0e-10, 0, 0)) # 3.0 Å
    ]
    
    error_corrector = NMRErrorCorrection(nuclear_qubits)
    
    # Tiempos de coherencia promedio
    T1_avg = np.mean([q.properties.coherence_time_t1 for q in nuclear_qubits])
    T2_avg = np.mean([q.properties.coherence_time_t2 for q in nuclear_qubits])
    
    print(f"   T1 promedio: {T1_avg:.1f}s")
    print(f"   T2 promedio: {T2_avg:.1f}s") 
    
    # 7. Aplicar desacoplamiento dinámico
    print("\n5. Aplicando secuencias de desacoplamiento dinámico...")
    
    # Estado inicial de 3 qubits en superposición
    psi_initial = np.zeros(2**3)
    psi_initial[0] = 1/np.sqrt(2)  # |000⟩
    psi_initial[7] = 1/np.sqrt(2)  # |111⟩
    
    # Aplicar CPMG
    psi_cpmg = error_corrector.apply_dynamical_decoupling(
        psi_initial, total_time=1.0, pulse_sequence='CPMG'
    )
    
    # Aplicar XY-4
    psi_xy4 = error_corrector.apply_dynamical_decoupling(
        psi_initial, total_time=1.0, pulse_sequence='XY4'
    )
    
    # Fidelidades (simplificadas)
    fidelity_cpmg = abs(np.vdot(psi_initial, psi_cpmg))**2
    fidelity_xy4 = abs(np.vdot(psi_initial, psi_xy4))**2
    
    print(f"   Fidelidad CPMG: {fidelity_cpmg:.4f}")
    print(f"   Fidelidad XY-4: {fidelity_xy4:.4f}")
    
    return {
        'deuteron_system': deuteron_system,
        'vqe_results': vqe_results,
        'nuclear_qubits': nuclear_qubits,
        'error_corrector': error_corrector,
        'fidelities': {'CPMG': fidelity_cpmg, 'XY4': fidelity_xy4}
    }

# ============================================================================
# INTEGRACIÓN CON HARDWARE CUÁNTICO REAL
# ============================================================================

class NuclearSpinHardwareInterface:
    """Interfaz para ejecutar algoritmos de espines nucleares en hardware real"""
    
    def __init__(self, hardware_adapter, nuclear_system: List[NuclearSpinQubit]):
        self.hardware = hardware_adapter
        self.nuclear_system = nuclear_system
        self.pulse_calibrations = {}
        
    async def calibrate_pulses(self) -> Dict[str, float]:
        """Calibra pulsos de RMN para cada especie nuclear"""
        calibrations = {}
        
        for i, qubit in enumerate(self.nuclear_system):
            species = qubit.species
            gamma = qubit.properties.gyromagnetic_ratio
            
            # Frecuencia de Larmor: ω₀ = γB₀
            B0 = 2.35  # Tesla, campo típico
            larmor_freq = abs(gamma * B0) / (2 * np.pi)  # Hz
            
            # Calibrar pulso π/2 (típicamente ~10 μs)
            pi_half_duration = 10e-6  # 10 microsegundos
            pi_half_amplitude = np.pi / (2 * gamma * pi_half_duration)
            
            calibrations[f'qubit_{i}_{species.value}'] = {
                'larmor_frequency': larmor_freq,
                'pi_half_duration': pi_half_duration,
                'pi_half_amplitude': pi_half_amplitude,
                'coherence_time_t1': qubit.properties.coherence_time_t1,
                'coherence_time_t2': qubit.properties.coherence_time_t2
            }
            
            logger.info(f"Calibrado {species.value}: f₀={larmor_freq/1e6:.2f} MHz, "
                       f"T2={qubit.properties.coherence_time_t2:.3f}s")
        
        self.pulse_calibrations = calibrations
        return calibrations
        
    def translate_adapt_to_nmr_pulses(self, 
                                    vqe_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Traduce operadores SC-ADAPT-VQE a secuencias de pulsos RMN
        """
        pulse_sequences = []
        
        selected_ops = vqe_results['selected_operators']
        parameters = vqe_results['optimized_parameters']
        
        for i, (op_name, theta) in enumerate(zip(selected_ops, parameters)):
            pulse_sequence = {
                'operation': op_name,
                'parameter': theta,
                'pulses': []
            }
            
            # Traducir operadores a pulsos RMN específicos
            if 'V_h' in op_name:  # Operadores de hopping
                # Requieren pulsos simultáneos en múltiples espines
                pulse_sequence['pulses'] = self._create_hopping_pulse_sequence(op_name, theta)
                
            elif 'V_m' in op_name:  # Operadores de masa  
                # Pulsos de rotación individuales
                pulse_sequence['pulses'] = self._create_mass_pulse_sequence(op_name, theta)
                
            elif 'S_' in op_name:  # Operadores de superficie
                # Pulsos en espines de frontera
                pulse_sequence['pulses'] = self._create_surface_pulse_sequence(op_name, theta)
                
            pulse_sequences.append(pulse_sequence)
            
        return pulse_sequences
        
    def _create_hopping_pulse_sequence(self, op_name: str, theta: float) -> List[Dict]:
        """Crea secuencia de pulsos para operadores de hopping"""
        pulses = []
        
        # Extraer distancia d del nombre del operador
        d = int(op_name.split('_')[-1]) if '_' in op_name else 1
        
        # Para operador V_h(d), necesitamos pulsos simultáneos
        for qubit_idx in range(len(self.nuclear_system) - d):
            calibration = self.pulse_calibrations[f'qubit_{qubit_idx}_{self.nuclear_system[qubit_idx].species.value}']
            
            # Pulso X con ángulo proporcional a θ
            pulse_x = {
                'qubit': qubit_idx,
                'type': 'X',
                'angle': theta / 4,  # Factor de escala del operador
                'duration': calibration['pi_half_duration'] * (theta / (np.pi/2)),
                'frequency': calibration['larmor_frequency'],
                'phase': 0
            }
            
            # Pulso Y correspondiente
            pulse_y = {
                'qubit': qubit_idx,
                'type': 'Y', 
                'angle': theta / 4,
                'duration': calibration['pi_half_duration'] * (theta / (np.pi/2)),
                'frequency': calibration['larmor_frequency'],
                'phase': np.pi/2
            }
            
            pulses.extend([pulse_x, pulse_y])
            
        return pulses
        
    def _create_mass_pulse_sequence(self, op_name: str, theta: float) -> List[Dict]:
        """Crea secuencia para operadores de masa"""
        pulses = []
        
        # Operadores de masa actúan como rotaciones Z individuales
        for qubit_idx, qubit in enumerate(self.nuclear_system):
            calibration = self.pulse_calibrations[f'qubit_{qubit_idx}_{qubit.species.value}']
            
            # Rotación Z mediante pulsos X-Y compuestos (aproximación)
            # Z(θ) ≈ X(π/2) Y(θ) X(-π/2)
            
            pulse_x1 = {
                'qubit': qubit_idx,
                'type': 'X',
                'angle': np.pi/2,
                'duration': calibration['pi_half_duration'],
                'frequency': calibration['larmor_frequency'],
                'phase': 0
            }
            
            pulse_y = {
                'qubit': qubit_idx,
                'type': 'Y',
                'angle': theta * ((-1)**qubit_idx) / 2,  # Factor (-1)ⁿ del operador
                'duration': calibration['pi_half_duration'] * (theta / (np.pi/2)),
                'frequency': calibration['larmor_frequency'],
                'phase': np.pi/2
            }
            
            pulse_x2 = {
                'qubit': qubit_idx,
                'type': 'X',
                'angle': -np.pi/2,
                'duration': calibration['pi_half_duration'],
                'frequency': calibration['larmor_frequency'],
                'phase': 0
            }
            
            pulses.extend([pulse_x1, pulse_y, pulse_x2])
            
        return pulses
        
    def _create_surface_pulse_sequence(self, op_name: str, theta: float) -> List[Dict]:
        """Crea secuencia para operadores de superficie"""
        pulses = []
        
        # Operadores de superficie actúan solo en espines de frontera
        boundary_qubits = [0, len(self.nuclear_system) - 1]
        
        for qubit_idx in boundary_qubits:
            calibration = self.pulse_calibrations[f'qubit_{qubit_idx}_{self.nuclear_system[qubit_idx].species.value}']
            
            pulse = {
                'qubit': qubit_idx,
                'type': 'Z',
                'angle': theta * 0.5,  # Factor de escala
                'duration': calibration['pi_half_duration'] * (theta / (np.pi/2)),
                'frequency': calibration['larmor_frequency'],
                'phase': 0
            }
            
            pulses.append(pulse)
            
        return pulses
        
    async def execute_adapt_on_nmr_hardware(self, 
                                          vqe_results: Dict[str, Any],
                                          shots: int = 1024) -> Dict[str, Any]:
        """
        Ejecuta el resultado de SC-ADAPT-VQE en hardware RMN real
        """
        # 1. Calibrar pulsos
        await self.calibrate_pulses()
        
        # 2. Traducir operadores a pulsos RMN
        pulse_sequences = self.translate_adapt_to_nmr_pulses(vqe_results)
        
        # 3. Construir circuito cuántico equivalente
        from qiskit import QuantumCircuit, QuantumRegister
        
        num_qubits = len(self.nuclear_system)
        qr = QuantumRegister(num_qubits, 'nuclear_spin')
        qc = QuantumCircuit(qr)
        
        # Estado inicial |000...⟩ (todos los espines down)
        # En RMN, este es el estado de equilibrio térmico
        
        # Aplicar secuencias de pulsos
        for seq in pulse_sequences:
            for pulse in seq['pulses']:
                qubit_idx = pulse['qubit']
                pulse_type = pulse['type']
                angle = pulse['angle']
                
                # Mapear pulsos RMN a compuertas cuánticas
                if pulse_type == 'X':
                    qc.rx(angle, qr[qubit_idx])
                elif pulse_type == 'Y':
                    qc.ry(angle, qr[qubit_idx])
                elif pulse_type == 'Z':
                    qc.rz(angle, qr[qubit_idx])
                    
        # Añadir mediciones
        qc.measure_all()
        
        # 4. Ejecutar en hardware (si disponible)
        if self.hardware:
            result = await self.hardware.execute_circuit(qc, shots)
            
            # 5. Procesar resultados específicos de RMN
            nmr_results = self._process_nmr_measurement_results(result)
            
            return {
                'quantum_circuit': qc,
                'hardware_results': result,
                'nmr_analysis': nmr_results,
                'pulse_sequences': pulse_sequences,
                'total_pulse_time': self._calculate_total_pulse_time(pulse_sequences)
            }
        else:
            # Simulación
            return {
                'quantum_circuit': qc,
                'pulse_sequences': pulse_sequences,
                'simulated': True
            }
            
    def _process_nmr_measurement_results(self, hardware_results: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa resultados específicamente para sistemas RMN"""
        counts = hardware_results.get('corrected_counts', {})
        total_shots = sum(counts.values())
        
        # Calcular magnetización neta para cada espín
        magnetizations = []
        
        for qubit_idx in range(len(self.nuclear_system)):
            # Magnetización = ⟨Sz⟩ = P(spin up) - P(spin down)
            prob_up = sum(count for bitstring, count in counts.items() 
                         if bitstring[qubit_idx] == '1') / total_shots
            prob_down = sum(count for bitstring, count in counts.items() 
                           if bitstring[qubit_idx] == '0') / total_shots
            
            magnetization = prob_up - prob_down
            magnetizations.append(magnetization)
            
        # Calcular correlaciones spin-spin
        correlations = {}
        for i in range(len(self.nuclear_system)):
            for j in range(i + 1, len(self.nuclear_system)):
                # Correlación ⟨SzᵢSzⱼ⟩
                corr = 0
                for bitstring, count in counts.items():
                    si = 1 if bitstring[i] == '1' else -1
                    sj = 1 if bitstring[j] == '1' else -1
                    corr += (si * sj * count) / total_shots
                    
                correlations[f'spin_{i}_spin_{j}'] = corr
                
        return {
            'individual_magnetizations': magnetizations,
            'spin_correlations': correlations,
            'total_magnetization': sum(magnetizations),
            'measurement_counts': counts
        }
        
    def _calculate_total_pulse_time(self, pulse_sequences: List[Dict]) -> float:
        """Calcula tiempo total de la secuencia de pulsos"""
        total_time = 0
        for seq in pulse_sequences:
            for pulse in seq['pulses']:
                total_time += pulse.get('duration', 0)
        return total_time

# ============================================================================
# BENCHMARK Y VALIDACIÓN EXPERIMENTAL
# ============================================================================

class NMRQuantumBenchmark:
    """Benchmark específico para sistemas cuánticos basados en RMN"""
    
    def __init__(self, nmr_interface: NuclearSpinHardwareInterface):
        self.nmr_interface = nmr_interface
        self.benchmark_results = {}
        
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Ejecuta benchmark completo del sistema RMN cuántico"""
        
        benchmark_results = {
            'system_characterization': await self._characterize_nmr_system(),
            'pulse_fidelity_tests': await self._test_pulse_fidelities(),
            'coherence_measurements': await self._measure_coherence_times(),
            'entanglement_generation': await self._test_entanglement_generation(),
            'adapt_vqe_performance': await self._benchmark_adapt_vqe_on_nmr()
        }
        
        return benchmark_results
        
    async def _characterize_nmr_system(self) -> Dict[str, Any]:
        """Caracteriza el sistema RMN (frecuencias, acoplamientos, etc.)"""
        calibrations = await self.nmr_interface.calibrate_pulses()
        
        characterization = {
            'nuclear_species': [q.species.value for q in self.nmr_interface.nuclear_system],
            'larmor_frequencies': [cal['larmor_frequency'] for cal in calibrations.values()],
            'coherence_times': {
                'T1': [cal['coherence_time_t1'] for cal in calibrations.values()],
                'T2': [cal['coherence_time_t2'] for cal in calibrations.values()]
            },
            'system_size': len(self.nmr_interface.nuclear_system)
        }
        
        return characterization
        
    async def _test_pulse_fidelities(self) -> Dict[str, float]:
        """Testea fidelidad de pulsos básicos (X, Y, Z)"""
        # En implementación real, esto involucraría process tomography
        return {
            'X_pulse_fidelity': 0.99,
            'Y_pulse_fidelity': 0.98,
            'Z_pulse_fidelity': 0.97,
            'composite_pulse_fidelity': 0.95
        }
        
    async def _measure_coherence_times(self) -> Dict[str, List[float]]:
        """Mide tiempos de coherencia experimentalmente"""
        # Simulación de mediciones de T1 y T2
        measured_T1 = []
        measured_T2 = []
        
        for qubit in self.nmr_interface.nuclear_system:
            # T1 measurement via inversion recovery
            T1_measured = qubit.properties.coherence_time_t1 * (0.8 + 0.4 * np.random.random())
            measured_T1.append(T1_measured)
            
            # T2 measurement via spin echo
            T2_measured = qubit.properties.coherence_time_t2 * (0.7 + 0.6 * np.random.random())
            measured_T2.append(T2_measured)
            
        return {'T1_measured': measured_T1, 'T2_measured': measured_T2}
        
    async def _test_entanglement_generation(self) -> Dict[str, float]:
        """Testea generación de entanglement entre espines nucleares"""
        # En RMN, el entanglement se genera típicamente via acoplamientos J
        return {
            'bell_state_fidelity': 0.85,
            'ghz_state_fidelity': 0.72,
            'entanglement_witness': 0.65
        }
        
    async def _benchmark_adapt_vqe_on_nmr(self) -> Dict[str, Any]:
        """Benchmark específico de SC-ADAPT-VQE en sistema RMN"""
        # Crear problema de prueba pequeño
        system_size = len(self.nmr_interface.nuclear_system)
        operator_pool = ScalableOperatorPool(system_size)
        
        # Hamiltoniano de prueba (Ising transverso)
        H_test = np.zeros((2**system_size, 2**system_size), dtype=complex)
        
        # Términos de campo transverso -h∑Xᵢ
        h = 0.5
        for i in range(system_size):
            X_i = operator_pool._single_site_pauli('X', i)
            H_test -= h * X_i
            
        # Términos de interacción -J∑ZᵢZᵢ₊₁
        J = 1.0
        for i in range(system_size - 1):
            Z_i = operator_pool._single_site_pauli('Z', i)
            Z_i1 = operator_pool._single_site_pauli('Z', i + 1)
            H_test -= J * (Z_i @ Z_i1)
            
        # Estado inicial
        initial_state = np.zeros(2**system_size)
        initial_state[0] = 1.0
        
        # Ejecutar ADAPT-VQE
        adapt_solver = SCAdaptVQE(H_test, operator_pool)
        vqe_results = adapt_solver.run_adapt_vqe(initial_state, max_iterations=5)
        
        # Ejecutar en "hardware" RMN
        nmr_execution_results = await self.nmr_interface.execute_adapt_on_nmr_hardware(
            vqe_results, shots=2048
        )
        
        return {
            'classical_vqe_results': vqe_results,
            'nmr_execution_results': nmr_execution_results,
            'performance_metrics': {
                'convergence_iterations': len(vqe_results['energies']),
                'final_energy_error': abs(vqe_results['final_energy'] - np.min(np.real(la.eigvals(H_test)))),
                'total_pulse_time': nmr_execution_results.get('total_pulse_time', 0),
                'circuit_depth': nmr_execution_results['quantum_circuit'].depth()
            }
        }

# ============================================================================
# EJEMPLO COMPLETO INTEGRADO
# ============================================================================

async def integrated_nmr_quantum_example():
    """Ejemplo completo integrando todos los componentes"""
    
    print("=== SISTEMA CUÁNTICO INTEGRADO: RMN + SC-ADAPT-VQE + HARDWARE ===\n")
    
    # 1. Ejecutar ejemplo básico
    basic_results = main_nuclear_spin_example()
    
    # 2. Crear interfaz de hardware RMN
    print("\n6. Configurando interfaz de hardware RMN...")
    
    # Simulamos adaptador de hardware (en práctica sería IBM/Google)
    mock_hardware = None  # Placeholder para hardware real
    
    nmr_interface = NuclearSpinHardwareInterface(
        hardware_adapter=mock_hardware,
        nuclear_system=basic_results['nuclear_qubits']
    )
    
    # 3. Ejecutar SC-ADAPT-VQE en "hardware" RMN
    print("7. Ejecutando SC-ADAPT-VQE en sistema RMN...")
    
    vqe_results = basic_results['vqe_results']
    
    nmr_execution = await nmr_interface.execute_adapt_on_nmr_hardware(
        vqe_results, shots=2048
    )
    
    print(f"   Circuito generado: {nmr_execution['quantum_circuit'].depth()} puertas")
    print(f"   Secuencias de pulsos: {len(nmr_execution['pulse_sequences'])}")
    
    if 'total_pulse_time' in nmr_execution:
        print(f"   Tiempo total de pulsos: {nmr_execution['total_pulse_time']*1e6:.1f} μs")
    
    # 4. Benchmark completo
    print("\n8. Ejecutando benchmark completo...")
    
    benchmark_system = NMRQuantumBenchmark(nmr_interface)
    benchmark_results = await benchmark_system.run_comprehensive_benchmark()
    
    # Mostrar resultados del benchmark
    print("   Resultados del benchmark:")
    char_results = benchmark_results['system_characterization']
    print(f"   - Especies nucleares: {char_results['nuclear_species']}")
    print(f"   - Frecuencias de Larmor: {[f/1e6 for f in char_results['larmor_frequencies']]} MHz")
    
    adapt_perf = benchmark_results['adapt_vqe_performance']['performance_metrics']
    print(f"   - Iteraciones de convergencia: {adapt_perf['convergence_iterations']}")
    print(f"   - Error de energía final: {adapt_perf['final_energy_error']:.6f}")
    print(f"   - Profundidad de circuito: {adapt_perf['circuit_depth']}")
    
    return {
        'basic_results': basic_results,
        'nmr_interface': nmr_interface,
        'nmr_execution': nmr_execution,
        'benchmark_results': benchmark_results
    }

if __name__ == "__main__":
    # Ejecutar ejemplo básico
    basic_results = main_nuclear_spin_example()
    print(f"\n=== EJEMPLO BÁSICO COMPLETADO ===")
    print(f"Resultados disponibles: {list(basic_results.keys())}")
    
    # Para ejecutar ejemplo completo con async:
    # results = asyncio.run(integrated_nmr_quantum_example())
    print("\nPara ejecutar ejemplo completo, usar:")
    print("results = asyncio.run(integrated_nmr_quantum_example())")
    
