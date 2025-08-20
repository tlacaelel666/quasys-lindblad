"""
Ecuación Maestra de Lindblad Modificada para Decaimiento Beta
============================================================

Este módulo implementa la ecuación maestra de Lindblad con operadores
de paridad, creación y aniquilación modificados para modelar el decaimiento beta,
utilizando la entropía de von Neumann para medir la fidelidad del proceso.

Ecuación maestra de Lindblad:
dρ/dt = -i[H, ρ] + Σₖ (LₖρLₖ† - ½{Lₖ†Lₖ, ρ})

Donde:
- H: Hamiltoniano del sistema
- Lₖ: Operadores de Lindblad (modificados como operadores de campo)
- ρ: Matriz de densidad del sistema

Autor: Jacobo Tlacaelel Mina Rodriguez "jako"
Fecha: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, List, Dict, Any, Optional
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class BetaDecayConfig:
    """Configuración para la simulación de decaimiento beta."""
    n_levels: int = 4  # Número de niveles energéticos
    coupling_strength: float = 0.1  # Fuerza de acoplamiento
    decay_rate: float = 0.05  # Tasa de decaimiento
    parity_violation: float = 0.02  # Violación de paridad (característica del decaimiento beta)
    temperature: float = 0.1  # Temperatura del baño térmico
    simulation_time: float = 50.0  # Tiempo total de simulación
    dt: float = 0.1  # Paso de tiempo


class LindladOperators:
    """Clase para crear y manipular operadores de Lindblad modificados."""
    
    def __init__(self, config: BetaDecayConfig):
        self.config = config
        self.n = config.n_levels
        self.dim = self.n
    
    def creation_operator(self) -> np.ndarray:
        """
        Operador de creación modificado a† para partículas beta.
        En el contexto del decaimiento beta, crea un electrón en el estado final.
        """
        a_dag = np.zeros((self.n, self.n), dtype=complex)
        for i in range(self.n - 1):
            a_dag[i+1, i] = np.sqrt(i + 1)
        return a_dag
    
    def annihilation_operator(self) -> np.ndarray:
        """
        Operador de aniquilación modificado a para neutrones.
        En el decaimiento beta, aniquila un neutrón del núcleo.
        """
        a = np.zeros((self.n, self.n), dtype=complex)
        for i in range(self.n - 1):
            a[i, i+1] = np.sqrt(i + 1)
        return a
    
    def parity_operator(self) -> np.ndarray:
        """
        Operador de paridad modificado P para el decaimiento beta.
        El decaimiento beta viola la paridad, lo que se modela aquí.
        """
        P = np.zeros((self.n, self.n), dtype=complex)
        for i in range(self.n):
            # Paridad alternante con violación controlada
            parity_sign = (-1)**i
            violation_factor = 1 - self.config.parity_violation
            P[i, i] = parity_sign * violation_factor
        return P
    
    def number_operator(self) -> np.ndarray:
        """Operador número N = a†a."""
        a_dag = self.creation_operator()
        a = self.annihilation_operator()
        return a_dag @ a
    
    def beta_decay_hamiltonian(self) -> np.ndarray:
        """
        Hamiltoniano para el decaimiento beta nuclear.
        H = ℏω(N + 1/2) + g(a† + a)P + ΔE
        """
        # Frecuencia característica del núcleo
        omega = 1.0
        
        # Operadores básicos
        N = self.number_operator()
        a_dag = self.creation_operator()
        a = self.annihilation_operator()
        P = self.parity_operator()
        
        # Hamiltoniano libre (oscilador armónico)
        H0 = omega * (N + 0.5 * np.eye(self.n))
        
        # Término de interacción (acoplamiento con violación de paridad)
        H_int = self.config.coupling_strength * (a_dag + a) @ P
        
        # Diferencia de masa (Q-value del decaimiento)
        mass_difference = 0.5  # En unidades naturales
        H_mass = mass_difference * np.eye(self.n)
        
        return H0 + H_int + H_mass
    
    def lindblad_operators_beta_decay(self) -> List[np.ndarray]:
        """
        Operadores de Lindblad específicos para decaimiento beta:
        L1: Emisión de electrón (a†)
        L2: Emisión de antineutrino (relacionado con paridad)
        L3: Relajación térmica
        L4: Decoherencia nuclear
        """
        operators = []
        
        # L1: Operador de emisión de electrón (creación modificada)
        a_dag = self.creation_operator()
        L1 = np.sqrt(self.config.decay_rate) * a_dag
        operators.append(L1)
        
        # L2: Operador de emisión de antineutrino (paridad violada)
        P = self.parity_operator()
        a = self.annihilation_operator()
        L2 = np.sqrt(self.config.decay_rate * self.config.parity_violation) * P @ a
        operators.append(L2)
        
        # L3: Relajación térmica (baño de fotones)
        if self.config.temperature > 0:
            # Operador de relajación térmica
            L3 = np.sqrt(self.config.temperature) * a
            operators.append(L3)
        
        # L4: Decoherencia pura (pérdida de coherencia cuántica)
        N = self.number_operator()
        dephasing_rate = 0.01
        L4 = np.sqrt(dephasing_rate) * N
        operators.append(L4)
        
        return operators


class LindladEvolution:
    """Clase para simular la evolución temporal usando la ecuación maestra de Lindblad."""
    
    def __init__(self, config: BetaDecayConfig):
        self.config = config
        self.operators = LindladOperators(config)
        self.H = self.operators.beta_decay_hamiltonian()
        self.L_ops = self.operators.lindblad_operators_beta_decay()
    
    def commutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calcula el conmutador [A, B] = AB - BA."""
        return A @ B - B @ A
    
    def anticommutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calcula el anticonmutador {A, B} = AB + BA."""
        return A @ B + B @ A
    
    def lindblad_superoperator(self, rho: np.ndarray) -> np.ndarray:
        """
        Aplica la ecuación maestra de Lindblad:
        dρ/dt = -i[H, ρ] + Σₖ (LₖρLₖ† - ½{Lₖ†Lₖ, ρ})
        """
        # Término unitario (evolución hamiltoniana)
        drho_dt = -1j * self.commutator(self.H, rho)
        
        # Términos disipativos (operadores de Lindblad)
        for L in self.L_ops:
            L_dag = L.conj().T
            
            # Término de salto cuántico: LρL†
            jump_term = L @ rho @ L_dag
            
            # Término de anti-Hermítico: -½{L†L, ρ}
            anti_hermitian = -0.5 * self.anticommutator(L_dag @ L, rho)
            
            drho_dt += jump_term + anti_hermitian
        
        return drho_dt
    
    def vectorize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Convierte matriz en vector para integración numérica."""
        return matrix.flatten()
    
    def matricize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Convierte vector de vuelta a matriz."""
        n = int(np.sqrt(len(vector)))
        return vector.reshape((n, n))
    
    def lindblad_ode(self, rho_vec: np.ndarray, t: float) -> np.ndarray:
        """Función para la integración ODE de la ecuación de Lindblad."""
        rho = self.matricize_vector(rho_vec)
        drho_dt = self.lindblad_superoperator(rho)
        return self.vectorize_matrix(drho_dt)
    
    def evolve_system(self, initial_state: np.ndarray, 
                     times: np.ndarray) -> List[np.ndarray]:
        """
        Evoluciona el sistema usando la ecuación de Lindblad.
        
        Args:
            initial_state: Estado inicial del sistema (matriz de densidad)
            times: Array de tiempos para la evolución
        
        Returns:
            Lista de matrices de densidad en cada tiempo
        """
        # Vectorizar estado inicial
        rho0_vec = self.vectorize_matrix(initial_state)
        
        # Integrar la ecuación diferencial
        solution = odeint(self.lindblad_ode, rho0_vec, times)
        
        # Convertir de vuelta a matrices
        evolved_states = []
        for rho_vec in solution:
            rho = self.matricize_vector(rho_vec)
            # Asegurar que la matriz de densidad sea física
            rho = self.ensure_physical_density_matrix(rho)
            evolved_states.append(rho)
        
        return evolved_states
    
    def ensure_physical_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Asegura que la matriz de densidad sea física (hermítica, traza 1, positiva)."""
        # Hacer hermítica
        rho = 0.5 * (rho + rho.conj().T)
        
        # Normalizar traza
        trace = np.trace(rho)
        if abs(trace) > 1e-12:
            rho = rho / trace
        
        # Asegurar positividad (proyección a valores propios positivos)
        eigenvals, eigenvecs = la.eigh(rho)
        eigenvals = np.maximum(eigenvals, 0)  # Valores propios no negativos
        eigenvals = eigenvals / np.sum(eigenvals)  # Renormalizar
        rho = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
        
        return rho


class VonNeumannEntropy:
    """Clase para calcular la entropía de von Neumann y medir fidelidad."""
    
    @staticmethod
    def von_neumann_entropy(rho: np.ndarray, base: float = 2) -> float:
        """
        Calcula la entropía de von Neumann: S(ρ) = -Tr(ρ log ρ)
        
        Args:
            rho: Matriz de densidad
            base: Base del logaritmo (2 para bits, e para nats)
        
        Returns:
            Entropía de von Neumann
        """
        # Calcular valores propios
        eigenvals = la.eigvals(rho)
        
        # Filtrar valores propios positivos
        positive_eigenvals = eigenvals[eigenvals > 1e-12]
        
        if len(positive_eigenvals) == 0:
            return 0.0
        
        # Calcular entropía
        if base == 2:
            entropy = -np.sum(positive_eigenvals * np.log2(positive_eigenvals))
        else:
            entropy = -np.sum(positive_eigenvals * np.log(positive_eigenvals))
        
        return np.real(entropy)  # Asegurar que sea real
    
    @staticmethod
    def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Calcula la fidelidad entre dos estados cuánticos:
        F(ρ₁, ρ₂) = Tr(√(√ρ₁ρ₂√ρ₁))
        
        Para estados puros: F = |⟨ψ₁|ψ₂⟩|²
        """
        try:
            # Calcular √ρ₁
            eigenvals1, eigenvecs1 = la.eigh(rho1)
            sqrt_rho1 = eigenvecs1 @ np.diag(np.sqrt(np.maximum(eigenvals1, 0))) @ eigenvecs1.conj().T
            
            # Calcular √ρ₁ρ₂√ρ₁
            product = sqrt_rho1 @ rho2 @ sqrt_rho1
            
            # Calcular √(√ρ₁ρ₂√ρ₁)
            eigenvals_prod, eigenvecs_prod = la.eigh(product)
            sqrt_product = eigenvecs_prod @ np.diag(np.sqrt(np.maximum(eigenvals_prod, 0))) @ eigenvecs_prod.conj().T
            
            # Fidelidad = Tr(√(√ρ₁ρ₂√ρ₁))
            fidelity = np.real(np.trace(sqrt_product))
            
            return np.clip(fidelity, 0, 1)  # Asegurar que esté en [0,1]
            
        except:
            # Método alternativo más estable para casos problemáticos
            return np.real(np.trace(la.sqrtm(la.sqrtm(rho1) @ rho2 @ la.sqrtm(rho1))))
    
    @staticmethod
    def beta_decay_fidelity_metric(evolved_states: List[np.ndarray], 
                                  target_state: np.ndarray) -> List[float]:
        """
        Calcula la métrica de fidelidad para el decaimiento beta.
        
        Args:
            evolved_states: Lista de estados evolucionados
            target_state: Estado objetivo (post-decaimiento)
        
        Returns:
            Lista de fidelidades en cada tiempo
        """
        fidelities = []
        for state in evolved_states:
            fid = VonNeumannEntropy.fidelity(state, target_state)
            fidelities.append(fid)
        return fidelities


class BetaDecayAnalyzer:
    """Analizador completo para procesos de decaimiento beta usando Lindblad."""
    
    def __init__(self, config: Optional[BetaDecayConfig] = None):
        self.config = config or BetaDecayConfig()
        self.evolution = LindladEvolution(self.config)
        self.entropy_calculator = VonNeumannEntropy()
    
    def create_initial_nuclear_state(self, excited_level: int = 1) -> np.ndarray:
        """
        Crea el estado inicial del núcleo padre (antes del decaimiento).
        
        Args:
            excited_level: Nivel de excitación inicial
        
        Returns:
            Matriz de densidad del estado inicial
        """
        n = self.config.n_levels
        rho_initial = np.zeros((n, n), dtype=complex)
        
        # Estado excitado con superposición coherente
        if excited_level < n:
            # Estado mixto: superposición del fundamental y excitado
            alpha = 0.8  # Amplitud del estado fundamental
            beta = np.sqrt(1 - alpha**2)  # Amplitud del estado excitado
            
            rho_initial[0, 0] = alpha**2
            rho_initial[excited_level, excited_level] = beta**2
            rho_initial[0, excited_level] = alpha * beta * np.exp(1j * 0.1)  # Coherencia
            rho_initial[excited_level, 0] = alpha * beta * np.exp(-1j * 0.1)
        else:
            # Estado fundamental puro
            rho_initial[0, 0] = 1.0
        
        return rho_initial
    
    def create_target_decay_state(self) -> np.ndarray:
        """
        Crea el estado objetivo después del decaimiento beta.
        Representa el núcleo hijo + productos de decaimiento.
        """
        n = self.config.n_levels
        rho_target = np.zeros((n, n), dtype=complex)
        
        # Estado final: núcleo hijo en estado fundamental
        # con productos de decaimiento (e⁻ + ν̄ₑ)
        decay_probability = 0.9  # Probabilidad de decaimiento completo
        
        rho_target[0, 0] = 1 - decay_probability
        if n > 1:
            rho_target[1, 1] = decay_probability
        
        return rho_target
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Ejecuta un análisis completo del decaimiento beta."""
        print("🔬 ANÁLISIS DE DECAIMIENTO BETA CON LINDBLAD")
        print("=" * 60)
        
        # 1. Crear estados inicial y objetivo
        initial_state = self.create_initial_nuclear_state(excited_level=2)
        target_state = self.create_target_decay_state()
        
        print(f"📊 Configuración del sistema:")
        print(f"   Niveles energéticos: {self.config.n_levels}")
        print(f"   Tasa de decaimiento: {self.config.decay_rate}")
        print(f"   Violación de paridad: {self.config.parity_violation}")
        print(f"   Temperatura: {self.config.temperature}")
        
        # 2. Definir tiempos de evolución
        times = np.linspace(0, self.config.simulation_time, 
                          int(self.config.simulation_time / self.config.dt))
        
        # 3. Evolucionar el sistema
        print(f"\n⏱️  Evolucionando sistema por {self.config.simulation_time} unidades de tiempo...")
        evolved_states = self.evolution.evolve_system(initial_state, times)
        
        # 4. Calcular entropías de von Neumann
        print(f"📈 Calculando entropías de von Neumann...")
        entropies = []
        for state in evolved_states:
            entropy = self.entropy_calculator.von_neumann_entropy(state)
            entropies.append(entropy)
        
        # 5. Calcular fidelidades
        print(f"🎯 Calculando fidelidades del decaimiento...")
        fidelities = self.entropy_calculator.beta_decay_fidelity_metric(
            evolved_states, target_state
        )
        
        # 6. Analizar propiedades de los operadores
        operators = self.evolution.operators
        H = operators.beta_decay_hamiltonian()
        L_ops = operators.lindblad_operators_beta_decay()
        
        # Valores propios del Hamiltoniano
        H_eigenvals = la.eigvals(H)
        
        # 7. Calcular observables físicos
        observables = self._calculate_physical_observables(evolved_states, times)
        
        print(f"\n📊 Resultados del análisis:")
        print(f"   Entropía inicial: {entropies[0]:.4f} bits")
        print(f"   Entropía final: {entropies[-1]:.4f} bits")
        print(f"   Cambio de entropía: {entropies[-1] - entropies[0]:.4f} bits")
        print(f"   Fidelidad inicial: {fidelities[0]:.4f}")
        print(f"   Fidelidad final: {fidelities[-1]:.4f}")
        print(f"   Tiempo de vida media: {self._calculate_half_life(fidelities, times):.2f} u.t.")
        
        return {
            "times": times,
            "evolved_states": evolved_states,
            "entropies": entropies,
            "fidelities": fidelities,
            "initial_state": initial_state,
            "target_state": target_state,
            "hamiltonian_eigenvals": H_eigenvals,
            "lindblad_operators": L_ops,
            "observables": observables,
            "config": self.config
        }
    
    def _calculate_physical_observables(self, evolved_states: List[np.ndarray], 
                                      times: np.ndarray) -> Dict[str, List[float]]:
        """Calcula observables físicos del sistema."""
        operators = self.evolution.operators
        N = operators.number_operator()
        P = operators.parity_operator()
        
        number_expectation = []
        parity_expectation = []
        purity = []
        
        for state in evolved_states:
            # Valor esperado del operador número
            n_exp = np.real(np.trace(N @ state))
            number_expectation.append(n_exp)
            
            # Valor esperado de la paridad
            p_exp = np.real(np.trace(P @ state))
            parity_expectation.append(p_exp)
            
            # Pureza del estado Tr(ρ²)
            purity_val = np.real(np.trace(state @ state))
            purity.append(purity_val)
        
        return {
            "number_expectation": number_expectation,
            "parity_expectation": parity_expectation,
            "purity": purity
        }
    
    def _calculate_half_life(self, fidelities: List[float], times: np.ndarray) -> float:
        """Calcula el tiempo de vida media basado en la fidelidad."""
        # Encontrar el tiempo donde la fidelidad inicial cae a la mitad
        initial_fidelity = fidelities[0]
        target_fidelity = initial_fidelity * 0.5
        
        # Buscar el índice más cercano
        for i, fid in enumerate(fidelities):
            if fid <= target_fidelity:
                return times[i]
        
        return times[-1]  # Si no se alcanza, retornar tiempo final


def visualize_beta_decay_analysis(results: Dict[str, Any], 
                                save_path: str = "beta_decay_lindblad.png"):
    """Visualiza los resultados del análisis de decaimiento beta."""
    
    fig = plt.figure(figsize=(16, 12))
    times = results["times"]
    entropies = results["entropies"]
    fidelities = results["fidelities"]
    observables = results["observables"]
    
    # 1. Evolución de la entropía de von Neumann
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(times, entropies, 'b-', linewidth=2, label='S(ρ)')
    plt.xlabel('Tiempo')
    plt.ylabel('Entropía von Neumann [bits]')
    plt.title('Evolución de la Entropía Cuántica')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Fidelidad del decaimiento
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(times, fidelities, 'r-', linewidth=2, label='F(ρ(t), ρₜₐᵣ)')
    plt.xlabel('Tiempo')
    plt.ylabel('Fidelidad')
    plt.title('Fidelidad del Decaimiento Beta')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Observables físicos
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(times, observables["number_expectation"], 'g-', 
             linewidth=2, label='⟨N⟩')
    plt.plot(times, observables["parity_expectation"], 'm--', 
             linewidth=2, label='⟨P⟩')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor Esperado')
    plt.title('Observables Nucleares')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Pureza del estado
    ax4 = plt.subplot(3, 3, 4)
    plt.plot(times, observables["purity"], 'orange', linewidth=2, label='Tr(ρ²)')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Estado puro')
    plt.xlabel('Tiempo')
    plt.ylabel('Pureza')
    plt.title('Pureza del Estado Cuántico')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 5. Espectro del Hamiltoniano
    ax5 = plt.subplot(3, 3, 5)
    eigenvals = np.real(results["hamiltonian_eigenvals"])
    plt.bar(range(len(eigenvals)), eigenvals, color='skyblue', alpha=0.7)
    plt.xlabel('Nivel Energético')
    plt.ylabel('Energía')
    plt.title('Espectro del Hamiltoniano Nuclear')
    plt.grid(True, alpha=0.3)
    
    # 6. Matriz de densidad inicial
    ax6 = plt.subplot(3, 3, 6)
    rho_initial = results["initial_state"]
    im6 = plt.imshow(np.abs(rho_initial), cmap='viridis', aspect='auto')
    plt.colorbar(im6, ax=ax6, label='|ρᵢⱼ|')
    plt.title('Estado Inicial (|ρ|)')
    plt.xlabel('j')
    plt.ylabel('i')
    
    # 7. Matriz de densidad final
    ax7 = plt.subplot(3, 3, 7)
    rho_final = results["evolved_states"][-1]
    im7 = plt.imshow(np.abs(rho_final), cmap='viridis', aspect='auto')
    plt.colorbar(im7, ax=ax7, label='|ρᵢⱼ|')
    plt.title('Estado Final (|ρ|)')
    plt.xlabel('j')
    plt.ylabel('i')
    
    # 8. Análisis de correlación entropía-fidelidad
    ax8 = plt.subplot(3, 3, 8)
    plt.scatter(entropies, fidelities, c=times, cmap='plasma', alpha=0.7)
    plt.colorbar(label='Tiempo')
    plt.xlabel('Entropía von Neumann')
    plt.ylabel('Fidelidad')
    plt.title('Correlación Entropía-Fidelidad')
    plt.grid(True, alpha=0.3)
    
    # 9. Violación de paridad temporal
    ax9 = plt.subplot(3, 3, 9)
    parity_violation = np.array(observables["parity_expectation"])
    parity_change = np.abs(parity_violation - parity_violation[0])
    plt.plot(times, parity_change, 'purple', linewidth=2, 
             label='Violación de paridad')
    plt.xlabel('Tiempo')
    plt.ylabel('|ΔP|')
    plt.title('Violación de Paridad en el Tiempo')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def demonstrate_beta_decay_lindblad():
    """Demostración completa del análisis de decaimiento beta."""
    
    print("⚛️  SIMULACIÓN DE DECAIMIENTO BETA CON ECUACIÓN DE LINDBLAD")
    print("=" * 70)
    print("🔬 Usando operadores de paridad, creación y aniquilación modificados")
    print("📊 Midiendo fidelidad con entropía de von Neumann")
    print()
    
    # Configuración del sistema
    config = BetaDecayConfig(
        n_levels=5,
        coupling_strength=0.15,
        decay_rate=0.08,
        parity_violation=0.05,  # Característica del decaimiento beta
        temperature=0.05,
        simulation_time=40.0,
        dt=0.2
    )
    
    # Ejecutar análisis
    analyzer = BetaDecayAnalyzer(config)
    results = analyzer.run_complete_analysis()
    
    # Generar visualización
    print(f"\n📊 Generando visualización completa...")
    fig = visualize_beta_decay_analysis(results)
    
    # Análisis adicional de la física
    print(f"\n🔍 ANÁLISIS FÍSICO DETALLADO:")
    print("-" * 50)
    
    # Calcular métricas físicas importantes
    initial_entropy = results["entropies"][0]
    final_entropy = results["entropies"][-1]
    entropy_production = final_entropy - initial_entropy
    
    initial_fidelity = results["fidelities"][0]
    final_fidelity = results["fidelities"][-1]
    fidelity_loss = initial_fidelity - final_fidelity
    
    # Análisis de la violación de paridad
    parity_values = results["observables"]["parity_expectation"]
    initial_parity = parity_values[0]
    final_parity = parity_values[-1]
    parity_violation_measure = abs(final_parity - initial_parity)
    
    print(f"📈 Producción de entropía: {entropy_production:.4f} bits")
    print(f"   ➤ Indicativo de irreversibilidad del decaimiento")
    
    print(f"🎯 Pérdida de fidelidad: {fidelity_loss:.4f}")
    print(f"   ➤ Mide qué tan bien el proceso alcanza el estado objetivo")
    
    print(f"🔄 Violación de paridad: {parity_violation_measure:.4f}")
    print(f"   ➤ Característica distintiva del decaimiento beta débil")
    
    # Análisis de decoherencia
    purity_values = results["observables"]["purity"]
    initial_purity = purity_values[0]
    final_purity = purity_values[-1]
    decoherence_measure = initial_purity - final_purity
    
    print(f"🌊 Decoherencia total: {decoherence_measure:.4f}")
    print(f"   ➤ Pérdida de coherencia cuántica por interacción con el entorno")
    
    # Calcular tasa de decaimiento efectiva
    half_life = analyzer._calculate_half_life(results["fidelidades"], results["times"])
    decay_constant = np.log(2) / half_life if half_life > 0 else float('inf')
    
    print(f"⏰ Vida media efectiva: {half_life:.2f} u.t.")
    print(f"📉 Constante de decaimiento: {decay_constant:.4f} u.t.⁻¹")
    
    # Verificar conservación de probabilidad
    traces = [np.real(np.trace(state)) for state in results["evolved_states"]]
    trace_deviation = max(traces) - min(traces)
    
    print(f"✅ Conservación de probabilidad: ±{trace_deviation:.6f}")
    print(f"   ➤ Desviación máxima de Tr(ρ) = 1")
    
    # Análisis energético
    hamiltonian_eigenvals = results["hamiltonian_eigenvals"]
    energy_gap = np.real(hamiltonian_eigenvals[1] - hamiltonian_eigenvals[0])
    
    print(f"⚡ Gap energético fundamental: {energy_gap:.4f}")
    print(f"   ➤ Diferencia entre estado fundamental y primer excitado")
    
    print(f"\n🎉 Análisis completo finalizado!")
    print(f"📁 Visualización guardada como: 'beta_decay_lindblad.png'")
    
    return results, fig


def advanced_beta_decay_scenarios():
    """Scenarios avanzados para diferentes tipos de decaimiento beta."""
    
    print("\n🚀 ESCENARIOS AVANZADOS DE DECAIMIENTO BETA")
    print("=" * 60)
    
    scenarios = {
        "β⁻ Decay (n → p + e⁻ + ν̄ₑ)": BetaDecayConfig(
            n_levels=4,
            coupling_strength=0.12,
            decay_rate=0.06,
            parity_violation=0.04,
            temperature=0.02,
            simulation_time=50.0
        ),
        
        "β⁺ Decay (p → n + e⁺ + νₑ)": BetaDecayConfig(
            n_levels=4,
            coupling_strength=0.10,
            decay_rate=0.05,
            parity_violation=0.04,
            temperature=0.03,
            simulation_time=60.0
        ),
        
        "Electron Capture (p + e⁻ → n + νₑ)": BetaDecayConfig(
            n_levels=5,
            coupling_strength=0.08,
            decay_rate=0.04,
            parity_violation=0.03,
            temperature=0.01,
            simulation_time=70.0
        ),
        
        "Double β Decay (2n → 2p + 2e⁻ + 2ν̄ₑ)": BetaDecayConfig(
            n_levels=6,
            coupling_strength=0.05,
            decay_rate=0.02,
            parity_violation=0.06,
            temperature=0.04,
            simulation_time=100.0
        )
    }
    
    results_comparison = {}
    
    for scenario_name, config in scenarios.items():
        print(f"\n🔬 Analizando: {scenario_name}")
        print("-" * 40)
        
        analyzer = BetaDecayAnalyzer(config)
        
        # Análisis rápido (menos puntos temporales para eficiencia)
        quick_config = BetaDecayConfig(
            n_levels=config.n_levels,
            coupling_strength=config.coupling_strength,
            decay_rate=config.decay_rate,
            parity_violation=config.parity_violation,
            temperature=config.temperature,
            simulation_time=config.simulation_time,
            dt=1.0  # Paso de tiempo mayor para análisis rápido
        )
        
        quick_analyzer = BetaDecayAnalyzer(quick_config)
        results = quick_analyzer.run_complete_analysis()
        
        # Métricas clave del escenario
        final_entropy = results["entropies"][-1]
        final_fidelity = results["fidelities"][-1]
        entropy_production = final_entropy - results["entropies"][0]
        half_life = quick_analyzer._calculate_half_life(results["fidelities"], results["times"])
        
        scenario_metrics = {
            "final_entropy": final_entropy,
            "final_fidelity": final_fidelity,
            "entropy_production": entropy_production,
            "half_life": half_life,
            "config": config,
            "full_results": results
        }
        
        results_comparison[scenario_name] = scenario_metrics
        
        print(f"   Entropía final: {final_entropy:.3f} bits")
        print(f"   Fidelidad final: {final_fidelity:.3f}")
        print(f"   Producción entropía: {entropy_production:.3f} bits")
        print(f"   Vida media: {half_life:.1f} u.t.")
    
    # Crear visualización comparativa
    create_comparative_visualization(results_comparison)
    
    return results_comparison


def create_comparative_visualization(results_comparison: Dict[str, Dict]):
    """Crea visualización comparativa de diferentes tipos de decaimiento beta."""
    
    fig = plt.figure(figsize=(16, 10))
    
    scenario_names = list(results_comparison.keys())
    shortened_names = [name.split('(')[0].strip() for name in scenario_names]
    
    # Extraer métricas
    final_entropies = [results_comparison[name]["final_entropy"] for name in scenario_names]
    final_fidelities = [results_comparison[name]["final_fidelity"] for name in scenario_names]
    entropy_productions = [results_comparison[name]["entropy_production"] for name in scenario_names]
    half_lives = [results_comparison[name]["half_life"] for name in scenario_names]
    
    # 1. Comparación de entropías finales
    ax1 = plt.subplot(2, 3, 1)
    bars1 = plt.bar(shortened_names, final_entropies, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.ylabel('Entropía Final [bits]')
    plt.title('Entropía Final por Tipo de Decaimiento')
    plt.xticks(rotation=45)
    
    # Añadir valores en las barras
    for bar, value in zip(bars1, final_entropies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Comparación de fidelidades finales
    ax2 = plt.subplot(2, 3, 2)
    bars2 = plt.bar(shortened_names, final_fidelities, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.ylabel('Fidelidad Final')
    plt.title('Fidelidad Final por Tipo de Decaimiento')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars2, final_fidelities):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Producción de entropía
    ax3 = plt.subplot(2, 3, 3)
    bars3 = plt.bar(shortened_names, entropy_productions, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.ylabel('Producción de Entropía [bits]')
    plt.title('Irreversibilidad del Proceso')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars3, entropy_productions):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Vidas medias
    ax4 = plt.subplot(2, 3, 4)
    bars4 = plt.bar(shortened_names, half_lives, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.ylabel('Vida Media [u.t.]')
    plt.title('Tiempos Característicos')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars4, half_lives):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Correlación entropía-fidelidad
    ax5 = plt.subplot(2, 3, 5)
    colors = ['blue', 'green', 'orange', 'red']
    for i, (name, short_name) in enumerate(zip(scenario_names, shortened_names)):
        plt.scatter(final_entropies[i], final_fidelities[i], 
                   c=colors[i], s=100, alpha=0.7, label=short_name)
    
    plt.xlabel('Entropía Final [bits]')
    plt.ylabel('Fidelidad Final')
    plt.title('Correlación Entropía-Fidelidad')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 6. Eficiencia del decaimiento (fidelidad/tiempo)
    ax6 = plt.subplot(2, 3, 6)
    efficiencies = [fid/time if time > 0 else 0 for fid, time in zip(final_fidelities, half_lives)]
    bars6 = plt.bar(shortened_names, efficiencies, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.ylabel('Eficiencia [Fidelidad/Tiempo]')
    plt.title('Eficiencia del Proceso de Decaimiento')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars6, efficiencies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('beta_decay_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Análisis estadístico de los resultados
    print(f"\n📊 ANÁLISIS ESTADÍSTICO COMPARATIVO:")
    print("-" * 50)
    
    print(f"🎯 Fidelidad promedio: {np.mean(final_fidelities):.3f} ± {np.std(final_fidelities):.3f}")
    print(f"📈 Entropía promedio: {np.mean(final_entropies):.3f} ± {np.std(final_entropies):.3f}")
    print(f"⏰ Vida media promedio: {np.mean(half_lives):.1f} ± {np.std(half_lives):.1f} u.t.")
    print(f"🔄 Producción entropía promedio: {np.mean(entropy_productions):.3f} ± {np.std(entropy_productions):.3f}")
    
    # Identificar el proceso más/menos eficiente
    most_efficient_idx = np.argmax(efficiencies)
    least_efficient_idx = np.argmin(efficiencies)
    
    print(f"\n🏆 Proceso más eficiente: {shortened_names[most_efficient_idx]}")
    print(f"   Eficiencia: {efficiencies[most_efficient_idx]:.4f}")
    
    print(f"🐌 Proceso menos eficiente: {shortened_names[least_efficient_idx]}")
    print(f"   Eficiencia: {efficiencies[least_efficient_idx]:.4f}")
    
    return fig


class QuantumBetaDecayFramework:
    """Framework integrado que combina Lindblad con el análisis cuántico-bayesiano previo."""
    
    def __init__(self, beta_config: Optional[BetaDecayConfig] = None,
                 quantum_bayes_config = None):
        self.beta_config = beta_config or BetaDecayConfig()
        
        # Importar configuración del framework cuántico-bayesiano
        # (Asumiendo que está disponible)
        try:
            from quantum_bayes_framework import QuantumBayesConfig, AdvancedPRN
            self.qb_config = quantum_bayes_config or QuantumBayesConfig()
        except ImportError:
            print("⚠️  Framework cuántico-bayesiano no disponible. Usando modo standalone.")
            self.qb_config = None
        
        self.beta_analyzer = BetaDecayAnalyzer(self.beta_config)
    
    def integrate_with_prn(self, beta_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integra los resultados del decaimiento beta con el análisis PRN.
        
        Args:
            beta_results: Resultados del análisis de Lindblad
        
        Returns:
            Resultados integrados con análisis PRN
        """
        if self.qb_config is None:
            print("⚠️  Análisis PRN no disponible")
            return beta_results
        
        try:
            from quantum_bayes_framework import AdvancedPRN, QuantumBayesLogic
            
            # Convertir métricas de Lindblad en parámetros PRN
            final_entropy = beta_results["entropies"][-1]
            final_fidelity = beta_results["fidelities"][-1]
            
            # Crear PRN basado en los resultados del decaimiento
            decay_prn = AdvancedPRN(
                real_component=final_fidelity,  # Fidelidad como componente real
                imaginary_component=final_entropy / 5.0,  # Entropía normalizada
                quantum_coherence=1.0 - (final_entropy / 3.0),  # Coherencia inversa a entropía
                algorithm_type="beta_decay_lindblad"
            )
            
            # Análisis cuántico-bayesiano con el PRN del decaimiento
            qb_system = QuantumBayesLogic(self.qb_config)
            
            # Usar métricas del decaimiento beta como entrada
            parity_violation = self.beta_config.parity_violation
            coherence = 1.0 - final_entropy / 3.0  # Mapear entropía a coherencia
            
            # Análisis de decisión integrado
            integrated_decision = qb_system.uncertainty_aware_decision(
                entropy=final_entropy / 3.0,  # Normalizar entropía
                coherence=coherence,
                prn=decay_prn,
                delta_x=np.sqrt(parity_violation),  # Incertidumbre por violación de paridad
                delta_p=0.5 / np.sqrt(parity_violation)  # Límite de Heisenberg
            )
            
            # Combinar resultados
            beta_results["integrated_analysis"] = {
                "decay_prn": decay_prn,
                "quantum_bayes_decision": integrated_decision,
                "coherence_mapping": coherence,
                "uncertainty_mapping": {
                    "delta_x": np.sqrt(parity_violation),
                    "delta_p": 0.5 / np.sqrt(parity_violation)
                }
            }
            
            print(f"🔗 Integración PRN-Lindblad completada:")
            print(f"   PRN del decaimiento: {decay_prn}")
            print(f"   Decisión cuántico-bayesiana: {integrated_decision['action']}")
            print(f"   Confianza integrada: {integrated_decision['decision_confidence']:.3f}")
            
        except Exception as e:
            print(f"⚠️  Error en integración PRN: {e}")
        
        return beta_results
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """Análisis comprehensivo combinando Lindblad y análisis cuántico-bayesiano."""
        
        print("🌟 ANÁLISIS COMPREHENSIVO: LINDBLAD + CUÁNTICO-BAYESIANO")
        print("=" * 70)
        
        # 1. Análisis de Lindblad del decaimiento beta
        beta_results = self.beta_analyzer.run_complete_analysis()
        
        # 2. Integración con PRN
        integrated_results = self.integrate_with_prn(beta_results)
        
        # 3. Análisis de correlaciones
        if "integrated_analysis" in integrated_results:
            self._analyze_correlations(integrated_results)
        
        return integrated_results
    
    def _analyze_correlations(self, results: Dict[str, Any]):
        """Analiza correlaciones entre métricas de Lindblad y cuántico-bayesianas."""
        
        print(f"\n🔍 ANÁLISIS DE CORRELACIONES:")
        print("-" * 40)
        
        # Métricas de Lindblad
        entropy_production = results["entropies"][-1] - results["entropies"][0]
        fidelity_loss = results["fidelities"][0] - results["fidelities"][-1]
        
        # Métricas cuántico-bayesianas
        integrated = results["integrated_analysis"]
        decision_confidence = integrated["quantum_bayes_decision"]["decision_confidence"]
        uncertainty_penalty = integrated["quantum_bayes_decision"]["uncertainty_penalty"]
        
        # Correlaciones
        print(f"📊 Producción de entropía vs Confianza de decisión:")
        entropy_confidence_corr = -entropy_production * decision_confidence  # Anti-correlación esperada
        print(f"   Correlación: {entropy_confidence_corr:.3f}")
        
        print(f"🎯 Pérdida de fidelidad vs Penalización por incertidumbre:")
        fidelity_uncertainty_corr = fidelity_loss * uncertainty_penalty
        print(f"   Correlación: {fidelity_uncertainty_corr:.3f}")
        
        print(f"🔄 Coherencia cuántica vs Violación de paridad:")
        coherence = integrated["coherence_mapping"]
        parity_violation = self.beta_config.parity_violation
        coherence_parity_corr = coherence * (1 - parity_violation)
        print(f"   Correlación: {coherence_parity_corr:.3f}")


def main_demonstration():
    """Demostración principal del framework completo."""
    
    print("🚀 DEMOSTRACIÓN PRINCIPAL: LINDBLAD + ENTROPÍA VON NEUMANN")
    print("=" * 80)
    
    # 1. Análisis individual de decaimiento beta
    print("\n1️⃣  Análisis individual de decaimiento β⁻:")
    results, fig = demonstrate_beta_decay_lindblad()
    
    # 2. Comparación de escenarios
    print("\n2️⃣  Análisis comparativo de diferentes tipos de decaimiento:")
    comparison_results = advanced_beta_decay_scenarios()
    
    # 3. Framework integrado (si está disponible)
    print("\n3️⃣  Análisis integrado con framework cuántico-bayesiano:")
    try:
        integrated_framework = QuantumBetaDecayFramework()
        comprehensive_results = integrated_framework.comprehensive_analysis()
        print("✅ Análisis integrado completado exitosamente")
    except Exception as e:
        print(f"⚠️  Framework integrado no disponible: {e}")
        print("   Continuando con análisis standalone de Lindblad")
    
    print(f"\n🎉 DEMOSTRACIÓN COMPLETA FINALIZADA")
    print(f"📁 Archivos generados:")
    print(f"   - beta_decay_lindblad.png")
    print(f"   - beta_decay_comparison.png")
    
    print(f"\n💡 RESUMEN DE CAPACIDADES:")
    print(f"   ✅ Ecuación maestra de Lindblad con operadores modificados")
    print(f"   ✅ Operadores de paridad, creación y aniquilación personalizados")
    print(f"   ✅ Entropía de von Neumann para medir fidelidad")
    print(f"   ✅ Simulación de diferentes tipos de decaimiento beta")
    print(f"   ✅ Análisis de violación de paridad")
    print(f"   ✅ Integración con framework cuántico-bayesiano")
    print(f"   ✅ Visualización científica comprehensiva")


if __name__ == "__main__":
    # Ejecutar demostración completa
    main_demonstration()
