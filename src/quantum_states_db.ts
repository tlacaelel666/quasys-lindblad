import type { QuantumState } from '../types';

// NOTE: Estos vectores de características son representaciones ilustrativas para una base de datos simulada
// de estados cuánticos. En una aplicación del mundo real, estos serían pre-computados analizando
// un gran conjunto de datos de estados cuánticos etiquetados usando la función `extractQuantumFeatures`.

export const QUANTUM_DATABASE: QuantumState[] = [
  {
    id: 'coherent_superposition',
    name: 'Superposición Coherente',
    // Caracterizado por alta coherencia cuántica y baja entropía de Shannon.
    // Alto valor de pureza, baja entropía de von Neumann, incertidumbre mínima.
    featureVector: [
      0.95, 0.02, 25, 0.08, 5, 15, 30, 25, 10, 4, 1.2, 0.8,
      2.5, 2.2, 1.8, 1.5, 1.2, 1.0, 3, 2.8, 2.5, 2.2, 2, 1.8, 1.5, 1.2, 1, 0.8, 0.5, 0.2, 0.1, 0.05
    ],
    // Propiedades cuánticas específicas
    shannonEntropy: 0.02,
    vonNeumannEntropy: 0.05,
    heisenbergUncertainty: 0.5,
    lindbladian: 0.1,
    creationOperator: 1.0,
    annihilationOperator: 1.0,
    parityOperator: 1.0
  },
  {
    id: 'entangled_state',
    name: 'Estado Entrelazado',
    // Estado altamente correlacionado con entropía elevada.
    // Muy alta coherencia entre subsistemas, alta incertidumbre de Heisenberg.
    featureVector: [
      0.85, 1.2, 15, 0.12, 2, 5, 10, 20, 35, 15, 2.0, 1.5,
      1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 3.5, 3, 2.5, 2, 1.5, 1
    ],
    shannonEntropy: 1.2,
    vonNeumannEntropy: 0.8,
    heisenbergUncertainty: 1.4,
    lindbladian: 0.3,
    creationOperator: 0.7,
    annihilationOperator: 0.7,
    parityOperator: -1.0
  },
  {
    id: 'mixed_thermal',
    name: 'Estado Térmico Mixto',
    // Estado mixto con alta entropía, simple estructura espectral.
    // Baja pureza, alta complejidad térmica, alta incertidumbre.
    featureVector: [
      0.3, 2.8, 8, 0.25, 20, 10, 5, 3, 1, 0.5, 0.8, 0.4,
      3.0, 1.0, 0.8, 0.6, 0.4, 0.2, 4, 3.5, 1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    ],
    shannonEntropy: 2.8,
    vonNeumannEntropy: 2.5,
    heisenbergUncertainty: 2.0,
    lindbladian: 0.8,
    creationOperator: 0.2,
    annihilationOperator: 0.2,
    parityOperator: 0.0
  },
  {
    id: 'vacuum_state',
    name: 'Estado de Vacío',
    // Estado fundamental con mínima energía y entropía.
    // Muy baja entropía, mínima incertidumbre, estado puro.
    featureVector: [
      1.0, 0.0, 5, 0.15, 15, 25, 10, 8, 4, 2, 0.5, 0.3,
      0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 2, 2, 2, 2, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1, 1
    ],
    shannonEntropy: 0.0,
    vonNeumannEntropy: 0.0,
    heisenbergUncertainty: 0.5,
    lindbladian: 0.0,
    creationOperator: 0.0,
    annihilationOperator: 1.0,
    parityOperator: 1.0
  },
  {
    id: 'squeezed_state',
    name: 'Estado Comprimido',
    // Estado con incertidumbre reducida en una cuadratura.
    // Moderada entropía, incertidumbre asimétrica de Heisenberg.
    featureVector: [
      0.8, 0.5, 12, 0.18, 8, 18, 15, 12, 8, 3, 1.0, 0.6,
      1.5, 1.8, 1.2, 1.0, 0.8, 0.4, 2.5, 2.2, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08
    ],
    shannonEntropy: 0.5,
    vonNeumannEntropy: 0.3,
    heisenbergUncertainty: 0.3, // Reducida en una dirección
    lindbladian: 0.2,
    creationOperator: 0.9,
    annihilationOperator: 0.9,
    parityOperator: 0.5
  },
  {
    id: 'fock_state',
    name: 'Estado de Fock',
    // Estado con número definido de excitaciones.
    // Baja entropía Shannon, estructura discreta bien definida.
    featureVector: [
      0.9, 0.3, 20, 0.1, 3, 8, 25, 30, 20, 8, 1.5, 1.0,
      2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 3.5, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05
    ],
    shannonEntropy: 0.3,
    vonNeumannEntropy: 0.1,
    heisenbergUncertainty: 0.7,
    lindbladian: 0.15,
    creationOperator: Math.sqrt(5), // Para n=5
    annihilationOperator: Math.sqrt(5),
    parityOperator: -1.0 // Impar
  }
];

// Tipos adicionales para propiedades cuánticas
export interface QuantumProperties {
  // Entropía de Shannon: H = -Σ p_i log(p_i)
  shannonEntropy: number;
  
  // Entropía de von Neumann: S = -Tr(ρ log ρ)
  vonNeumannEntropy: number;
  
  // Principio de incertidumbre de Heisenberg: ΔxΔp ≥ ℏ/2
  heisenbergUncertainty: number;
  
  // Operador de Lindblad (decoherencia)
  lindbladian: number;
  
  // Operador de creación a†
  creationOperator: number;
  
  // Operador de aniquilación a
  annihilationOperator: number;
  
  // Operador de paridad P = (-1)^n
  parityOperator: number;
}

// Función auxiliar para calcular propiedades cuánticas
export function calculateQuantumProperties(state: QuantumState): QuantumProperties {
  return {
    shannonEntropy: state.shannonEntropy,
    vonNeumannEntropy: state.vonNeumannEntropy,
    heisenbergUncertainty: state.heisenbergUncertainty,
    lindbladian: state.lindbladian,
    creationOperator: state.creationOperator,
    annihilationOperator: state.annihilationOperator,
    parityOperator: state.parityOperator
  };
}

// Función para evaluar la pureza del estado cuántico
export function calculatePurity(vonNeumannEntropy: number): number {
  return Math.exp(-vonNeumannEntropy);
}

// Función para verificar el principio de incertidumbre
export function verifyUncertaintyPrinciple(deltaX: number, deltaP: number): boolean {
  const hbar = 1.054571817e-34; // Constante de Planck reducida
  return (deltaX * deltaP) >= (hbar / 2);
}