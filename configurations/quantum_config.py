#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extensión para Hardware Cuántico Real
Adaptación para trabajar con dispositivos cuánticos reales (IBM, Google, etc.)
"""

import numpy as np
import time
from typing import Dict, List, Optional, Union, Any
from enum import Enum, auto
import logging
from dataclasses import dataclass, field
import asyncio
import json

# Imports para hardware cuántico real
try:
    # IBM Qiskit para hardware IBM
    from qiskit import IBMQ, QuantumCircuit, transpile, execute
    from qiskit.providers.ibmq import least_busy
    from qiskit.providers import Backend
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.ignis.mitigation.measurement import (
        complete_meas_cal, CompleteMeasFitter
    )
    HAS_IBM_BACKEND = True
except ImportError:
    HAS_IBM_BACKEND = False

try:
    # Google Cirq para hardware Google
    import cirq
    import cirq_google
    HAS_GOOGLE_BACKEND = True
except ImportError:
    HAS_GOOGLE_BACKEND = False

try:
    # Amazon Braket
    from braket.aws import AwsDevice
    from braket.devices import LocalSimulator
    from braket.circuits import Circuit
    HAS_BRAKET = True
except ImportError:
    HAS_BRAKET = False

logger = logging.getLogger("QuantumHardwareAdapter")

class HardwareProvider(Enum):
    """Proveedores de hardware cuántico soportados"""
    IBM_QUANTUM = auto()
    GOOGLE_QUANTUM_AI = auto()
    AMAZON_BRAKET = auto()
    RIGETTI_QUANTUM = auto()
    IONQ = auto()
    LOCAL_SIMULATOR = auto()

class DeviceStatus(Enum):
    """Estados posibles de un dispositivo cuántico"""
    AVAILABLE = auto()
    BUSY = auto()
    MAINTENANCE = auto()
    OFFLINE = auto()
    UNKNOWN = auto()

@dataclass
class HardwareSpecs:
    """Especificaciones de hardware cuántico"""
    provider: HardwareProvider
    device_name: str
    num_qubits: int
    connectivity: List[List[int]]  # Qubits conectados físicamente
    gate_times: Dict[str, float]  # Tiempo de ejecución por compuerta (μs)
    error_rates: Dict[str, float]  # Tasas de error por operación
    coherence_time_t1: float  # Tiempo de relajación (μs)
    coherence_time_t2: float  # Tiempo de decoherencia (μs)
    readout_error: float  # Error de medición
    max_shots: int = 8192
    queue_time_estimate: Optional[float] = None  # Tiempo estimado en cola (min)

class QuantumHardwareAdapter:
    """
    Adaptador para ejecutar operaciones cuánticas en hardware real
    """
    
    def __init__(self, provider: HardwareProvider = HardwareProvider.IBM_QUANTUM):
        self.provider = provider
        self.device: Optional[Any] = None
        self.backend: Optional[Backend] = None
        self.device_specs: Optional[HardwareSpecs] = None
        self.noise_model: Optional[NoiseModel] = None
        self.measurement_fitter: Optional[Any] = None
        
    async def initialize_provider(self, credentials: Dict[str, str]) -> bool:
        """
        Inicializa la conexión con el proveedor de hardware
        
        Args:
            credentials: Credenciales específicas del proveedor
                        IBM: {"token": "your_token"}
                        Google: {"project_id": "your_project", "credentials_path": "path"}
                        Braket: {"aws_access_key": "key", "aws_secret_key": "secret"}
        """
        try:
            if self.provider == HardwareProvider.IBM_QUANTUM:
                return await self._init_ibm(credentials)
            elif self.provider == HardwareProvider.GOOGLE_QUANTUM_AI:
                return await self._init_google(credentials)
            elif self.provider == HardwareProvider.AMAZON_BRAKET:
                return await self._init_braket(credentials)
            else:
                logger.error(f"Proveedor {self.provider} no soportado aún")
                return False
        except Exception as e:
            logger.error(f"Error inicializando proveedor {self.provider}: {e}")
            return False
    
    async def _init_ibm(self, credentials: Dict[str, str]) -> bool:
        """Inicializa conexión con IBM Quantum"""
        if not HAS_IBM_BACKEND:
            raise ImportError("Qiskit IBM provider no disponible")
            
        try:
            # Cargar cuenta IBM
            IBMQ.save_account(credentials["token"], overwrite=True)
            IBMQ.load_account()
            
            # Obtener provider
            provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
            
            # Seleccionar dispositivo menos ocupado
            available_devices = provider.backends(
                filters=lambda x: x.configuration().n_qubits >= 5 and 
                                 x.status().operational == True
            )
            
            if not available_devices:
                logger.error("No hay dispositivos IBM disponibles")
                return False
                
            self.backend = least_busy(available_devices)
            logger.info(f"Dispositivo IBM seleccionado: {self.backend.name()}")
            
            # Obtener especificaciones del dispositivo
            await self._load_device_specs_ibm()
            
            # Configurar modelo de ruido
            self.noise_model = NoiseModel.from_backend(self.backend)
            
            # Configurar calibración de medición
            await self._setup_measurement_calibration_ibm()
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando IBM: {e}")
            return False
    
    async def _init_google(self, credentials: Dict[str, str]) -> bool:
        """Inicializa conexión con Google Quantum AI"""
        if not HAS_GOOGLE_BACKEND:
            raise ImportError("Cirq Google no disponible")
            
        try:
            # Configurar autenticación Google Cloud
            engine = cirq_google.Engine(project_id=credentials["project_id"])
            
            # Listar procesadores disponibles
            processors = engine.list_processors()
            if not processors:
                logger.error("No hay procesadores Google disponibles")
                return False
                
            # Seleccionar primer procesador disponible
            self.device = processors[0]
            logger.info(f"Procesador Google seleccionado: {self.device.processor_id}")
            
            await self._load_device_specs_google()
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando Google: {e}")
            return False
    
    async def _init_braket(self, credentials: Dict[str, str]) -> bool:
        """Inicializa conexión con Amazon Braket"""
        if not HAS_BRAKET:
            raise ImportError("Amazon Braket no disponible")
            
        try:
            # Configurar dispositivos Braket
            available_devices = [
                "arn:aws:braket:::device/qpu/ionq/ionQdevice",
                "arn:aws:braket:::device/qpu/rigetti/Aspen-11"
            ]
            
            for device_arn in available_devices:
                try:
                    device = AwsDevice(device_arn)
                    if device.status == "ONLINE":
                        self.device = device
                        logger.info(f"Dispositivo Braket seleccionado: {device.name}")
                        break
                except:
                    continue
                    
            if not self.device:
                logger.warning("No hay dispositivos Braket disponibles, usando simulador")
                self.device = LocalSimulator()
                
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando Braket: {e}")
            return False
    
    async def _load_device_specs_ibm(self):
        """Carga especificaciones del dispositivo IBM"""
        if not self.backend:
            return
            
        config = self.backend.configuration()
        properties = self.backend.properties()
        
        # Extraer conectividad
        connectivity = []
        for gate in config.gates:
            if gate.name in ['cx', 'cnot']:
                connectivity.extend(gate.coupling_map)
        
        # Extraer tasas de error
        error_rates = {}
        gate_times = {}
        
        if properties:
            for gate in properties.gates:
                if gate.gate in error_rates:
                    error_rates[gate.gate] = np.mean([param.value for param in gate.parameters 
                                                    if param.name == 'gate_error'])
                    gate_times[gate.gate] = np.mean([param.value for param in gate.parameters 
                                                   if param.name == 'gate_length']) * 1e6  # a μs
        
        # Obtener tiempos de coherencia promedio
        t1_times = [qubit.t1 for qubit in properties.qubits if qubit.t1]
        t2_times = [qubit.t2 for qubit in properties.qubits if qubit.t2]
        
        self.device_specs = HardwareSpecs(
            provider=HardwareProvider.IBM_QUANTUM,
            device_name=self.backend.name(),
            num_qubits=config.n_qubits,
            connectivity=connectivity,
            gate_times=gate_times,
            error_rates=error_rates,
            coherence_time_t1=np.mean(t1_times) * 1e6 if t1_times else 0,  # a μs
            coherence_time_t2=np.mean(t2_times) * 1e6 if t2_times else 0,
            readout_error=np.mean([qubit.readout_error for qubit in properties.qubits 
                                 if qubit.readout_error]),
            max_shots=config.max_shots
        )
        
        logger.info(f"Especificaciones cargadas: {config.n_qubits} qubits, "
                   f"T1≈{self.device_specs.coherence_time_t1:.1f}μs")
    
    async def _setup_measurement_calibration_ibm(self):
        """Configura calibración de errores de medición para IBM"""
        if not self.backend or self.device_specs is None:
            return
            
        try:
            # Crear circuitos de calibración
            cal_circuits, state_labels = complete_meas_cal(
                range(min(5, self.device_specs.num_qubits)),  # Calibrar hasta 5 qubits
                circlabel='mcal'
            )
            
            # Ejecutar calibración
            cal_job = execute(cal_circuits, self.backend, shots=1024)
            cal_results = cal_job.result()
            
            # Crear objeto de corrección
            self.measurement_fitter = CompleteMeasFitter(cal_results, state_labels)
            
            logger.info("Calibración de medición configurada")
            
        except Exception as e:
            logger.warning(f"Error configurando calibración de medición: {e}")
    
    async def execute_circuit(self, 
                            circuit: QuantumCircuit, 
                            shots: int = 1024,
                            optimize: bool = True) -> Dict[str, Any]:
        """
        Ejecuta un circuito cuántico en hardware real
        
        Args:
            circuit: Circuito cuántico a ejecutar
            shots: Número de mediciones
            optimize: Si optimizar el circuito para el hardware
            
        Returns:
            Resultados de ejecución con correcciones aplicadas
        """
        if not self.backend:
            raise RuntimeError("Hardware no inicializado")
        
        try:
            # 1. Optimizar circuito para el hardware específico
            if optimize:
                optimized_circuit = transpile(
                    circuit, 
                    self.backend,
                    optimization_level=3,  # Máxima optimización
                    scheduling_method='alap'  # As Late As Possible
                )
                logger.info(f"Circuito optimizado: {circuit.depth()} -> {optimized_circuit.depth()} puertas")
            else:
                optimized_circuit = circuit
            
            # 2. Ejecutar en hardware
            job = execute(optimized_circuit, self.backend, shots=shots)
            logger.info(f"Trabajo enviado: {job.job_id()}")
            
            # 3. Monitorear progreso
            await self._monitor_job(job)
            
            # 4. Obtener resultados
            result = job.result()
            counts = result.get_counts(optimized_circuit)
            
            # 5. Aplicar corrección de errores de medición si disponible
            if self.measurement_fitter:
                corrected_counts = self.measurement_fitter.filter.apply(counts)
                logger.info("Corrección de errores de medición aplicada")
            else:
                corrected_counts = counts
            
            # 6. Calcular métricas de calidad
            execution_info = {
                'raw_counts': counts,
                'corrected_counts': corrected_counts,
                'job_id': job.job_id(),
                'execution_time': getattr(result, 'time_taken', None),
                'device': self.backend.name(),
                'shots': shots,
                'circuit_depth': optimized_circuit.depth(),
                'circuit_width': optimized_circuit.width(),
                'success': result.success
            }
            
            return execution_info
            
        except Exception as e:
            logger.error(f"Error ejecutando circuito: {e}")
            raise
    
    async def _monitor_job(self, job, check_interval: int = 30):
        """Monitorea el progreso de un trabajo cuántico"""
        while job.status().name not in ['DONE', 'CANCELLED', 'ERROR']:
            status = job.status()
            queue_info = job.queue_position() if hasattr(job, 'queue_position') else None
            
            logger.info(f"Estado del trabajo: {status.name}")
            if queue_info:
                logger.info(f"Posición en cola: {queue_info}")
                
            await asyncio.sleep(check_interval)
    
    def estimate_execution_time(self, circuit: QuantumCircuit) -> float:
        """
        Estima el tiempo de ejecución de un circuito
        
        Returns:
            Tiempo estimado en segundos
        """
        if not self.device_specs:
            return 0.0
        
        total_time = 0.0
        
        # Sumar tiempos de compuertas
        for instruction in circuit.data:
            gate_name = instruction[0].name
            if gate_name in self.device_specs.gate_times:
                total_time += self.device_specs.gate_times[gate_name]
        
        # Añadir tiempo de medición
        measurement_time = 1.0  # μs por medición (estimado)
        total_time += measurement_time * circuit.num_clbits
        
        return total_time / 1e6  # Convertir a segundos
    
    def get_device_info(self) -> Dict[str, Any]:
        """Obtiene información detallada del dispositivo"""
        if not self.device_specs:
            return {}
        
        return {
            'provider': self.device_specs.provider.name,
            'device_name': self.device_specs.device_name,
            'num_qubits': self.device_specs.num_qubits,
            'connectivity_graph': self.device_specs.connectivity,
            'coherence_times': {
                't1_us': self.device_specs.coherence_time_t1,
                't2_us': self.device_specs.coherence_time_t2
            },
            'error_rates': self.device_specs.error_rates,
            'readout_error': self.device_specs.readout_error,
            'estimated_queue_time_min': self.device_specs.queue_time_estimate
        }

# Extensión de la clase original para hardware real
class RealQuantumMomentumRepresentation:
    """
    Extensión de QuantumMomentumRepresentation para hardware cuántico real
    """
    
    def __init__(self, num_qubits: int, provider: HardwareProvider = HardwareProvider.IBM_QUANTUM):
        self.num_qubits = num_qubits
        self.hardware_adapter = QuantumHardwareAdapter(provider)
        self.circuit_history: List[QuantumCircuit] = []
        self.execution_results: List[Dict[str, Any]] = []
        
    async def initialize_hardware(self, credentials: Dict[str, str]) -> bool:
        """Inicializa la conexión con hardware cuántico"""
        success = await self.hardware_adapter.initialize_provider(credentials)
        if success:
            device_info = self.hardware_adapter.get_device_info()
            logger.info(f"Hardware inicializado: {device_info}")
        return success
    
    async def apply_gate_on_hardware(self, 
                                   gate_name: str, 
                                   target_qubit: int,
                                   control_qubit: Optional[int] = None,
                                   params: Optional[List[float]] = None,
                                   shots: int = 1024) -> Dict[str, Any]:
        """
        Aplica una compuerta cuántica en hardware real y mide el resultado
        
        Note: En hardware real, no podemos mantener el estado cuántico entre operaciones
        como en simulación. Cada operación requiere preparación y medición completa.
        """
        # Crear circuito cuántico
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Preparar estado inicial (si es necesario)
        # En hardware real, siempre empezamos desde |000...0⟩
        
        # Aplicar compuerta solicitada
        if gate_name.lower() == 'h':
            qc.h(target_qubit)
        elif gate_name.lower() in ['cx', 'cnot']:
            if control_qubit is None:
                raise ValueError("CNOT requiere control_qubit")
            qc.cx(control_qubit, target_qubit)
        elif gate_name.lower() == 'rz':
            if not params:
                raise ValueError("RZ requiere parámetro de ángulo")
            qc.rz(params[0], target_qubit)
        # Añadir más compuertas según necesidad
        
        # Añadir mediciones
        qc.measure_all()
        
        # Ejecutar en hardware
        result = await self.hardware_adapter.execute_circuit(qc, shots)
        
        # Guardar historial
        self.circuit_history.append(qc)
        self.execution_results.append(result)
        
        return result
    
    async def create_entangled_state_on_hardware(self, 
                                               qubit_pairs: List[List[int]], 
                                               shots: int = 1024) -> Dict[str, Any]:
        """
        Crea estados entrelazados usando hardware real
        
        Args:
            qubit_pairs: Lista de pares de qubits para entrelazar [[0,1], [2,3], ...]
            shots: Número de mediciones
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Crear estados Bell para cada par
        for pair in qubit_pairs:
            if len(pair) != 2:
                raise ValueError("Cada par debe tener exactamente 2 qubits")
            q1, q2 = pair
            qc.h(q1)      # Hadamard en primer qubit
            qc.cx(q1, q2) # CNOT para entrelazar
        
        qc.measure_all()
        
        result = await self.hardware_adapter.execute_circuit(qc, shots)
        
        # Analizar entrelazamiento en los resultados
        entanglement_metrics = self._analyze_entanglement_from_counts(
            result['corrected_counts'], qubit_pairs
        )
        result['entanglement_analysis'] = entanglement_metrics
        
        self.circuit_history.append(qc)
        self.execution_results.append(result)
        
        return result
    
    def _analyze_entanglement_from_counts(self, 
                                        counts: Dict[str, int], 
                                        qubit_pairs: List[List[int]]) -> Dict[str, float]:
        """
        Analiza entrelazamiento a partir de conteos de medición
        (Aproximación estadística, no equivale a métricas de estado puro)
        """
        metrics = {}
        total_shots = sum(counts.values())
        
        for i, pair in enumerate(qubit_pairs):
            q1, q2 = pair
            
            # Contar coincidencias (ambos 0 o ambos 1)
            coincidences = 0
            for bitstring, count in counts.items():
                if bitstring[q1] == bitstring[q2]:  # Mismo estado
                    coincidences += count
            
            # Calcular correlación simple
            correlation = (2 * coincidences / total_shots) - 1
            metrics[f'pair_{q1}_{q2}_correlation'] = correlation
        
        return metrics
    
    async def benchmark_hardware(self) -> Dict[str, Any]:
        """
        Ejecuta benchmarks básicos del hardware cuántico
        """
        benchmarks = {}
        
        # 1. Test de compuerta Hadamard
        qc_h = QuantumCircuit(1, 1)
        qc_h.h(0)
        qc_h.measure_all()
        
        h_result = await self.hardware_adapter.execute_circuit(qc_h, shots=1024)
        h_counts = h_result['corrected_counts']
        
        # Verificar distribución 50-50
        prob_0 = h_counts.get('0', 0) / 1024
        prob_1 = h_counts.get('1', 0) / 1024
        h_fidelity = 1 - abs(0.5 - prob_0) - abs(0.5 - prob_1)
        benchmarks['hadamard_fidelity'] = h_fidelity
        
        # 2. Test de estado Bell
        qc_bell = QuantumCircuit(2, 2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)
        qc_bell.measure_all()
        
        bell_result = await self.hardware_adapter.execute_circuit(qc_bell, shots=1024)
        bell_counts = bell_result['corrected_counts']
        
        # Verificar correlaciones Bell
        correlated = bell_counts.get('00', 0) + bell_counts.get('11', 0)
        uncorrelated = bell_counts.get('01', 0) + bell_counts.get('10', 0)
        bell_correlation = (correlated - uncorrelated) / 1024
        benchmarks['bell_correlation'] = bell_correlation
        
        # 3. Información del dispositivo
        benchmarks['device_info'] = self.hardware_adapter.get_device_info()
        
        return benchmarks

# Ejemplo de uso con hardware real
async def example_real_hardware():
    """Ejemplo de uso con hardware cuántico real"""
    
    # Configurar credenciales (¡NO incluir credenciales reales en código!)
    credentials = {
        "token": "YOUR_IBM_TOKEN_HERE"  # Reemplazar con token real
    }
    
    # Crear instancia para hardware real
    real_quantum = RealQuantumMomentumRepresentation(
        num_qubits=5, 
        provider=HardwareProvider.IBM_QUANTUM
    )
    
    try:
        # Inicializar conexión
        logger.info("Inicializando conexión con hardware IBM...")
        success = await real_quantum.initialize_hardware(credentials)
        
        if not success:
            logger.error("No se pudo conectar con hardware")
            return
        
        # Ejecutar benchmarks
        logger.info("Ejecutando benchmarks...")
        benchmark_results = await real_quantum.benchmark_hardware()
        print("Resultados de benchmark:")
        print(json.dumps(benchmark_results, indent=2))
        
        # Crear estado entrelazado
        logger.info("Creando estado entrelazado...")
        entangled_result = await real_quantum.create_entangled_state_on_hardware(
            qubit_pairs=[[0, 1], [2, 3]], 
            shots=2048
        )
        
        print("Estado entrelazado creado:")
        print(f"Conteos: {entangled_result['corrected_counts']}")
        print(f"Análisis: {entangled_result['entanglement_analysis']}")
        
    except Exception as e:
        logger.error(f"Error en ejemplo: {e}")

if __name__ == "__main__":
    # Ejecutar ejemplo (requiere credenciales reales)
    # asyncio.run(example_real_hardware())
    print("Código de adaptación para hardware cuántico real preparado.")
    print("Para usar, configurar credenciales reales y descomentar la línea anterior.")