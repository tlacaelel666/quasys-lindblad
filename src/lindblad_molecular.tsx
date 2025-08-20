import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Area, AreaChart } from 'recharts';
import { Play, Pause, RotateCcw, Settings, Info, Zap, Mic, MicOff, Volume2, Waves, Atom } from 'lucide-react';

// Constantes físicas
const SOUND_SPEED = 343; // m/s
const AIR_DENSITY = 1.225; // kg/m³
const BOLTZMANN = 1.381e-23;
const FFT_SIZE = 1024;

// Clase para moléculas cuánticas
class QuantumMolecule {
  constructor(position, velocity, quantumState) {
    this.position = position;
    this.velocity = velocity;
    this.quantumState = quantumState;
    this.phase = 0;
    this.coherenceTime = Math.random() * 1000;
  }

  updateQuantumState(pressureWave, temperature = 293) {
    const pressureEffect = pressureWave * 1e-6;
    const thermalEffect = Math.sqrt(BOLTZMANN * temperature);
    
    this.phase += pressureEffect * (this.quantumState.real + this.quantumState.imag);
    const decoherence = Math.exp(-Math.random() * thermalEffect);
    
    return {
      real: this.quantumState.real * Math.cos(this.phase) * decoherence,
      imag: this.quantumState.imag * Math.sin(this.phase) * decoherence,
      probability: Math.abs(this.quantumState.real)**2 + Math.abs(this.quantumState.imag)**2
    };
  }
}

// Sistema acusto-cuántico
class AcoustoQuantumSystem {
  constructor(numMolecules = 500) {
    this.molecules = [];
    this.pressureField = [];
    this.temperature = 293;
    this.initializeMolecules(numMolecules);
  }

  initializeMolecules(num) {
    this.molecules = [];
    for (let i = 0; i < num; i++) {
      const position = [Math.random() * 100, Math.random() * 100, Math.random() * 100];
      const velocity = [(Math.random() - 0.5) * 500, (Math.random() - 0.5) * 500, (Math.random() - 0.5) * 500];
      const quantumState = { real: Math.random() - 0.5, imag: Math.random() - 0.5 };
      
      this.molecules.push(new QuantumMolecule(position, velocity, quantumState));
    }
  }

  updatePressureField(audioData, frequency, time) {
    const wavelength = SOUND_SPEED / frequency;
    this.pressureField = [];
    
    for (let x = 0; x < 100; x += 2) {
      const amplitude = audioData.reduce((sum, val, idx) => {
        const k = 2 * Math.PI * idx / wavelength;
        return sum + val * Math.sin(k * x + 2 * Math.PI * frequency * time);
      }, 0);
      
      this.pressureField.push({
        position: x,
        pressure: amplitude * 100,
        amplitude: amplitude
      });
    }
  }

  evolveQuantumMolecules(dt = 0.01) {
    const quantumCoherence = [];
    const quantumEntanglement = [];
    const molecularDensity = [];
    
    this.molecules.forEach((molecule, idx) => {
      const x_idx = Math.floor(molecule.position[0] / 2);
      const localPressure = this.pressureField[x_idx]?.pressure || 0;
      
      const newQuantumState = molecule.updateQuantumState(localPressure, this.temperature);
      molecule.quantumState = newQuantumState;
      
      const coherence = Math.sqrt(newQuantumState.real**2 + newQuantumState.imag**2);
      quantumCoherence.push(coherence);
      
      if (idx > 0) {
        const neighbor = this.molecules[idx - 1];
        const distance = Math.sqrt(
          (molecule.position[0] - neighbor.position[0])**2 +
          (molecule.position[1] - neighbor.position[1])**2
        );
        const entanglement = Math.exp(-distance / 10) * coherence;
        quantumEntanglement.push(entanglement);
      }
    });

    for (let x = 0; x < 100; x += 10) {
      const moleculesInRegion = this.molecules.filter(m => 
        m.position[0] >= x && m.position[0] < x + 10
      ).length;
      
      molecularDensity.push({
        position: x,
        density: moleculesInRegion,
        quantumDensity: moleculesInRegion * this.getAverageCoherence(x, x + 10)
      });
    }

    return {
      coherence: quantumCoherence.reduce((a, b) => a + b, 0) / quantumCoherence.length,
      entanglement: quantumEntanglement.length > 0 ? quantumEntanglement.reduce((a, b) => a + b, 0) / quantumEntanglement.length : 0,
      molecularDensity: molecularDensity,
      totalQuantumInformation: this.calculateQuantumInformation()
    };
  }

  getAverageCoherence(xStart, xEnd) {
    const relevantMolecules = this.molecules.filter(m => 
      m.position[0] >= xStart && m.position[0] < xEnd
    );
    
    if (relevantMolecules.length === 0) return 0;
    
    return relevantMolecules.reduce((sum, molecule) => {
      return sum + Math.sqrt(molecule.quantumState.real**2 + molecule.quantumState.imag**2);
    }, 0) / relevantMolecules.length;
  }

  calculateQuantumInformation() {
    const totalStates = this.molecules.length;
    let entropy = 0;
    
    this.molecules.forEach(molecule => {
      const p = molecule.quantumState.probability || 0.01;
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    });
    
    return entropy / totalStates;
  }
}

// Funciones de análisis de audio
const calculateSpectralCentroid = (amplitudes, sampleRate) => {
  let weightedSum = 0;
  let totalEnergy = 0;
  amplitudes.forEach((amp, idx) => {
    const freq = (idx * sampleRate) / (2 * amplitudes.length);
    weightedSum += freq * amp;
    totalEnergy += amp;
  });
  return totalEnergy > 0 ? weightedSum / totalEnergy : 0;
};

const extractAudioFeatures = (amplitudes, sampleRate) => {
  const features = [];
  features.push(calculateSpectralCentroid(amplitudes, sampleRate));
  
  const frequencyBands = [
    { start: 0, end: 60 }, { start: 60, end: 250 }, { start: 250, end: 500 },
    { start: 500, end: 2000 }, { start: 2000, end: 4000 }, { start: 4000, end: 8000 }
  ];
  
  frequencyBands.forEach(band => {
    const startBin = Math.floor((band.start / sampleRate) * FFT_SIZE);
    const endBin = Math.min(amplitudes.length, Math.floor((band.end / sampleRate) * FFT_SIZE));
    const bandEnergy = amplitudes.slice(startBin, endBin).reduce((sum, amp) => sum + amp * amp, 0);
    features.push(Math.sqrt(bandEnergy));
  });

  return features;
};

// Componente principal
const AcoustoQuantumLindladSimulator = () => {
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationData, setSimulationData] = useState(null);
  const [animationStep, setAnimationStep] = useState(0);
  const [showInfo, setShowInfo] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [audioFeatures, setAudioFeatures] = useState(null);
  const [spectralData, setSpectralData] = useState([]);
  const [molecularData, setMolecularData] = useState([]);
  
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const microphoneRef = useRef(null);
  const animationRef = useRef(null);
  const systemRef = useRef(null);
  
  const [parameters, setParameters] = useState({
    numMolecules: 500,
    temperature: 293,
    couplingStrength: 1.0,
    decoherenceRate: 0.1,
    timeEnd: 10
  });

  // Inicializar sistema
  useEffect(() => {
    systemRef.current = new AcoustoQuantumSystem(parameters.numMolecules);
  }, [parameters.numMolecules]);

  const simulateAcoustoQuantumSystem = useCallback((params, audioData = null) => {
    if (!systemRef.current) return { data: [], observables: [], title: '', description: '' };
    
    const { numMolecules, temperature, couplingStrength, timeEnd } = params;
    const nPoints = 100;
    const dt = timeEnd / nPoints;
    const data = [];
    
    systemRef.current.temperature = temperature;
    
    for (let i = 0; i <= nPoints; i++) {
      const t = i * dt;
      
      let currentAudioData = audioData || [Math.sin(2 * Math.PI * 440 * t)];
      const dominantFreq = audioFeatures?.dominantFreq || 440;
      
      systemRef.current.updatePressureField(currentAudioData, dominantFreq, t);
      const quantumMetrics = systemRef.current.evolveQuantumMolecules(dt);
      
      const avgPressure = systemRef.current.pressureField.reduce((sum, p) => sum + Math.abs(p.pressure), 0) / systemRef.current.pressureField.length;
      const thermalDecoherence = Math.exp(-0.001 * temperature * t);
      
      data.push({
        time: t,
        pressure: avgPressure / 100,
        quantumCoherence: quantumMetrics.coherence * thermalDecoherence,
        quantumEntanglement: quantumMetrics.entanglement,
        quantumInformation: quantumMetrics.totalQuantumInformation,
        molecularDensity: quantumMetrics.molecularDensity.reduce((sum, d) => sum + d.density, 0) / quantumMetrics.molecularDensity.length,
        thermalEffect: thermalDecoherence,
        couplingEffect: couplingStrength * avgPressure / 1000
      });
      
      if (i === Math.floor(nPoints * 0.8)) {
        setMolecularData(quantumMetrics.molecularDensity);
      }
    }
    
    return {
      data,
      observables: ['pressure', 'quantumCoherence', 'quantumEntanglement', 'quantumInformation'],
      title: 'Sistema Acusto-Cuántico Molecular',
      description: `${numMolecules} moléculas cuánticas en ondas de presión acústicas`
    };
  }, [audioFeatures, parameters]);

  const startAudioAnalysis = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      microphoneRef.current = audioContextRef.current.createMediaStreamSource(stream);
      
      analyserRef.current.fftSize = FFT_SIZE;
      microphoneRef.current.connect(analyserRef.current);
      
      setIsListening(true);
      
      const analyze = () => {
        if (!analyserRef.current || !isListening) return;
        
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        
        const amplitudes = Array.from(dataArray).map(val => val / 255);
        const features = extractAudioFeatures(amplitudes, audioContextRef.current.sampleRate);
        
        const spectralPoints = amplitudes.map((amp, idx) => ({
          frequency: (idx * audioContextRef.current.sampleRate) / (2 * amplitudes.length),
          amplitude: amp,
          quantumCoupling: amp * 0.1
        })).filter(point => point.frequency < 8000);
        
        setSpectralData(spectralPoints);
        
        setAudioFeatures({
          spectralCentroid: features[0] || 0,
          bandEnergies: features.slice(1, 7),
          totalEnergy: amplitudes.reduce((sum, amp) => sum + amp * amp, 0),
          dominantFreq: spectralPoints.reduce((max, point) => 
            point.amplitude > max.amplitude ? point : max, { amplitude: 0, frequency: 440 }).frequency,
          rawAmplitudes: amplitudes
        });
        
        animationRef.current = requestAnimationFrame(analyze);
      };
      
      analyze();
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setIsListening(false);
    }
  }, [isListening]);

  const stopAudioAnalysis = useCallback(() => {
    setIsListening(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    setAudioFeatures(null);
    setSpectralData([]);
  }, []);

  const runSimulation = useCallback(async () => {
    setIsSimulating(true);
    setAnimationStep(0);
    
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const audioData = audioFeatures?.rawAmplitudes || null;
    const results = simulateAcoustoQuantumSystem(parameters, audioData);
    
    setSimulationData(results);
    setIsSimulating(false);
    
    const animateData = () => {
      setAnimationStep(prev => {
        if (prev < results.data.length - 1) {
          setTimeout(animateData, 100);
          return prev + 1;
        }
        return prev;
      });
    };
    setTimeout(animateData, 100);
  }, [parameters, audioFeatures, simulateAcoustoQuantumSystem]);

  const resetSimulation = () => {
    setSimulationData(null);
    setAnimationStep(0);
    setMolecularData([]);
  };

  const getDisplayData = () => {
    if (!simulationData) return [];
    return simulationData.data.slice(0, animationStep + 1);
  };

  const getColors = () => {
    return ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899'];
  };

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-800 min-h-screen text-white">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
          Sistema Acusto-Cuántico Molecular
        </h1>
        <p className="text-slate-300 text-lg">
          Estados Cuánticos Encapsulados en Ondas de Presión
        </p>
        <div className="text-sm text-slate-400 mt-2">
          Moléculas como Portadoras de Información Cuántica • por Jacobo Tlacaelel Mina Rodriguez
        </div>
      </div>

      {/* Panel de Control */}
      <div className="bg-slate-800/80 backdrop-blur rounded-xl p-6 mb-6 border border-slate-700">
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Waves className="w-6 h-6 text-blue-400" />
              <Atom className="w-4 h-4 text-purple-400 absolute -top-1 -right-1" />
            </div>
            <span className="font-semibold text-xl">Sistema Acusto-Cuántico</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Moléculas Cuánticas
            </label>
            <input
              type="range"
              min="100"
              max="1000"
              step="50"
              value={parameters.numMolecules}
              onChange={(e) => setParameters(prev => ({
                ...prev,
                numMolecules: parseInt(e.target.value)
              }))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="text-xs text-slate-400 mt-1">
              {parameters.numMolecules} moléculas
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Temperatura (K)
            </label>
            <input
              type="range"
              min="200"
              max="400"
              step="10"
              value={parameters.temperature}
              onChange={(e) => setParameters(prev => ({
                ...prev,
                temperature: parseInt(e.target.value)
              }))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="text-xs text-slate-400 mt-1">
              {parameters.temperature}K
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Acoplamiento
            </label>
            <input
              type="range"
              min="0.1"
              max="5.0"
              step="0.1"
              value={parameters.couplingStrength}
              onChange={(e) => setParameters(prev => ({
                ...prev,
                couplingStrength: parseFloat(e.target.value)
              }))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="text-xs text-slate-400 mt-1">
              λ = {parameters.couplingStrength}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Decoherencia
            </label>
            <input
              type="range"
              min="0.01"
              max="1.0"
              step="0.01"
              value={parameters.decoherenceRate}
              onChange={(e) => setParameters(prev => ({
                ...prev,
                decoherenceRate: parseFloat(e.target.value)
              }))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="text-xs text-slate-400 mt-1">
              γ = {parameters.decoherenceRate}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={runSimulation}
            disabled={isSimulating}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 disabled:from-gray-500 disabled:to-gray-600 text-white font-semibold rounded-lg transition-all duration-200 disabled:cursor-not-allowed"
          >
            {isSimulating ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            {isSimulating ? 'Evolucionando...' : 'Evolucionar Sistema'}
          </button>

          <button
            onClick={resetSimulation}
            className="flex items-center gap-2 px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-all duration-200"
          >
            <RotateCcw className="w-5 h-5" />
            Reset
          </button>

          <button
            onClick={isListening ? stopAudioAnalysis : startAudioAnalysis}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
              isListening
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            {isListening ? 'Detener Audio' : 'Iniciar Audio'}
          </button>

          <button
            onClick={() => setShowInfo(!showInfo)}
            className="flex items-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-all duration-200"
          >
            <Info className="w-5 h-5" />
            Teoría
          </button>
        </div>
      </div>

      {/* Panel de Información */}
      {showInfo && (
        <div className="bg-slate-800/80 backdrop-blur rounded-xl p-6 mb-6 border border-slate-700">
          <h3 className="text-2xl font-bold mb-4 text-purple-400">
            Teoría del Sistema Acusto-Cuántico Molecular
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-blue-400 mb-3">Modelo Físico:</h4>
              <div className="space-y-2 text-sm text-slate-300">
                <p>• <strong>Moléculas Cuánticas:</strong> Cada molécula del aire porta un estado cuántico |ψ⟩ = α|0⟩ + β|1⟩</p>
                <p>• <strong>Acoplamiento:</strong> Las ondas de presión modulan los coeficientes cuánticos</p>
                <p>• <strong>Hamiltoniano:</strong> H = H₀ + λP(x,t)σz donde P(x,t) es el campo de presión</p>
                <p>• <strong>Decoherencia:</strong> El movimiento molecular causa pérdida de coherencia</p>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-green-400 mb-3">Observables:</h4>
              <div className="space-y-2 text-sm text-slate-300">
                <p>• <strong>Coherencia:</strong> ⟨ψ|ψ⟩ promedio del ensemble molecular</p>
                <p>• <strong>Entrelazamiento:</strong> Correlaciones cuánticas entre moléculas vecinas</p>
                <p>• <strong>Información Cuántica:</strong> Entropía de von Neumann del sistema</p>
                <p>• <strong>Densidad Molecular:</strong> Distribución de portadores cuánticos</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Análisis de Audio */}
      {isListening && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-slate-800/80 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-bold mb-4 text-blue-400">Espectro Acústico</h3>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={spectralData.slice(0, 50)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="frequency" stroke="#9CA3AF" fontSize={10} />
                <YAxis stroke="#9CA3AF" fontSize={10} />
                <Area type="monotone" dataKey="amplitude" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-slate-800/80 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-bold mb-4 text-green-400">Métricas de Audio</h3>
            {audioFeatures && (
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span>Centroide Espectral:</span>
                  <span className="font-mono text-blue-400">
                    {audioFeatures.spectralCentroid.toFixed(1)} Hz
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Freq. Dominante:</span>
                  <span className="font-mono text-purple-400">
                    {audioFeatures.dominantFreq.toFixed(1)} Hz
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Energía Total:</span>
                  <span className="font-mono text-orange-400">
                    {(audioFeatures.totalEnergy * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Visualización Principal */}
      {simulationData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-slate-800/80 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-bold mb-4 text-center text-purple-400">
              {simulationData.title}
            </h3>
            <p className="text-sm text-slate-400 text-center mb-4">
              {simulationData.description}
            </p>
            
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={getDisplayData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                
                {simulationData.observables.map((obs, index) => (
                  <Line
                    key={obs}
                    type="monotone"
                    dataKey={obs}
                    stroke={getColors()[index]}
                    strokeWidth={2}
                    dot={false}
                    name={obs.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-slate-800/80 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-bold mb-4 text-orange-400">Distribución Molecular</h3>
            
            {molecularData.length > 0 && (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={molecularData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="position" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="density" fill="#F59E0B" name
