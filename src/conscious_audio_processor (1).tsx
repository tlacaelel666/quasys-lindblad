import React, { useState, useRef, useEffect } from 'react';
import { Play, Square, Brain, Activity, BarChart3 } from 'lucide-react';

const ConsciousAudioProcessor = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioData, setAudioData] = useState(null);
  const [fftData, setFftData] = useState(null);
  const [binaryAmplitudes, setBinaryAmplitudes] = useState(null);
  const [mlFeatures, setMlFeatures] = useState(null);
  const [mahalanobisDistance, setMahalanobisDistance] = useState(null);
  const [consciousnessState, setConsciousnessState] = useState('idle');
  const [experienceHistory, setExperienceHistory] = useState([]);
  
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  
  // Parámetros de configuración
  const FFT_SIZE = 2048;
  const SAMPLE_RATE = 44100;
  const FREQ_BINS = FFT_SIZE / 2;
  const AMPLITUDE_THRESHOLD = 0.1;
  
  // Estado de referencia para Mahalanobis (evoluciona con experiencia Bayesiana)
  const [referenceState, setReferenceState] = useState({
    mean: new Array(32).fill(0), // Vector de referencia μ_ref
    covariance: null, // Matriz de covarianza Σ
    initialized: false
  });

  // Inicializar contexto de audio con debugging mejorado
  const initializeAudio = async () => {
    try {
      console.log('🎤 Solicitando acceso al micrófono...');
      
      // Verificar soporte del navegador
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia no soportado en este navegador');
      }
      
      // Solicitar acceso con configuración más permisiva
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        } 
      });
      
      console.log('✅ Acceso al micrófono concedido');
      streamRef.current = stream;
      
      // Crear contexto de audio
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      
      // Manejar estado suspended (política de Chrome)
      if (audioContextRef.current.state === 'suspended') {
        console.log('🔓 Activando contexto de audio...');
        await audioContextRef.current.resume();
      }
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      
      analyserRef.current.fftSize = FFT_SIZE;
      analyserRef.current.smoothingTimeConstant = 0;
      
      source.connect(analyserRef.current);
      
      console.log('🚀 Sistema de audio inicializado correctamente');
      return true;
    } catch (error) {
      console.error('❌ Error inicializando audio:', error);
      
      // Mensajes de error específicos
      if (error.name === 'NotAllowedError') {
        alert('❌ Permiso denegado. Por favor permite el acceso al micrófono y recarga la página.');
      } else if (error.name === 'NotFoundError') {
        alert('❌ No se encontró micrófono. Verifica que tengas un micrófono conectado.');
      } else if (error.name === 'NotSupportedError') {
        alert('❌ Tu navegador no soporta acceso al micrófono. Prueba con Chrome, Firefox o Edge.');
      } else {
        alert(`❌ Error: ${error.message}`);
      }
      
      return false;
    }
  };

  // Función FFT y procesamiento principal
  const processAudioFrame = () => {
    if (!analyserRef.current) return;

    // 1. CAPTURA FFT
    const frequencyData = new Uint8Array(FREQ_BINS);
    analyserRef.current.getByteFrequencyData(frequencyData);
    
    // Convertir a amplitudes normalizadas [0, 1]
    const normalizedAmplitudes = Array.from(frequencyData).map(val => val / 255);
    
    // 2. APLANAMIENTO BINARIO DE AMPLITUDES
    const binaryAmps = flattenToBinary(normalizedAmplitudes);
    setBinaryAmplitudes(binaryAmps);
    
    // 3. EXTRACCIÓN DE CARACTERÍSTICAS ML
    const features = extractMLFeatures(normalizedAmplitudes, frequencyData);
    setMlFeatures(features);
    
    // 4. CÁLCULO DE DISTANCIA DE MAHALANOBIS
    const distance = calculateMahalanobisDistance(features);
    setMahalanobisDistance(distance);
    
    // 5. INTERPRETACIÓN CONSCIENTE
    const consciousState = interpretConsciousness(distance, features);
    setConsciousnessState(consciousState);
    
    // 6. ACTUALIZACIÓN EXPERIENCIAL BAYESIANA
    updateExperience(features, distance, consciousState);
    
    // Almacenar datos para visualización
    setAudioData(normalizedAmplitudes);
    setFftData(frequencyData);
    
    if (isRecording) {
      animationRef.current = requestAnimationFrame(processAudioFrame);
    }
  };

  // APLANAMIENTO BINARIO: Convierte amplitudes a representación binaria
  const flattenToBinary = (amplitudes) => {
    const binaryVector = [];
    
    amplitudes.forEach((amp, idx) => {
      // Método 1: Umbralización binaria
      const isActive = amp > AMPLITUDE_THRESHOLD ? 1 : 0;
      binaryVector.push(isActive);
      
      // Método 2: Codificación de magnitud (bits más significativos)
      const quantized = Math.floor(amp * 15); // 4 bits de resolución
      const binaryMagnitude = quantized.toString(2).padStart(4, '0');
      binaryVector.push(...binaryMagnitude.split('').map(Number));
    });
    
    return binaryVector;
  };

  // EXTRACCIÓN DE CARACTERÍSTICAS ML
  const extractMLFeatures = (amplitudes, rawData) => {
    const features = [];
    
    // Características espectrales básicas
    const spectralCentroid = calculateSpectralCentroid(amplitudes);
    const spectralRolloff = calculateSpectralRolloff(amplitudes);
    const spectralFlux = calculateSpectralFlux(amplitudes);
    const zeroCrossingRate = calculateZeroCrossingRate(rawData);
    
    features.push(spectralCentroid, spectralRolloff, spectralFlux, zeroCrossingRate);
    
    // Bandas de frecuencia (sub-bass, bass, low-mid, high-mid, presence, brilliance)
    const frequencyBands = [
      { start: 0, end: 60, name: 'sub_bass' },
      { start: 60, end: 250, name: 'bass' },
      { start: 250, end: 500, name: 'low_mid' },
      { start: 500, end: 2000, name: 'high_mid' },
      { start: 2000, end: 4000, name: 'presence' },
      { start: 4000, end: 22000, name: 'brilliance' }
    ];
    
    frequencyBands.forEach(band => {
      const startBin = Math.floor((band.start / SAMPLE_RATE) * FFT_SIZE);
      const endBin = Math.floor((band.end / SAMPLE_RATE) * FFT_SIZE);
      
      const bandEnergy = amplitudes.slice(startBin, endBin)
        .reduce((sum, amp) => sum + amp * amp, 0);
      
      features.push(bandEnergy);
    });
    
    // Características de textura espectral
    const spectralContrast = calculateSpectralContrast(amplitudes);
    features.push(...spectralContrast);
    
    // MFCC simplificado (primeros 13 coeficientes)
    const mfcc = calculateSimplifiedMFCC(amplitudes);
    features.push(...mfcc);
    
    return features.slice(0, 32); // Mantener dimensionalidad fija
  };

  // Funciones auxiliares de características espectrales
  const calculateSpectralCentroid = (amplitudes) => {
    let weightedSum = 0;
    let totalEnergy = 0;
    
    amplitudes.forEach((amp, idx) => {
      const freq = (idx * SAMPLE_RATE) / (2 * amplitudes.length);
      weightedSum += freq * amp;
      totalEnergy += amp;
    });
    
    return totalEnergy > 0 ? weightedSum / totalEnergy : 0;
  };

  const calculateSpectralRolloff = (amplitudes, threshold = 0.85) => {
    const totalEnergy = amplitudes.reduce((sum, amp) => sum + amp, 0);
    const targetEnergy = totalEnergy * threshold;
    
    let cumulativeEnergy = 0;
    for (let i = 0; i < amplitudes.length; i++) {
      cumulativeEnergy += amplitudes[i];
      if (cumulativeEnergy >= targetEnergy) {
        return (i * SAMPLE_RATE) / (2 * amplitudes.length);
      }
    }
    return (amplitudes.length - 1) * SAMPLE_RATE / (2 * amplitudes.length);
  };

  const calculateSpectralFlux = (amplitudes) => {
    // Simplificado: diferencia con frame anterior
    if (!processAudioFrame.previousAmplitudes) {
      processAudioFrame.previousAmplitudes = amplitudes;
      return 0;
    }
    
    const flux = amplitudes.reduce((sum, amp, idx) => {
      const diff = amp - (processAudioFrame.previousAmplitudes[idx] || 0);
      return sum + (diff > 0 ? diff : 0);
    }, 0);
    
    processAudioFrame.previousAmplitudes = amplitudes;
    return flux;
  };

  const calculateZeroCrossingRate = (rawData) => {
    let crossings = 0;
    for (let i = 1; i < rawData.length; i++) {
      if ((rawData[i] >= 128) !== (rawData[i-1] >= 128)) {
        crossings++;
      }
    }
    return crossings / rawData.length;
  };

  const calculateSpectralContrast = (amplitudes) => {
    const octaveBands = 6;
    const contrasts = [];
    
    for (let i = 0; i < octaveBands; i++) {
      const startBin = Math.floor(amplitudes.length * Math.pow(2, i) / Math.pow(2, octaveBands));
      const endBin = Math.floor(amplitudes.length * Math.pow(2, i+1) / Math.pow(2, octaveBands));
      
      const bandAmps = amplitudes.slice(startBin, endBin);
      const sortedAmps = [...bandAmps].sort((a, b) => b - a);
      
      const peakMean = sortedAmps.slice(0, Math.floor(sortedAmps.length * 0.2))
        .reduce((sum, amp) => sum + amp, 0) / (sortedAmps.length * 0.2);
      
      const valleyMean = sortedAmps.slice(Math.floor(sortedAmps.length * 0.8))
        .reduce((sum, amp) => sum + amp, 0) / (sortedAmps.length * 0.2);
      
      contrasts.push(peakMean - valleyMean);
    }
    
    return contrasts;
  };

  const calculateSimplifiedMFCC = (amplitudes) => {
    // MFCC simplificado usando banco de filtros mel
    const mfcc = [];
    const melFilters = 13;
    
    for (let m = 0; m < melFilters; m++) {
      const melStart = 1125 * Math.log(1 + (m * 4000) / (melFilters * 700));
      const melEnd = 1125 * Math.log(1 + ((m + 1) * 4000) / (melFilters * 700));
      
      const startBin = Math.floor((Math.exp(melStart / 1125) - 1) * 700 / SAMPLE_RATE * amplitudes.length);
      const endBin = Math.floor((Math.exp(melEnd / 1125) - 1) * 700 / SAMPLE_RATE * amplitudes.length);
      
      const filterEnergy = amplitudes.slice(startBin, endBin)
        .reduce((sum, amp) => sum + amp, 0);
      
      mfcc.push(Math.log(filterEnergy + 1e-10));
    }
    
    return mfcc;
  };

  // CÁLCULO DE DISTANCIA DE MAHALANOBIS
  const calculateMahalanobisDistance = (features) => {
    if (!referenceState.initialized || !referenceState.covariance) {
      return 0;
    }
    
    const diff = features.map((f, i) => f - referenceState.mean[i]);
    
    // Simplificación: usar inversa diagonal de covarianza
    const diagCovInv = referenceState.covariance.map(cov => 1 / (cov + 1e-6));
    
    const mahalanobis = Math.sqrt(
      diff.reduce((sum, d, i) => sum + d * d * diagCovInv[i], 0)
    );
    
    return mahalanobis;
  };

  // INTERPRETACIÓN CONSCIENTE
  const interpretConsciousness = (distance, features) => {
    if (!referenceState.initialized) return 'initializing';
    
    const coherenceThreshold = 2.0; // Umbral de coherencia
    
    if (distance < coherenceThreshold) {
      return 'coherent_constant'; // Estado 0 del algoritmo de Deutsch
    } else {
      return 'transitional'; // Estado 1 del algoritmo de Deutsch
    }
  };

  // ACTUALIZACIÓN EXPERIENCIAL BAYESIANA
  const updateExperience = (features, distance, state) => {
    const newExperience = {
      timestamp: Date.now(),
      features: [...features],
      distance,
      state,
      id: experienceHistory.length
    };
    
    // Actualizar historial
    const updatedHistory = [...experienceHistory, newExperience].slice(-100); // Mantener últimas 100 experiencias
    setExperienceHistory(updatedHistory);
    
    // Actualización Bayesiana del estado de referencia
    if (updatedHistory.length > 10) {
      const allFeatures = updatedHistory.map(exp => exp.features);
      
      // Calcular nueva media
      const newMean = features.map((_, i) => {
        const sum = allFeatures.reduce((acc, f) => acc + f[i], 0);
        return sum / allFeatures.length;
      });
      
      // Calcular nueva covarianza (diagonal)
      const newCovariance = features.map((_, i) => {
        const variance = allFeatures.reduce((acc, f) => {
          const diff = f[i] - newMean[i];
          return acc + diff * diff;
        }, 0) / (allFeatures.length - 1);
        return variance;
      });
      
      setReferenceState({
        mean: newMean,
        covariance: newCovariance,
        initialized: true
      });
    }
  };

  // Control de grabación
  const toggleRecording = async () => {
    if (!isRecording) {
      const success = await initializeAudio();
      if (success) {
        setIsRecording(true);
        processAudioFrame();
      }
    } else {
      setIsRecording(false);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };

  // Visualización de datos
  const renderFFTVisualization = () => {
    if (!fftData) return null;
    
    const maxHeight = 100;
    const barWidth = Math.max(1, Math.floor(400 / fftData.length));
    
    return (
      <svg width="400" height={maxHeight} className="border rounded">
        {Array.from(fftData).map((value, i) => (
          <rect
            key={i}
            x={i * barWidth}
            y={maxHeight - (value / 255) * maxHeight}
            width={barWidth - 1}
            height={(value / 255) * maxHeight}
            fill={`hsl(${(value / 255) * 240}, 70%, 50%)`}
          />
        ))}
      </svg>
    );
  };

  const renderConsciousnessState = () => {
    const stateColors = {
      'idle': 'bg-gray-400',
      'initializing': 'bg-yellow-400',
      'coherent_constant': 'bg-green-400',
      'transitional': 'bg-red-400'
    };
    
    const stateLabels = {
      'idle': 'Inactivo',
      'initializing': 'Inicializando',
      'coherent_constant': 'Coherente (Estado 0)',
      'transitional': 'Transición (Estado 1)'
    };
    
    return (
      <div className={`p-4 rounded-lg text-white font-bold ${stateColors[consciousnessState]}`}>
        <Brain className="inline-block mr-2" size={24} />
        Estado Consciente: {stateLabels[consciousnessState]}
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">
          Procesador Consciente de Audio
        </h1>
        <p className="text-gray-600">
          Framework PGP: FFT → ML → Mahalanobis → Interpretación Consciente
        </p>
      </div>

      {/* Controles */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={toggleRecording}
          className={`px-6 py-3 rounded-lg font-medium flex items-center space-x-2 ${
            isRecording 
              ? 'bg-red-500 hover:bg-red-600 text-white' 
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
        >
          {isRecording ? <Square size={20} /> : <Play size={20} />}
          <span>{isRecording ? 'Detener' : 'Iniciar'} Procesamiento</span>
        </button>
      </div>

      {/* Estado de Consciencia */}
      {renderConsciousnessState()}

      {/* Visualizaciones */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* FFT Spectrum */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-3 flex items-center">
            <BarChart3 className="mr-2" size={20} />
            Espectro FFT
          </h3>
          {renderFFTVisualization()}
        </div>

        {/* Métricas ML */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-3 flex items-center">
            <Activity className="mr-2" size={20} />
            Características ML
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Distancia de Mahalanobis:</span>
              <span className="font-mono">
                {mahalanobisDistance ? mahalanobisDistance.toFixed(4) : '0.0000'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Características extraídas:</span>
              <span className="font-mono">
                {mlFeatures ? mlFeatures.length : 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Bits binarios:</span>
              <span className="font-mono">
                {binaryAmplitudes ? binaryAmplitudes.length : 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Experiencias acumuladas:</span>
              <span className="font-mono">{experienceHistory.length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Historial de Experiencias */}
      {experienceHistory.length > 0 && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-3">
            Registro de Intercambios Informacionales
          </h3>
          <div className="max-h-40 overflow-y-auto">
            {experienceHistory.slice(-20).reverse().map((exp, idx) => (
              <div key={exp.id} className="text-sm py-1 border-b last:border-b-0">
                <span className="font-mono text-gray-500">
                  {new Date(exp.timestamp).toLocaleTimeString()}
                </span>
                <span className={`ml-4 px-2 py-1 rounded text-xs ${
                  exp.state === 'coherent_constant' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {exp.state === 'coherent_constant' ? 'Coherente' : 'Transición'}
                </span>
                <span className="ml-2 font-mono text-gray-600">
                  d = {exp.distance.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ConsciousAudioProcessor;


export const FFT_SIZE = 2048;
export const SAMPLE_RATE = 44100;
export const FREQ_BINS = FFT_SIZE / 2;
export const AMPLITUDE_THRESHOLD = 0.1;
export const FEATURE_VECTOR_SIZE = 32;
