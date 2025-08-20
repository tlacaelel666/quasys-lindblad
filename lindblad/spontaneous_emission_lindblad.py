def create_spontaneous_emission_system(n_levels: int = 2, 
                                     omega: float = 1.0, 
                                     gamma: float = 0.1,
                                     initial_state: str = "excited") -> SystemParameters:
    """
    Crea un sistema cuántico para modelar la emisión espontánea de un átomo de n niveles.
    
    Args:
        n_levels: Número de niveles energéticos (por defecto 2: ground + excited)
        omega: Frecuencia de transición entre niveles adyacentes
        gamma: Tasa de emisión espontánea (Einstein A coefficient)
        initial_state: Estado inicial ("excited", "ground", "superposition")
    
    Returns:
        SystemParameters: Parámetros del sistema configurado
    """
    
    # Validaciones
    if n_levels < 2:
        raise ValueError("Necesitas al menos 2 niveles para emisión espontánea")
    
    # ========== HAMILTONIANO ==========
    # Hamiltoniano diagonal con energías E_n = n * ℏω (n = 0, 1, 2, ...)
    H = np.zeros((n_levels, n_levels), dtype=complex)
    for n in range(n_levels):
        H[n, n] = n * omega
    
    # ========== OPERADORES DE LINDBLAD ==========
    # Para emisión espontánea, necesitamos operadores de descenso σ⁻
    # que conectan el nivel |n⟩ con |n-1⟩
    L_operators = []
    
    for n in range(1, n_levels):  # Desde nivel 1 hasta n_levels-1
        # Operador de descenso |n-1⟩⟨n|
        sigma_minus = np.zeros((n_levels, n_levels), dtype=complex)
        sigma_minus[n-1, n] = 1.0
        
        # La tasa efectiva incluye la degeneración y factores geométricos
        # Para transiciones espontáneas: γ_eff = γ * n (proporcional al número cuántico)
        gamma_eff = gamma * np.sqrt(n)  # Factor √n por emisión estimulada
        
        L_operators.append(np.sqrt(gamma_eff) * sigma_minus)
    
    # ========== ESTADO INICIAL ==========
    rho_0 = np.zeros((n_levels, n_levels), dtype=complex)
    
    if initial_state == "excited":
        # Todos los átomos en el estado más excitado
        rho_0[n_levels-1, n_levels-1] = 1.0
        
    elif initial_state == "ground":
        # Todos los átomos en el estado fundamental
        rho_0[0, 0] = 1.0
        
    elif initial_state == "superposition":
        # Superposición coherente de todos los estados
        psi = np.ones(n_levels, dtype=complex) / np.sqrt(n_levels)
        rho_0 = np.outer(psi, psi.conj())
        
    elif initial_state == "thermal":
        # Distribución térmica (opcional, para casos avanzados)
        kT = 0.5 * omega  # Temperatura en unidades de energía
        Z = 0  # Función de partición
        for n in range(n_levels):
            Z += np.exp(-n * omega / kT)
        
        for n in range(n_levels):
            rho_0[n, n] = np.exp(-n * omega / kT) / Z
    
    else:
        raise ValueError(f"Estado inicial '{initial_state}' no reconocido")
    
    # ========== OBSERVABLES ==========
    observables = {}
    
    # Poblaciones de cada nivel
    for n in range(n_levels):
        P_n = np.zeros((n_levels, n_levels), dtype=complex)
        P_n[n, n] = 1.0
        observables[f'Población_nivel_{n}'] = P_n
    
    # Número de excitación total
    N_op = np.zeros((n_levels, n_levels), dtype=complex)
    for n in range(n_levels):
        N_op[n, n] = n
    observables['Numero_excitacion'] = N_op
    
    # Operador de inversión de población (para sistemas de 2 niveles)
    if n_levels == 2:
        inversion = np.array([[1, 0], [0, -1]], dtype=complex)
        observables['Inversion_poblacion'] = inversion
        
        # Coherencias (elementos off-diagonal)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        observables['Coherencia_X'] = sigma_x
        observables['Coherencia_Y'] = sigma_y
    
    # Energía promedio del sistema
    observables['Energia_promedio'] = H
    
    # ========== TIEMPO DE SIMULACIÓN ==========
    # El tiempo debe ser del orden de varias vidas medias: τ = 1/γ
    lifetime = 1.0 / gamma if gamma > 0 else 10.0
    t_end = 5 * lifetime  # Simular por 5 vidas medias
    
    return SystemParameters(
        name=f"emision_espontanea_{n_levels}niveles",
        dimension=n_levels,
        H=H,
        L_operators=L_operators,
        rho_0=rho_0,
        t_span=(0, t_end),
        observables=observables
    )


def create_cavity_qed_system() -> SystemParameters:
    """
    Crea un sistema más avanzado: átomo de 2 niveles en cavidad con emisión espontánea
    y acoplamiento al campo de la cavidad.
    """
    
    # Parámetros físicos
    omega_a = 1.0    # Frecuencia atómica
    omega_c = 1.0    # Frecuencia de la cavidad  
    g = 0.1          # Acoplamiento átomo-cavidad
    gamma = 0.01     # Emisión espontánea al continuo
    kappa = 0.05     # Pérdidas de la cavidad
    
    # Número de fotones máximos en la cavidad
    N_cavity = 5
    N_total = 2 * (N_cavity + 1)  # 2 niveles atómicos × (N_cavity + 1) estados cavidad
    
    # Base: |g,n⟩, |e,n⟩ donde g=ground, e=excited, n=photones en cavidad
    
    # ========== HAMILTONIANO ==========
    H = np.zeros((N_total, N_total), dtype=complex)
    
    def get_index(atomic_state, cavity_photons):
        """Convierte (estado_atomico, fotones_cavidad) a índice lineal."""
        return atomic_state * (N_cavity + 1) + cavity_photons
    
    # Términos diagonales: energías libre
    for n in range(N_cavity + 1):
        # |g,n⟩: energía = n * ωc
        idx_g = get_index(0, n)
        H[idx_g, idx_g] = n * omega_c
        
        # |e,n⟩: energía = ωa + n * ωc  
        idx_e = get_index(1, n)
        H[idx_e, idx_e] = omega_a + n * omega_c
    
    # Acoplamiento átomo-cavidad: g(a†σ⁻ + aσ⁺)
    for n in range(N_cavity):
        # Término a†σ⁻: |g,n+1⟩⟨e,n|
        idx_g_n1 = get_index(0, n+1)
        idx_e_n = get_index(1, n)
        H[idx_g_n1, idx_e_n] = g * np.sqrt(n + 1)
        
        # Término aσ⁺: |e,n⟩⟨g,n+1|
        H[idx_e_n, idx_g_n1] = g * np.sqrt(n + 1)
    
    # ========== OPERADORES DE LINDBLAD ==========
    L_operators = []
    
    # 1. Emisión espontánea: σ⁻ (sin cambio en fotones de cavidad)
    L_spontaneous = np.zeros((N_total, N_total), dtype=complex)
    for n in range(N_cavity + 1):
        idx_g = get_index(0, n)
        idx_e = get_index(1, n)
        L_spontaneous[idx_g, idx_e] = 1.0
    
    L_operators.append(np.sqrt(gamma) * L_spontaneous)
    
    # 2. Pérdidas de cavidad: operador a (aniquilación de fotón)
    L_cavity = np.zeros((N_total, N_total), dtype=complex)
    for n in range(1, N_cavity + 1):
        for atomic_state in [0, 1]:
            idx_n = get_index(atomic_state, n)
            idx_n_minus_1 = get_index(atomic_state, n-1)
            L_cavity[idx_n_minus_1, idx_n] = np.sqrt(n)
    
    L_operators.append(np.sqrt(kappa) * L_cavity)
    
    # ========== ESTADO INICIAL ==========
    # Átomo excitado, cavidad vacía: |e,0⟩
    rho_0 = np.zeros((N_total, N_total), dtype=complex)
    initial_idx = get_index(1, 0)  # |e,0⟩
    rho_0[initial_idx, initial_idx] = 1.0
    
    # ========== OBSERVABLES ==========
    observables = {}
    
    # Población atómica excitada
    P_excited = np.zeros((N_total, N_total), dtype=complex)
    for n in range(N_cavity + 1):
        idx = get_index(1, n)
        P_excited[idx, idx] = 1.0
    observables['Poblacion_excitada'] = P_excited
    
    # Número de fotones en cavidad
    N_photons = np.zeros((N_total, N_total), dtype=complex)
    for n in range(N_cavity + 1):
        for atomic_state in [0, 1]:
            idx = get_index(atomic_state, n)
            N_photons[idx, idx] = n
    observables['Fotones_cavidad'] = N_photons
    
    return SystemParameters(
        name="cavity_qed_emission",
        dimension=N_total,
        H=H,
        L_operators=L_operators,
        rho_0=rho_0,
        t_span=(0, 50.0),
        observables=observables
    )


def plot_spontaneous_emission_results(results: Dict, save_path: Optional[str] = None):
    """
    Visualización especializada para resultados de emisión espontánea.
    """
    if not results.get('success', False):
        logger.error("No se pueden graficar resultados fallidos")
        return

    time = results['time']
    observables = results['observables']
    
    # Determinar el tipo de sistema basado en observables disponibles
    is_multilevel = any('nivel' in obs for obs in observables.keys())
    is_cavity_qed = 'Fotones_cavidad' in observables.keys()
    
    if is_cavity_qed:
        # Gráficos específicos para Cavity QED
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Emisión Espontánea - Sistema Átomo-Cavidad', fontsize=16)
        
        # Población excitada vs tiempo
        excited_pop = np.real(observables['Poblacion_excitada'])
        axes[0, 0].plot(time, excited_pop, 'r-', linewidth=2, label='Población excitada')
        axes[0, 0].set_xlabel('Tiempo')
        axes[0, 0].set_ylabel('Probabilidad')
        axes[0, 0].set_title('Decaimiento Atómico')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Fotones en cavidad
        cavity_photons = np.real(observables['Fotones_cavidad'])
        axes[0, 1].plot(time, cavity_photons, 'b-', linewidth=2, label='⟨n⟩')
        axes[0, 1].set_xlabel('Tiempo')
        axes[0, 1].set_ylabel('Número promedio de fotones')
        axes[0, 1].set_title('Fotones en Cavidad')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
    elif is_multilevel:
        # Gráficos para sistema multinivel
        n_levels = sum(1 for obs in observables.keys() if 'nivel' in obs)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Emisión Espontánea - Sistema de {n_levels} Niveles', fontsize=16)
        
        # Poblaciones por nivel
        colors = plt.cm.viridis(np.linspace(0, 1, n_levels))
        for i in range(min(n_levels, 4)):  # Máximo 4 niveles para claridad
            obs_name = f'Población_nivel_{i}'
            if obs_name in observables:
                pop = np.real(observables[obs_name])
                axes[0, 0].plot(time, pop, color=colors[i], 
                              linewidth=2, label=f'Nivel {i}')
        
        axes[0, 0].set_xlabel('Tiempo')
        axes[0, 0].set_ylabel('Probabilidad')
        axes[0, 0].set_title('Poblaciones por Nivel')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Número de excitación total
        if 'Numero_excitacion' in observables:
            excitation = np.real(observables['Numero_excitacion'])
            axes[0, 1].plot(time, excitation, 'g-', linewidth=2, label='⟨N⟩')
            axes[0, 1].set_xlabel('Tiempo')
            axes[0, 1].set_ylabel('Excitación promedio')
            axes[0, 1].set_title('Decaimiento Total')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
    
    else:
        # Gráficos estándar para sistema de 2 niveles
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Emisión Espontánea - Sistema de 2 Niveles', fontsize=16)
        
        # Inversión de población
        if 'Inversion_poblacion' in observables:
            inversion = np.real(observables['Inversion_poblacion'])
            axes[0, 0].plot(time, inversion, 'r-', linewidth=2, label='⟨σz⟩')
            axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0, 0].set_xlabel('Tiempo')
            axes[0, 0].set_ylabel('Inversión de población')
            axes[0, 0].set_title('Decaimiento Exponencial')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Coherencias
        if 'Coherencia_X' in observables and 'Coherencia_Y' in observables:
            coh_x = np.real(observables['Coherencia_X'])
            coh_y = np.real(observables['Coherencia_Y'])
            
            axes[0, 1].plot(time, coh_x, 'b-', linewidth=2, label='⟨σx⟩')
            axes[0, 1].plot(time, coh_y, 'g-', linewidth=2, label='⟨σy⟩')
            axes[0, 1].set_xlabel('Tiempo')
            axes[0, 1].set_ylabel('Coherencia')
            axes[0, 1].set_title('Pérdida de Coherencia')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
    
    # Propiedades generales (pureza y traza) para todos los sistemas
    axes[1, 0].plot(time, results['traces'], 'b-', linewidth=2, label='Traza')
    axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Tiempo')
    axes[1, 0].set_ylabel('Traza')
    axes[1, 0].set_title('Conservación de Probabilidad')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time, results['purities'], 'purple', linewidth=2, label='Pureza')
    axes[1, 1].set_xlabel('Tiempo')
    axes[1, 1].set_ylabel('Tr(ρ²)')
    axes[1, 1].set_title('Evolución de Pureza')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figura guardada en {save_path}")
    
    plt.show()


# Ejemplo de uso para añadir al main()
def demo_spontaneous_emission():
    """Demostración de emisión espontánea."""
    
    print("\n=== DEMOSTRACIONES DE EMISIÓN ESPONTÁNEA ===")
    
    config = SimulationConfig(rtol=1e-9, atol=1e-11)
    simulator = LindladSimulator(config)
    
    # 1. Sistema simple de 2 niveles
    print("\n1. Átomo de 2 niveles con emisión espontánea...")
    simple_params = create_spontaneous_emission_system(
        n_levels=2, 
        omega=1.0, 
        gamma=0.2, 
        initial_state="excited"
    )
    
    results_simple = simulator.simulate(simple_params)
    if results_simple['success']:
        print(f"   ✓ Simulación exitosa")
        plot_spontaneous_emission_results(results_simple)
    
    # 2. Sistema de 3 niveles
    print("\n2. Átomo de 3 niveles (cascada cuántica)...")
    cascade_params = create_spontaneous_emission_system(
        n_levels=3, 
        omega=1.0, 
        gamma=0.15, 
        initial_state="excited"
    )
    
    results_cascade = simulator.simulate(cascade_params)
    if results_cascade['success']:
        print(f"   ✓ Simulación exitosa")
        plot_spontaneous_emission_results(results_cascade)
    
    # 3. Sistema Cavity QED (avanzado)
    print("\n3. Sistema átomo-cavidad (Cavity QED)...")
    cavity_params = create_cavity_qed_system()
    
    results_cavity = simulator.simulate(cavity_params)
    if results_cavity['success']:
        print(f"   ✓ Simulación exitosa")
        plot_spontaneous_emission_results(results_cavity)
    
    return simulator