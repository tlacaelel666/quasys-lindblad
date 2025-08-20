from flask import request, jsonify
from . import beta_decay_blueprint
from .lindblad_beta_decay import BetaDecayAnalyzer, BetaDecayConfig

@beta_decay_blueprint.route('/analyze', methods=['POST'])
def analyze_beta_decay():
    """
    Analyzes beta decay using the Lindblad master equation based on provided configuration.
    Expects a JSON body with BetaDecayConfig parameters.
    """
    config_data = request.get_json()

    if config_data:
        try:
            # Create BetaDecayConfig from JSON data
            config = BetaDecayConfig(**config_data)
        except TypeError as e:
            return jsonify({"error": f"Invalid configuration parameters: {e}"}), 400
    else:
        # Use default configuration if no JSON body is provided
        config = BetaDecayConfig()

    analyzer = BetaDecayAnalyzer(config)
    results = analyzer.run_complete_analysis()

    # Prepare results for JSON serialization (convert numpy arrays etc.)
    serializable_results = {
        "times": results["times"].tolist(),
        "evolved_states": [state.tolist() for state in results["evolved_states"]],
        "entropies": results["entropies"],
        "fidelities": results["fidelities"],
        "initial_state": results["initial_state"].tolist(),
        "target_state": results["target_state"].tolist(),
        "hamiltonian_eigenvals": results["hamiltonian_eigenvals"].tolist(),
        "lindblad_operators": [op.tolist() for op in results["lindblad_operators"]],
        "observables": {k: v for k, v in results["observables"].items()}, # Observables should already be lists of floats
        "config": results["config"].__dict__
    }


    return jsonify(serializable_results)
