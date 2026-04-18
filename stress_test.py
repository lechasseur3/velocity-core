"""
Stress Testing Module for Velocity Core Portfolio Engine
=========================================================
Scénarios de stress : crash 2008, hausse de taux, stagflation, black swan.
Chaque scénario calcule VaR, drawdown max, et perte en € sur le portefeuille.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# --- Scénarios prédéfinis ---

STRESS_SCENARIOS = {
    "crash_2008": {
        "name": "Crash 2008 (GFC)",
        "description": "Réplication du choc systémique de 2008-2009 : -40% marché actions, corrélation montante, volatilité x3",
        "equity_shock": -0.40,
        "vol_multiplier": 3.0,
        "correlation_floor": 0.85,
        "duration_days": 252,
        "rate_shock_bps": -150,  # taux baissent (flight to quality)
    },
    "rate_hike_300": {
        "name": "Hausse de taux (+300bps)",
        "description": "Choc de taux +300bps sur 6 mois : pression sur les valorisations, credit spread widening",
        "equity_shock": -0.15,
        "vol_multiplier": 1.8,
        "correlation_floor": 0.70,
        "duration_days": 126,
        "rate_shock_bps": 300,
    },
    "stagflation": {
        "name": "Stagflation",
        "description": "Croissance nulle + inflation persistante 6-8% : erosion des marges, taux réels négatifs",
        "equity_shock": -0.25,
        "vol_multiplier": 2.0,
        "correlation_floor": 0.75,
        "duration_days": 504,  # 2 ans
        "rate_shock_bps": 200,
    },
    "black_swan": {
        "name": "Black Swan (-60% sur 1 mois)",
        "description": "Événement extrême et imprévisible : -60% en 21 jours, liquidité évaporée",
        "equity_shock": -0.60,
        "vol_multiplier": 5.0,
        "correlation_floor": 0.95,
        "duration_days": 21,
        "rate_shock_bps": 0,
    },
}


def stress_var(
    portfolio_value_eur: float,
    expected_return_ann: float,
    volatility_ann: float,
    scenario: dict,
    confidence: float = 0.99,
) -> Dict[str, float]:
    """
    Calcule la VaR stressée pour un scénario donné.

    Returns dict with:
      - stressed_var_99: VaR 99% sous le scénario (en €)
      - stressed_cvar_99: CVaR 99% sous le scénario (en €)
      - max_drawdown: drawdown maximum estimé (en %)
      - loss_eur: perte estimée en €
      - stressed_vol: volatilité annualisée stressée
      - stressed_return: rendement annualisé stressé
    """
    from scipy import stats

    vol_stressed = volatility_ann * scenario["vol_multiplier"]

    # On travaille en rendement total sur la durée du scénario
    T = scenario["duration_days"]
    total_return_shocked = scenario["equity_shock"]
    vol_over_period = vol_stressed * np.sqrt(T / 252)

    z = stats.norm.ppf(confidence)

    # VaR paramétrique sur l'horizon T
    var_scenario = -(total_return_shocked - z * vol_over_period)
    var_scenario = max(0, min(var_scenario, 1.0))

    # CVaR paramétrique
    phi_z = stats.norm.pdf(z)
    cvar_scenario = var_scenario + vol_over_period * phi_z / (1 - confidence)
    cvar_scenario = max(0, min(cvar_scenario, 1.0))

    # Rendement annualisé pour reporting
    ret_stressed_annualized = (1 + total_return_shocked) ** (252 / T) - 1

    # Max drawdown estimé = choc equity ajusté pour la corrélation montante
    max_dd = abs(scenario["equity_shock"]) * scenario["correlation_floor"]
    max_dd = min(max_dd, 0.95)  # cap à 95%

    # Perte en €
    loss_eur = portfolio_value_eur * abs(scenario["equity_shock"])

    # VaR et CVaR en €
    var_eur = portfolio_value_eur * var_scenario
    cvar_eur = portfolio_value_eur * cvar_scenario

    return {
        "scenario_name": scenario["name"],
        "stressed_return_ann": round(ret_stressed_annualized, 4),
        "stressed_vol_ann": round(vol_stressed, 4),
        "var_99_pct": round(var_scenario, 4),
        "var_99_eur": round(var_eur, 2),
        "cvar_99_pct": round(cvar_scenario, 4),
        "cvar_99_eur": round(cvar_eur, 2),
        "max_drawdown_pct": round(max_dd, 4),
        "loss_eur": round(loss_eur, 2),
        "duration_days": scenario["duration_days"],
    }


def stress_test_portfolio(
    portfolio_value_eur: float,
    expected_return_ann: float,
    volatility_ann: float,
    scenarios: Optional[Dict] = None,
    confidence: float = 0.99,
) -> Dict:
    """
    Lance un stress test complet du portefeuille sur tous les scénarios.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    results = {}
    for key, scenario in scenarios.items():
        results[key] = stress_var(
            portfolio_value_eur, expected_return_ann, volatility_ann, scenario, confidence
        )

    # Résumé : pire scénario
    worst = max(results.items(), key=lambda x: x[1]["loss_eur"])

    return {
        "portfolio_value_eur": portfolio_value_eur,
        "base_expected_return": expected_return_ann,
        "base_volatility": volatility_ann,
        "confidence": confidence,
        "scenarios": results,
        "worst_scenario": worst[0],
        "worst_loss_eur": worst[1]["loss_eur"],
    }


def stress_test_from_analysis(analysis_result: Dict, portfolio_value_eur: float = 100000) -> Dict:
    """
    Interface pratique : prend le résultat de run_analysis() et lance le stress test.
    """
    perf = analysis_result.get("performance", {})
    expected_return = perf.get("expected_return", 0.08)
    volatility = perf.get("volatility", 0.15)

    return stress_test_portfolio(portfolio_value_eur, expected_return, volatility)


# --- CLI rapide ---

if __name__ == "__main__":
    # Demo avec un portefeuille de 100k€, rendement 8%, vol 15%
    result = stress_test_portfolio(100000, 0.08, 0.15)

    print("=" * 60)
    print("STRESS TEST — Velocity Core Portfolio Engine")
    print("=" * 60)
    print(f"Valeur portefeuille : {result['portfolio_value_eur']:,.0f} €")
    print(f"Rendement de base   : {result['base_expected_return']:.1%}")
    print(f"Volatilité de base   : {result['base_volatility']:.1%}")
    print()

    for key, s in result["scenarios"].items():
        print(f"--- {s['scenario_name']} ---")
        print(f"  Rendement stressé : {s['stressed_return_ann']:.1%}")
        print(f"  Volatilité stressée: {s['stressed_vol_ann']:.1%}")
        print(f"  VaR 99%           : {s['var_99_pct']:.2%} = {s['var_99_eur']:,.0f} €")
        print(f"  CVaR 99%          : {s['cvar_99_pct']:.2%} = {s['cvar_99_eur']:,.0f} €")
        print(f"  Max Drawdown      : {s['max_drawdown_pct']:.1%}")
        print(f"  Perte estimée     : {s['loss_eur']:,.0f} €")
        print(f"  Durée             : {s['duration_days']} jours")
        print()

    print(f"⚠️  Pire scénario : {result['worst_scenario']} — perte {result['worst_loss_eur']:,.0f} €")