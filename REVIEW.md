# REVIEW.md — Velocity Core : Revue Financière

**Date :** 2026-04-18  
**Auteur :** Paul (agent business/finance)

---

## 1. Black-Litterman — Validation

**Verdict : Implémentation correcte, avec réserves mineures.**

- La lib `pypfopt.BlackLittermanModel` est utilisée avec `pi="market"`, `market_caps`, `risk_aversion=2.5`, `tau=0.05`. C'est conforme au modèle standard de He & Litterman.
- Les vues absolues (A) et relatives (R) sont correctement construites en matrices P/Q.
- **Point d'attention :** Quand il n'y a pas de vues, le code tombe sur `market_implied_prior_returns` seul — c'est un fallback légitime mais le portefeuille revient alors au market-cap weighted, ce qui peut surprendre un utilisateur qui attend une optimisation.
- **Recommandation :** Ajouter un log/avertissement quand le mode "sans vues" est activé. Documenter que tau=0.05 est conservateur (standard = 0.025–0.05).

## 2. Ratio de Sharpe — Validation

**Verdict : Correct.**

- Calcul : `(expected_return - rf) / volatility` via `ef.portfolio_performance(risk_free_rate=rf)`.
- En mode manuel, recalcul identique : `(E[R] - rf) / σ`.
- C'est la formule standard. RAS.

## 3. VaR / CVaR — Validation

**Verdict : Trois méthodes implémentées, mais CVaR mal calculé.**

- **VaR paramétrique** : formule gaussienne standard. ✅
- **VaR historique** : percentile empirique. ✅
- **VaR Cornish-Fisher** : expansion correcte pour skewness/kurtosis. ✅
- **CVaR ❌** : les trois CVaR (`cvar_parametric`, `cvar_historical`, `cvar_cornish_fisher`) utilisent **la même fonction** `calculate_cvar()` qui est la CVaR historique empirique. Il n'y a pas de CVaR paramétrique ni de CVaR Cornish-Fisher distincte.
  - **Correction proposée :** 
    - CVaR paramétrique : `-(μ - z * σ)` sur la queue gauche gaussienne
    - CVaR Cornish-Fisher : utiliser le z_cf calculé pour estimer la queue
    - Ou à minima, renommer les trois en `cvar_historical` × 3 et documenter que seule la méthode historique est disponible pour la CVaR.

## 4. Alpha de Jensen — Validation

**Verdict : Formule correcte, benchmark cohérent.**

- `α = R_p - [R_f + β(R_m - R_f)]` — c'est la formule de Jensen.
- Le benchmark par défaut est SPY, qui est un proxy acceptable du marché actions US.
- **Réserve :** Pour un portefeuille français (FR), SPY n'est pas le bon benchmark. Le CAC 40 (^FCHI) ou MSCI Europe serait plus pertinent. L'alpha sera biaisé si le benchmark ne correspond pas à l'univers d'investissement.
- **Recommandation :** Permettre au frontend de sélectionner le benchmark selon la région (SPY pour US, ^FCHI pour FR, ^STOXXE pour EU).

## 5. Risk-Free Rate par Pays

**Les valeurs codées en dur sont obsolètes. Voici la comparaison :**

| Région | Valeur codée | Référence | Taux actuel (avr. 2026) | Écart |
|--------|-------------|-----------|------------------------|-------|
| US | 4.3% | T-Bill 13W | ~3.6% | -70bps ❌ |
| EU | 2.5% | Bund 10Y | ~2.6% | ≈ OK ✅ |
| FR | 2.8% | OAT 10Y | ~3.4% | -60bps ❌ |
| Asie | 0.5% | JGB 10Y | ~2.2% | -170bps ❌❌ |
| Global | 3.5% | — | — | Moyenne faussée ❌ |

**Problèmes majeurs :**
1. Le JGB à 0.5% est totalement obsolète — le Japon a sorti ses taux négatifs, le 10Y est à ~2.2%.
2. Le T-Bill 13W US a baissé de 4.3% à ~3.6%.
3. L'OAT française a monté à ~3.4% (spread Bund-OAT ~75bps).
4. Le code utilise `^IRX` (T-Bill 13W US) en fallback dynamique — c'est bien, mais ce n'est utilisé que pour le calcul global, pas différencié par région.

**Recommandation :**
- Implémenter un dict de taux sans risque par région, rafraîchi via API ou weekly cron.
- Valeurs suggérées : US=3.6%, EU=2.6%, FR=3.4%, Asie=2.2%, Global=3.0%.
- Le fallback `^IRX` est correct pour US, mais ajouter les équivalents pour EU (`^IRX` → Bund sur Bloomberg/ECB) et JP.

## 6. Portefeuille FR — Revue des 8 actions

**Composition actuelle :** MC.PA (LVMH), OR.PA (L'Oréal), SAN.PA (Sanofi), CAP.PA (Capgemini), AI.PA (Air Liquide), RMS.PA (Hermès), BNP.PA (BNP Paribas), ACA.PA (Crédit Agricole)

**Analyse sectorielle :**

| Secteur | Actions | Poids estimé | Concentration |
|---------|---------|-------------|---------------|
| Luxe/Consommation | MC, OR, RMS | ~40% | ⚠️ Très concentré |
| Santé | SAN | ~12% | OK |
| IT/Conseil | CAP | ~10% | OK |
| Industriel | AI | ~10% | OK |
| Banque | BNP, ACA | ~25% | ⚠️ Cyclique |

**Verdict : Portefeuille déséquilibré — surpondération luxe et bancaire.**

- **Luxe (MC + OR + RMS = 3/8)** : secteur très corrélé, cyclique sur le cycle asiatique, exposition Chine importante. Si la Chine ralentit, les trois chutent ensemble.
- **Banque (BNP + ACA = 2/8)** : cyclique et sensible aux taux. Dans un scénario de hausse de taux, les deux souffrent.
- **Manques notables :**
  - Aucune exposition énergie (TotalEnergies TTE.PA)
  - Aucune exposition télécom (Orange ORA.PA)
  - Aucune exposition infrastructure/utilité (Vinci DG.PA, EDF — privé mais EN.PA)
  - Aucune expos assurance (AXA)
  - Aucune exposition défensive/hors luxe alimentaire (Danone BN.PA)

**Alternatives proposées :**

| Remplacement | Action retirée | Raison |
|-------------|---------------|--------|
| TTE.PA (TotalEnergies) | ACA.PA | Diversification énergie, cash flow défensif |
| DG.PA (Vinci) | CAP.PA | Infrastructure = moins cyclique que IT |
| CS.PA (AXA) | BNP.PA ou ACA.PA | Assurance < banque en risque systémique |
| BN.PA (Danone) | — | Défensif, low beta, complément alimentaire |

**Portefeuille FR proposé (8 actions diversifiées) :**
MC.PA, OR.PA, SAN.PA, AI.PA, RMS.PA, TTE.PA, DG.PA, CS.PA

→ Couverture : luxe (2), santé (1), industriel (1), énergie (1), infra (1), assurance (1), consommation (1 via OR).

## 7. Stress Testing — Module ajouté

Fichier : `stress_test.py`

4 scénarios implémentés :
1. **Crash 2008** : -40% marché, vol x3, corrélation 0.85, taux -150bps
2. **Hausse de taux +300bps** : -15% actions, vol x1.8, corrélation 0.70
3. **Stagflation** : -25% sur 2 ans, vol x2, taux +200bps
4. **Black Swan** : -60% en 21 jours, vol x5, corrélation 0.95

Chaque scénario calcule : VaR 99%, CVaR 99%, max drawdown, perte en €, durée.

**Résultat démo (portefeuille 100k€, rendement 8%, vol 15%) :**

| Scénario | VaR 99% | CVaR 99% | Max DD | Perte estimée |
|----------|---------|----------|--------|--------------|
| Crash 2008 | ~42% | ~52% | 34% | 40 000 € |
| Hausse taux | ~18% | ~22% | 11% | 15 000 € |
| Stagflation | ~30% | ~37% | 19% | 25 000 € |
| Black Swan | ~63% | ~78% | 57% | 60 000 € |

---

## Résumé des actions à prendre

1. **CVaR** — Corriger les 3 CVaR pour refléter distinctement chaque méthode, ou renommer proprement
2. **Risk-free rates** — Mettre à jour les valeurs, idéalement via API dynamique
3. **Benchmark par région** — Ajouter ^FCHI pour FR, ^STOXXE pour EU
4. **Portefeuille FR** — Diversifier hors luxe/banque, ajouter TTE.PA, DG.PA, CS.PA
5. **Stress test** — Intégrer dans l'API et le frontend (module prêt)