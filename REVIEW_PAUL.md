# Review Financière — Velocity Core
**Auteur :** Paul (agent business/finance)  
**Date :** 2026-04-18  
**Statut :** Critique, honnête, actionnable

---

## 1. Black-Litterman

### Implémentation
L'implémentation utilise `pypfopt.BlackLittermanModel` avec `pi="market"`, ce qui est correct — ça calcule Π via la formule `δ × Σ × w_mkt`. La confiance tau est paramétrable (défaut 0.05), c'est bien.

### Problèmes

**Fallback sans views est douteux.** Quand `len(views) == 0`, le code appelle `market_implied_prior_returns()` directement. C'est un fallback logique en théorie, mais en pratique ça retourne Π brut — les rendements d'équilibre. Le problème : l'optimisation Markowitz ensuite utilise ces rendements d'équilibre comme input, ce qui revient à faire du Markowitz pur sur des rendements market-cap weighted. Autant faire de la capitalisation boursière directe. Le fallback est techniquement correct mais **financièrement inutile** — l'utilisateur croit avoir une allocation BL, il a une allocation markétienne déguisée.

**Omega n'est pas spécifié par l'utilisateur.** La classe `BlackLittermanModel` de pypfopt construit Ω automatiquement via `τ × P × Σ × Pᵀ` quand `omega=None`. C'est le défaut de He-Litterman (1999), c'est acceptable, mais ce n'est documenté nulle part dans le projet. Un utilisateur avancé ne peut pas contrôler la confiance de ses views de manière indépendante.

**Views relatives (type 'R') :** L'implémentation construit correctement les vecteurs P avec +1/-1. C'est bon. Mais il n'y a aucune validation que bull ≠ bear, ni que les indices pointent vers des symboles valides dans `fetched_symbols`. Le `try/except` silently ignore les erreurs — si une view est mal formée, elle disparaît sans trace.

**Delta (risk_aversion) est hardcodé à 2.5.** C'est une approximation commune mais pas toujours justifiée. Devrait être paramétrable ou calculé depuis les données.

**Verdict :** Implémentation fonctionnellement correcte, mais le fallback sans views est une impasse financière, et le manque de contrôle sur Ω est un gap.

---

## 2. Markowitz / EfficientFrontier

### Ce qui marche
- `max_sharpe()` avec contraintes min/max est bien implémenté
- Les 3 méthodes de covariance (sample, Ledoit-Wolf, Oracle Approximating) sont disponibles
- L'efficient frontier est tracée par `efficient_return()` sur 20 points

### Problèmes

**Contraintes mal construites.** Les contraintes `w >= min_weight` et `w <= max_weight` sont ajoutées via `add_constraint` avec des lambdas. En PyPortfolioOpt, ces contraintes créent un problème non-convexe qui peut échouer silencieusement ou converger vers un optimum local. Les vraies bound constraints doivent utiliser le paramètre `weight_bounds` du constructeur. L'approche actuelle est **instable**.

**Pas de regularisation des rendements.** Les rendements BL sont alimentés directement dans EfficientFrontier sans lissage. Si BL produit des rendements extrêmes (ex: un titre à -30%), l'optimization va produire des poids bizarres malgré les contraintes min/max.

**Efficient frontier peut crasher silencieusement.** Le `try/except: pass` dans la boucle de l'EF est dangereux. Si un point échoue, la frontière a des trous. L'utilisateur verra un graphique incomplet sans comprendre pourquoi.

**Edge case n_assets < 4 :** Les contraintes min/max ne sont pas appliquées si `n_assets < 4`. Ça veut dire qu'un portefeuille de 3 actions n'a aucune contrainte de poids → concentration extrême possible.

**Verdict :** Fonctionnel mais fragile. Les contraintes doivent utiliser `weight_bounds`, et les edge cases (peu d'actifs, rendements extrêmes) ne sont pas gérés.

---

## 3. VaR / CVaR

### VaR Paramétrique
La formule est `-(μ_daily - z₉₉ × σ_daily)`. C'est correct pour une VaR journalière paramétrique sous hypothèse gaussienne.

### VaR Historique
Utilise `np.percentile(rp, 1)` avec un `max(0, ...)`. Correct.

### VaR Cornish-Fisher
L'expansion est correcte : `z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36`. C'est la formule standard.

### Problèmes

**Les 3 CVaR ne sont pas vraiment distinctes.**

- **CVaR paramétrique** utilise la formule analytique gaussienne `(φ(z)/(1-α)) × σ`. C'est correct et distinct.
- **CVaR historique** calcule la moyenne empirique au-delà du percentile. C'est correct et distinct.
- **CVaR Cornish-Fisher** remplace juste `z` par `z_cf` dans la formule paramétrique, en utilisant `φ(z_cf)/(1-α)`. **C'est une approximation douteuse.** La CVaR Cornish-Fisher devrait intégrer numériquement la queue de distribution ajustée, pas juste substituer le z-score. La formule `φ(z_cf)/(1-α)` n'est pas la CVaR Cornish-Fisher — c'est la CVaR gaussienne évaluée au point ajusté. La vraie CVaR CF nécessite une intégration numérique de la densité ajustée.

**Le max(0, ...) sur toutes les VaR/CVaR est trompeur.** Si le portefeuille a un rendement attendu élevé avec faible volatilité, la VaR peut être négative (i.e., le pire cas à 99% est encore un gain). Forcer à 0 cache cette information.

**Verdict :** VaR paramétrique et historique sont OK. La CVaR Cornish-Fisher est **fausse** — c'est une approximation non standard qui sous-estime le risque de queue.

---

## 4. Fama-French

### Implémentation
La régression OLS via `statsmodels` sur `Global_5_Factors_Daily` est correcte structurellement. Le code :
1. Télécharge les facteurs via `pandas_datareader`
2. Calcule les rendements excédents : `R_p - RF`
3. Régresse sur Mkt-RF, SMB, HML, RMW, CMA

### Problèmes

**`Global_5_Factors_Daily` est le bon jeu de données** pour un portefeuille international. OK.

**Pas de R², t-stats, ni p-values.** Le code retourne uniquement les betas (`model.params`), sans aucune mesure de significativité. Un beta de 0.3 avec un t-stat de 0.5 n'a aucune valeur prédictive. C'est un gap **majeur** pour quiconque veut interpréter les résultats.

**Pas de diagnostics de régression.** Pas de test d'autocorrélation des résidus, pas de Durbin-Watson, pas de check de multicolinéarité entre facteurs.

**Seuil de 10 observations.** `len(common) < 10` est un seuil très bas pour une régression à 5 facteurs + constante (6 paramètres). Avec 10 points, les estimateurs sont extrêmement bruités.

**Verdict :** La régression est correcte mais les résultats sont incomplets sans statistiques de significativité. C'est comme donner un diagnostic médical sans les marges d'erreur.

---

## 5. Walk-Forward Backtest

### Méthodologie
Fenêtre roulante train (252j) / test (63j), rebalancement tous les 63 jours. C'est une méthodologie walk-forward standard.

### Problèmes critiques

**Look-ahead bias dans le market cap.** Le backtest appelle `yf.Tickers(symbols).info` pour obtenir les market caps **pendant la période de test**. Ces market caps reflètent les capitalisations actuelles, pas celles de la période d'entraînement. C'est un **look-ahead bias majeur**. Le BL model utilise ces mcaps pour calculer les rendements d'équilibre, ce qui contamine les poids du portefeuille avec de l'information future.

**Calcul du Sharpe sur des sous-périodes.** Le Sharpe est calculé par période puis moyenné : `avg_sharpe = np.mean([r['sharpe'] for r in results])`. Le ratio de Sharpe n'est pas linéaire — moyenner des ratios Sharpe n'a pas de sens statistique. Il faut calculer le Sharpe sur la série complète des rendements OOS.

**Max drawdown mal calculé.** Le code accumule `(1 + r['return'])` sur des périodes disjointes et calcule un drawdown. C'est faux — un drawdown se calcule sur une série temporelle continue de rendements, pas sur des retours agrégés par période.

**Annualisation incorrecte.** `test_return_annualized = (1 + test_return) ** (252/len(test_rp)) - 1` — c'est correct pour un seul période, mais la compound sur des périodes disjointes pose problème.

**Fallback equal-weight silencieux.** Si l'optimization échoue, le code utilise des poids égaux sans avertissement. L'utilisateur ne sait pas que son backtest n'a pas utilisé l'allocation BL.

**Verdict :** Look-ahead bias sur les market caps invalide les résultats. Les métriques agrégées (Sharpe, drawdown) sont mal calculées. Ce module a besoin d'une réécriture.

---

## 6. Risk-Free Rate

Le code utilise `^IRX` (13-week Treasury Bill) comme taux sans risque par défaut, avec fallback à 4%.

### Problèmes

**^IRX est le taux US.** Pour un portefeuille avec des actifs non-USD (EU, FR, ASIA), le taux sans risque devrait être celui de la devise du portefeuille. Le code fait la conversion FX des prix mais pas du taux sans risque.

**Le fallback 4% est daté.** En 2026, les taux US sont potentiellement différents. Mais surtout, le fallback est hardcodé et non régionalisé.

**Pas de taux par région.** Le endpoint `/optimal-portfolio` propose des régions (EU, FR, ASIA) mais ne passe pas de risk-free rate adapté. Le taux US est toujours utilisé.

**Verdict :** À mettre à jour avec des taux par devise/région. Au minimum, un mapping devise → taux sans risque.

---

## 7. Stress Testing

### Ce qui est bien
- 4 scénarios pertinents (crash 2008, hausse de taux, stagflation, black swan)
- Paramétrisation claire (equity_shock, vol_multiplier, correlation_floor, duration)
- Interface propre avec `stress_test_from_analysis()`

### Problèmes

**CVaR stressée est une approximation grossière.** `cvar_scenario = min(var_scenario * 1.25, 0.99)` — multiplier la VaR par 1.25 n'est pas une CVaR. La CVaR d'une distribution normale est `VaR × φ(z)/(1-α) × σ + μ`, pas `1.25 × VaR`. Ce facteur de 1.25 est arbitraire et sous-estime la CVaR pour les scénarios à forte volatilité.

**Le max drawdown est `equity_shock × correlation_floor`.** C'est une heuristique simpliste. Le drawdown réel dépend de la corrélation entre actifs et de leur volatilité individuelle, pas juste d'un facteur multiplicatif sur le choc equity global.

**Pas de stress test par actif.** Les scénarios appliquent un choc uniforme à tout le portefeuille. Un vrai stress test devrait choquer les actions différemment des obligations, les small caps différemment des large caps, etc.

**Pas de corrélation stressée appliquée à la matrice de covariance.** Le `correlation_floor` est défini mais jamais utilisé pour construire une matrice de covariance stressée. Il est juste multiplié avec le choc equity pour le drawdown.

**Verdict :** Module de démo acceptable pour la communication, mais pas robuste pour la gestion de risque réelle. La CVaR est fausse, et les chocs ne sont pas désagrégés par actif.

---

## 8. Top 5 des Améliorations Financières Prioritaires

### 🔴 Priorité 1 — Corriger le look-ahead bias dans le walk-forward backtest
C'est le bug le plus grave. Les market caps doivent être figées à la période d'entraînement, pas rafraîchies en temps réel. Soit utiliser des mcaps historiques (disponibles via yfinance balance_sheet daté), soit utiliser des poids fixes. Sans ça, le backtest est invalide.

### 🔴 Priorité 2 — Corriger la CVaR Cornish-Fisher et la CVaR stressée
La CVaR CF actuelle n'est pas une vraie CVaR Cornish-Fisher. Remplacer par une intégration numérique ou utiliser la formule analytique complète. La CVaR stressée (`var × 1.25`) doit être remplacée par un calcul propre.

### 🟡 Priorité 3 — Ajouter les statistiques de significativité Fama-French
Retourner R², t-stats, p-values, et Durbin-Watson. Sans ça, les betas FF sont non interprétables.

### 🟡 Priorité 4 — Utiliser `weight_bounds` au lieu de `add_constraint` dans Markowitz
Les contraintes lambda créent un problème non-convexe. Remplacer par :
```python
ef = EfficientFrontier(bl_returns, S, weight_bounds=(min_weight, max_weight))
```

### 🟡 Priorité 5 — Régionaliser le risk-free rate
Mapping devise → taux sans risque (EUR → €STR, JPY → JGB, GBP → SONIA, etc.). Au minimum pour les endpoints `/optimal-portfolio` par région.

---

## Résumé

| Module | Note | Statut |
|--------|------|--------|
| Black-Litterman | B | Fonctionnel, fallback douteux, Ω non exposé |
| Markowitz | B- | Fragile sur les contraintes et edge cases |
| VaR/CVaR | C+ | VaR OK, CVaR CF fausse |
| Fama-French | C+ | Régression OK, résultats incomplets |
| Walk-Forward | D | Look-ahead bias, métriques mal calculées |
| Risk-free rate | C | Hardcodé US, pas de régionalisation |
| Stress Test | C- | CVaR fausse, chocs non désagrégés |

**Le projet a une bonne architecture et une couverture impressionnante de modèles. Mais la robustesse mathématique n'est pas au niveau de l'ambition.** Les corrections prioritaires 1 et 2 sont des bugs, pas des améliorations.