# 📊 Formules de Risque Avancé - Velocity Core

## 5b. VaR Historique et Cornish-Fisher

### VaR Historique
**Pourquoi :** La méthode paramétrique suppose une distribution normale. Or, les marchés ont des "queues grasses" (evt extrêmes plus fréquents que prévu). La VaR historique utilise la distribution *réelle* des rendements passés.

*   **Formule :**
    `VaR_hist = - Percentile(P&L, 1%)`
    *   *Explication :* On prend le 1% le plus mauvais rendement historique et on le scale à la volatilité actuelle. Aucune hypothèse de normalité.

*   **Avantages :**
        - Capture la vraie distribution des marchés (skewness, kurtosis)
        - Pas d'hypothèse sur la forme de la distribution
        - Plus réaliste pour les marchés extrêmes

### VaR Cornish-Fisher
**Pourquoi :** C'est un compromis entre méthode paramétrique et historique. On corrige la VaR gaussienne avec les mesures de forme réelles (skewness, kurtosis) via une expansion de Cornish-Fisher.

*   **Formule (Expansion à 4 moments) :**
    `z_cf = z + (z² - 1) * S/6 + (z³ - 3z) * K/24 - (2z³ - 5z) * S²/36`
    
    `VaR_CF = - (μ - z_cf * σ)`
    
    *   **z** : Z-score normal (2.326 pour 99%)
    *   **S** : Skewness (asymétrie)
    *   **K** : Kurtosis excédentaire (queues grasses)
    *   **μ** : Rendement moyen
    *   **σ** : Volatilité

*   **Interprétation :**
    - Skewness négative → VaR plus haute (risque de baisses importantes)
    - Kurtosis élevé → VaR plus haute (risque d'événements extrêmes)

---

## 6. CVaR / Expected Shortfall
**Pourquoi :** Le VaR est une *threshold*, pas une *attente*. CVaR calcule la *moyenne des pertes qui dépassent le VaR*. C'est une mesure de risque plus complète.

*   **Formule :**
    `CVaR_α = E[P&L | P&L ≤ VaR_α]`
    
    *   *Explication :* Si VaR(99%) = -10%, CVaR calcule la moyenne de TOUS les rendements inférieurs à -10%. C'est souvent 20-30% plus élevé que VaR, ce qui montre la gravité des pertes extrêmes.

*   **Interprétation :**
    - CVaR est *convexe* → meilleure mesure de risque
    - Utilisé pour la réglementation (Basel III)
    - Plus sensible aux queues de distribution que VaR

---

## 7. Contraintes de Poids dans l'Optimisation
**Pourquoi :** Sans contraintes, Markowitz peut donner des poids extrêmes (ex: 40% dans une action, 0.1% dans une autre). Les contraintes rendent le portefeuille plus réaliste et stable.

*   **Bornes de poids :**
    `w_min ≤ W_i ≤ w_max`
    
    *   **w_min (par défaut 2%)** : Empêche les positions trop petites (trop chères à gérer)
    *   **w_max (par défaut 25%)** : Empêche la concentration excessive (risque idiosyncrasique)

*   **Implementation avec pypfopt :**
    ```python
    ef = EfficientFrontier(bl_returns, S)
    ef.add_constraint(lambda w: w >= 0.02)  # Min 2%
    ef.add_constraint(lambda w: w <= 0.25)  # Max 25%
    ef.max_sharpe(risk_free_rate=rf)
    ```

*   **Avantages :**
    - Diversification plus équilibrée
    - Réduction du turnover (moins de rebalancement)
    - Portefeuille plus robuste aux erreurs d'estimation

---

## 8. Covariance Shrinkage (Ledoit-Wolf, Oracle)
**Pourquoi :** La matrice de covariance empirique (`sample_cov`) est très bruyante avec peu de données. Les méthodes de shrinkage combinent l'estimation empirique avec une matrice cible (ex: identité ou diagonale) pour plus de stabilité.

*   **Ledoit-Wolf (asymptotiquement optimal) :**
    ```python
    from pypfopt.covariance_shrinkage import LedoitWolf
    S_shrunk = LedoitWolf(df).ledoit_wolf()
    ```
    
    *   Formule : `Σ_sw = λ * Φ + (1-λ) * Σ_sample`
    *   **λ** : Parameter choisi automatiquement pour minimiser l'erreur quadratique
    *   **Φ** : Matrice cible (souvent diagonale ou identité)
    *   **Σ_sample** : Covariance empirique

*   **Oracle Approximating (OAS) :**
    ```python
    from pypfopt.covariance_shrinkage import OracleApproximating
    S_oas = OracleApproximating(df).oracle_approximating()
    ```
    *   Basé sur la théorie des matrices aléatoires
    *   Généralement meilleur avec peu d'observations

*   **Avantages :**
        - Matrice plus stable, moins sensible au bruit
        - Meilleurs poids d'optimisation (moins de surajustement)
        - Particulièrement utile avec peu de données (ex: < 252 jours)

---

## 9. Walk-Forward Backtest (Rolling Window)
**Pourquoi :** Le backtest classique sur tout l'historique est biaisé (look-ahead bias). Le walk-forward simule le trading réel où on réévalue le portefeuille régulièrement.

*   **Procédure :**
    1. **Train window** (252 jours) : Optimiser les poids
    2. **Test window** (63 jours) : Appliquer les poids, mesurer performance
    3. **Roll forward** : Répéter avec nouvelles données

*   **Formule de performance OOS (Out-of-Sample) :**
    `Sharpe_OOS = mean(Rp_test) / std(Rp_test)`
    
    *   **Rp_test** : Rendements du portefeuille sur la fenêtre de test
    *   **Sharpe OOS** : Mesure de la robustesse de la stratégie

*   **Interprétation :**
    - Sharpe OOS proche du Sharpe IS (in-sample) → pas de surajustement
    - Sharpe OOS bien inférieur → surajustement (overfitting)
    - Drawdown OOS élevé → risque de rupture de pattern

*   **Implementation :**
    ```python
    def walk_forward_backtest(returns_df, symbols, rf, train_days=252, test_days=63):
        # rolling windows, optimize weights, measure OOS performance
        return {"results": [...], "summary": {"avg_return", "avg_sharpe", "max_dd"}}
    ```

---

## 10. Alpha de Jensen - Rendement Marché Observé
**Pourquoi :** Le fallback fixe à 12% est artificiel. L'alpha devrait être calculé par rapport au rendement *réellement observé* du benchmark.

*   **Formule :**
    `α = E[R_p] - [ R_f + β * (R_m,obs - R_f) ]`
    
    *   **R_m,obs** : Rendement annualisé du benchmark (SPY) sur la période
    *   Calcul : `R_m,obs = (1 + R_total)^(252/N) - 1`
    *   **N** : Nombre de jours dans la période
    *   **R_total** : Rendement total sur la période

*   **Avantages :**
    - Alpha plus réaliste et dynamique
    - Pas de dépendance à une constante arbitraire
    - Réaction aux conditions de marché actuelles

---

## Résumé des Métriques de Risque

| Métrique | Formule | Unité | Interprétation |
|----------|---------|-------|----------------|
| VaR (paramétrique) | `- (μ - z * σ)` | % | Perte max sous hypothèse normale |
| VaR (historique) | `-Percentile(P&L, 1%)` | % | Perte max dans données passées |
| VaR (Cornish-Fisher) | `- (μ - z_cf * σ)` | % | Perte max avec correction skew/kurt |
| CVaR | `E[P&L | P&L ≤ VaR]` | % | Perte moyenne au-delà du VaR |
| Alpha (Jensen) | `E[R_p] - [R_f + β*(R_m- R_f)]` | % | Performance après risque ajusté |

---

*Document généré automatiquement par Velocity Core*
*© 2026 - Karl BAUJON*
