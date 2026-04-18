# Review Velocity Core — Séraphin

Date : 2026-04-18

## 1. Architecture Backend

**Points positifs :**
- Séparation claire engine/main : moteur métier vs API
- Cache disque avec TTL, hash SHA256 — simple et fonctionnel
- Paramétrage riche (cov_method, tau, contraintes de poids)
- FX hedging automatique pour les devises non-USD

**Problèmes sérieux :**

- **CORS `allow_origins=["*"]`** — Ouvert à n'importe qui. En production, c'est une faille. Il faut restreindre au domaine du frontend.

- **Aucune authentification** — L'API est complètement ouverte. N'importe qui peut appeler `/analyze` et consommer des ressources.

- **`except: pass` et `except Exception`** — Trop de catch-all silencieux. Le bare `except: pass` dans `fetch_stock_data` (change_percent) et les `except Exception` dans `run_analysis` (benchmark, Fama-French) masquent des erreurs réelles. Impossible de debugger en production.

- **Walk-forward backtest fait un appel Yahoo Finance par période** — `yf.Tickers(symbols).info` à chaque fenêtre de rebalancement. C'est un appel réseau par période, non caché. Avec 20 périodes, ça fait 20 appels. Lent et fragile.

- **Pas de rate limiting** — Un utilisateur peut spammer `/analyze` et saturer le serveur + l'API Yahoo Finance.

- **Pas de validation des symboles côté serveur** — On peut envoyer n'importe quoi dans `symbols`. Pas de limite sur le nombre d'actifs (10? 50? 500?). Ça peut exploser en mémoire et en appels réseau.

- **`portfolio_performance()` sur un EF modifié** — Quand on ajoute des contraintes custom à `EfficientFrontier`, `portfolio_performance()` peut retourner des métriques incohérentes avec les poids réels. PyPortfolioOpt ne recalcule pas toujours correctement après `max_sharpe` avec contraintes custom.

- **Gestion d'erreur dans `/analyze`** — Un seul `HTTPException(500)` pour tout. Pas de distinction entre erreur de validation (400), data indisponible (503), et erreur interne (500).

## 2. Frontend

**Points positifs :**
- Design premium, palette cohérente (navy/wine/gold), CSS variables bien structurées
- Mode sombre/clair complet
- Animations subtiles (counter, bar-fill, skeleton)
- Responsive jusqu'à 480px
- Ticker strip live avec polling 30s
- Empty state, loading state, modale détail actif

**Problèmes :**

- **KPI grid 5 colonnes** — Sur tablette (768-1024px), 5 KPI cards par ligne c'est trop étroit. Le breakpoint passe à 2 colonnes seulement à <768px. Il manque un breakpoint intermédiaire à 3 colonnes.

- **Pas d'accessibilité** — Pas d'attributs `aria-label`, pas de navigation clavier visible, les modales n'ont pas de focus trap, les graphiques Recharts n'ont pas d'alt text. Aucun `role` sur les éléments interactifs.

- **Ticker strip infini** — Duplication des tickers `[...tickers, ...tickers]` pour l'animation CSS. Ça double le DOM. Pas grave avec 8 tickers, mais c'est un pattern fragile.

- **`manualWeights` jamais mis à jour** — `const [manualWeights] = useState<Record<string, string>>({})` — pas de setter. Le mode manuel est dans l'UI mais ne fonctionne pas réellement.

- **Pas de debouncing sur l'input ticker** — Chaque frappe dans le champ "Ajouter un ticker" pourrait déclencher un search. Actuellement ça ne le fait pas (juste `addTicker` on Enter), mais si on ajoute de l'autocomplete, il faudra debouncer.

- **`exportPDF` via `window.print()`** — C'est un export PDF très basique. Les graphiques Recharts ne s'impriment pas toujours correctement (problèmes connus avec SVG dans les PDF).

- **Bundle size** — Recharts est lourd (~400KB). Pour un dashboard, c'est acceptable, mais il n'y a pas de lazy loading sur les onglets. Tout est rendu d'un coup.

## 3. Modèles Financiers

### Black-Litterman

**Correct dans les grandes lignes**, mais :
- `pi="market"` avec `market_caps` — OK, c'est l'approche standard
- **Problème :** Quand il n'y a pas de vues (`views=[]`), on fallback sur `market_implied_prior_returns` seul. C'est correct, mais le code crée un `BlackLittermanModel` puis l'ignore si `bl_returns` est None. Le flux est confus.
- **Tau = 0.05** — Valeur standard dans la littérature. OK.
- **Pas de matrice Omega** — Les vues n'ont pas de matrice d'incertitude (`Omega`). PyPortfolioOpt la déduit par défaut (`tau * P @ S @ P.T`), ce qui est acceptable mais pas configurable par l'utilisateur.

### Markowitz (EfficientFrontier)

- `max_sharpe()` — Correct
- **Contraintes de poids** — `w >= min_weight` et `w <= max_weight` sont ajoutées via lambda. C'est correct mais fragile : si `min_weight * n_assets > 1`, le solveur va échouer silencieusement. Le code a un check (`if n_assets >= 4 and min_weight * n_assets <= 1.0`), mais il devrait envoyer une erreur claire à l'utilisateur plutôt que de silencieusement ignorer les contraintes.

### VaR/CVaR

- **VaR paramétrique** — `-(daily_ret - z_99 * daily_vol)` — Formule correcte pour la VaR gaussienne à 99%.
- **VaR historique** — `np.percentile(rp, 1)` — Correct.
- **VaR Cornish-Fisher** — La formule est correcte. L'expansion est standard.
- **CVaR paramétrique** — `phi(z) / (1-confidence) * sigma - mu` — Correct sous hypothèse normale.
- **CVaR Cornish-Fisher** — **Problème :** Utilise `phi(z_cf)` comme si c'était la densité au quantile Cornish-Fisher. Mais la densité d'une distribution modifiée par Cornish-Fisher n'est pas simplement `norm.pdf(z_cf)`. C'est une approximation grossière. La CVaR Cornish-Fisher correcte nécessite une intégration numérique ou une approximation spécifique (voir Boudt et al. 2008). En l'état, cette métrique est **peu fiable**.

- **CVaR historique** — `rp[rp <= var_threshold].mean()` — Correct, mais attention : si très peu d'observations sont sous le seuil (possible avec peu de données), la CVaR sera très bruitée.

### Fama-French 5 Facteurs

- **Régression OLS** — Standard et correcte.
- **Utilise les facteurs globaux** (`Global_5_Factors_Daily`) — OK pour un portefeuille international, mais inadapté pour un portefeuille US-only. Il faudrait permettre le choix du modèle (US 3-factor, US 5-factor, Global).
- **Pas de diagnostic de régression** — R², t-statistiques, p-values ne sont pas retournés. Impossible d'évaluer la significativité des expositions.

### Walk-Forward Backtest

- **Logique correcte** — Rolling window train/test, rebalancement périodique.
- **Problème majeur :** Appelle `yf.Tickers(symbols).info` à chaque période pour récupérer les market caps. En plus d'être lent, c'est **look-ahead bias potentiel** : les market caps actuelles sont utilisées pour des périodes historiques.
- **Fallback equal weights silencieux** — Si l'optimisation échoue, on passe à equal weights sans avertir l'utilisateur. Ça fausse les résultats du backtest.
- **Max drawdown calculé sur les rendements cumulés par période** — C'est une approximation. Le vrai max drawdown devrait être calculé sur les rendements journaliers agrégés de toute la période.

## 4. Points Faibles (ce qui peut mal tourner en production)

1. **Dépendance Yahoo Finance** — yfinance est notoirement instable. L'API Yahoo change régulièrement, les appels échouent, les données sont manquantes. Pas de fallback (autre source de données). En production, l'outil sera indisponible régulièrement.

2. **Pas de limites sur le nombre d'actifs** — Un utilisateur peut envoyer 100 tickers. Chaque ticker = 1 appel `yf.Ticker().info` en parallèle (max_workers=10) + download de 5 ans d'historique. Ça peut prendre 30-60 secondes et consommer beaucoup de mémoire.

3. **Race conditions sur le cache** — Pas de locking sur les fichiers cache. Si deux requêtes écrivent le même key simultanément, corruption possible.

4. **Pas de timeout sur les appels yfinance** — Si Yahoo ne répond pas, la requête HTTP peut bloquer indéfiniment. FastAPI avec `async def` mais `engine.run_analysis` est synchrone — ça bloque l'event loop.

5. **Le walk-forward backtest est un piège UX** — Il est lent (appels réseau par période) et peut échouer silencieusement. L'utilisateur attend longtemps et peut obtenir des résultats fallback (equal weights) sans le savoir.

6. **Pas de validation des vues Black-Litterman** — On peut créer des vues incohérentes (bull et bear identiques, valeur nulle, etc.) sans avertissement.

7. **FX hedging basique** — Le FX est appliqué sur les prix historiques via ffill/bfill. C'est une approximation correcte pour un backtest, mais en temps réel, les taux de change varient intra-day. Pas de gestion du risque de change explicite.

## 5. Améliorations Prioritaires (Top 5)

### 1. Sécuriser l'API
- CORS restrictif (domaine du frontend uniquement)
- Authentification minimale (API key ou JWT)
- Rate limiting (ex: 10 req/min par IP)
- Validation stricte des inputs (max 20 tickers, symboles autorisés uniquement)

### 2. Stabiliser la source de données
- Ajouter un fallback à Yahoo Finance (ex: EODHD, Alpha Vantage, ou même un cache plus agressif)
- Timeout sur tous les appels réseau (5-10s max)
- Runner `run_analysis` dans un `run_in_executor` pour ne pas bloquer l'event loop async

### 3. Corriger le walk-forward backtest
- Pré-charger les market caps une seule fois (pas par période)
- Utiliser les market caps de l'époque (pas les actuels) pour éviter le look-ahead bias
- Retourner un flag quand l'optimisation échoue et qu'on utilise equal weights
- Calculer le max drawdown sur les rendements journaliers continus, pas par période

### 4. Corriger la CVaR Cornish-Fisher
- Remplacer par une intégration numérique (Monte Carlo avec les moments estimés, ou approximation de Boudt et al.)
- Ou supprimer la métrique et documenter pourquoi elle n'est pas fiable

### 5. Améliorer le feedback utilisateur
- Mode manuel fonctionnel (les poids ne sont jamais sauvegardés dans le state)
- Retourner les diagnostics de régression Fama-French (R², p-values)
- Alerter l'utilisateur quand les contraintes de poids sont ignorées
- Ajouter un breakpoint 768-1024px pour les KPI (3 colonnes)
- Ajouter les attributs ARIA minimum sur les éléments interactifs

---

**Verdict global :** Le projet est ambitieux et visuellement abouti. Les modèles sont globalement corrects, sauf la CVaR Cornish-Fisher qui est théoriquement douteuse. L'architecture backend est la faiblesse principale : pas de sécu, dépendance fragile à Yahoo Finance, et un walk-forward backtest qui a un biais look-ahead. Ce sont des problèmes bloquants pour la production, pas des détails.