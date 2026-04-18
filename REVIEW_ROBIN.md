# Velocity Core — Code Review par Robin

**Date** : 2026-04-18  
**Relecteur** : Robin (agent dev)  
**Portée** : engine.py, main.py, cache.py, stress_test.py, frontend/src/App.tsx, frontend/src/App.css, frontend/src/index.css, requirements.txt

---

## 1. Qualité du code Python

### Bugs critiques

- **engine.py L108 — `future.result()` sans gestion d'exception** : Si `fetch_stock_data` lève une exception dans un thread, `future.result()` la propage mais le `for` loop continue partiellement. Les dicts `mcaps`, `debts`, `currencies` seront incomplets mais le code continue silencieusement. → Crash garanti plus tard (KeyError sur les symboles manquants).

- **engine.py L122-128 — FX conversion défaillante** : `yf.download()` avec un seul ticker retourne un DataFrame dont la colonne Close est une Series, pas un DataFrame. Le `isinstance(fx_data, pd.Series)` gère ce cas, mais si `fx_pairs` a plusieurs éléments et que l'un échoue, `fx_data` peut avoir des colonnes manquantes. Pas de vérification.

- **engine.py L136 — `df.reset_index()['Date'].astype(str)`** : Double appel à `reset_index()`. Le premier `reset_index()` est déjà fait ligne 135, le second recrée un DataFrame à partir du premier. Inefficace mais fonctionne. En revanche, si l'index s'appelle autrement que `Date` (par ex. si yfinance retourne un DatetimeIndex nommé différemment), cela silencieusement ne matchera pas le `cache_get` qui cherche une colonne `Date`.

- **engine.py L168-178 — Walk-forward backtest** : Le backtest appelle `yf.Tickers(symbols)` à chaque fenêtre de rebalancement pour récupérer les market caps. C'est un appel réseau dans une boucle, extrêmement lent et fragile. Si l'appel échoue, le `except:` silencieux fallback sur equal weights — ce qui masque totalement l'erreur.

- **engine.py L247 — `df.pct_change().dropna()`** : Si `df` a des NaN résiduels (après le `dropna` initial sur les prix), `pct_change().dropna()` les supprime. Mais le `dropna` initial sur les prix peut réduire le nombre de lignes à zéro pour certains tickers, et le portefeuille résultant sera faux sans avertissement.

- **engine.py L278-290 — Benchmark** : Le `except:` nu sur 3 lignes avale TOUTE erreur (y compris les bugs de code, MemoryError, KeyboardInterrupt). Si le benchmark échoue, `beta = 1.0` est renvoyé silencieusement — l'utilisateur croit avoir un beta calculé.

- **engine.py L295-300 — Market return pour Jensen's alpha** : Même pattern, `except:` nu. Si le calcul échoue, `market_return = 0.12` est utilisé silencieusement. L'alpha sera calculé sur une valeur hardcodée sans que l'utilisateur le sache.

- **stress_test.py L76-77 — CVaR calculée comme `var_scenario * 1.25`** : C'est une approximation grossière, pas une véritable CVaR. Le commentaire dit "queue épaisse" mais le ratio 1.25 est arbitraire et non documenté. Pour un module de risk management, c'est inacceptable.

- **stress_test.py L85 — `max_dd = abs(scenario["equity_shock"]) * scenario["correlation_floor"]`** : Le drawdown est estimé comme `choc × corrélation_floor`, ce qui n'a aucun fondement mathématique. La corrélation floor est un paramètre de scénario pour la matrice de corrélation stressée, pas un multiplicateur de drawdown.

### Erreurs silencieuses

- **engine.py L14-15 — `mcap` par défaut à 1e9, `total_debt` à 0** : Si `stock.info` échoue partiellement, ces valeurs par défaut seront utilisées sans avertissement. Un market cap d'1 milliard pour une micro-cap ou une erreur de data faussera tout le modèle Black-Litterman.

- **engine.py L30 — `except: pass` sur `change_percent`** : Toute erreur est avalée silencieusement, `change_percent` reste à 0.

- **engine.py L33-34 — `try/except` sur `total_debt`** : Même chose. Si le balance sheet n'est pas disponible, `total_debt` reste à 0 sans avertissement.

- **engine.py L299 — `market_return = 0.12`** : Valeur hardcodée de 12% si le calcul échoue. Aucune indication dans le résultat.

- **cache.py — Pas de limite de taille** : Le cache croît indéfiniment. `cache_cleanup` existe mais n'est jamais appelé automatiquement. Après des centaines d'analyses, le disque se remplit.

- **cache.py L20-28 — Race condition** : Pas de verrou fichier. Deux requêtes simultanées peuvent écrire/lire le même fichier de cache, menant à des JSON corrompus.

### Performance

- **engine.py L119 — `yf.download(fx_pairs, ...)`** : Télécharge 5 ans de données FX à chaque analyse, même si ce ne sont que des tickers USD. Pas de cache FX.

- **engine.py L168 — `yf.Tickers(symbols)` dans la boucle walk-forward** : Appel réseau à chaque fenêtre. Devrait être fait une seule fois avant la boucle.

- **engine.py L280 — `yf.Ticker(benchmark).history(period='5y')`** : Télécharge 5 ans de données benchmark à chaque analyse. Pas de cache.

- **engine.py L350-358 — Efficient frontier** : Crée 20 EfficientFrontier dans une boucle, avec des `try/except: pass` silencieux. Si la majorité échouent, la frontière sera quasi vide sans avertissement.

- **engine.py L360 — `returns_df[fetched_symbols].corr()`** : Calcule la matrice de corrélation complète à chaque analyse. Inutile si on a déjà la covariance matrix.

### Mémoire

- **engine.py L136 — `df.reset_index()` x2** : Crée des copies intermédiaires du DataFrame. Sur un grand portefeuille (20+ actifs, 5 ans de données quotidiennes), c'est ~5MB par copie, ×2 = 10MB gaspillés.

- **engine.py L365 — `historical_evolution`** : Sérialise tout le DataFrame en dict de dicts. Pour 20 actifs × 1260 jours = ~25,000 lignes, chaque ligne est un dict avec ~25 clés. C'est ~5-10MB de JSON par réponse API.

---

## 2. Qualité du code React

### State management

- **`manualWeights` est déclaré mais jamais modifié** (L82 `const [manualWeights] = useState<Record<string, string>>({})`). Le setter n'est pas déstructuré, donc les poids manuels ne fonctionneront jamais. Bug critique : le mode "Manuel" est inopérant.

- **`useCallback` avec dépendances incorrectes** (L170) : `analyze` dépend de `region` via `getRf()`, mais `region` n'est pas dans le tableau de dépendances. Si l'utilisateur change de région sans re-rendre le composant, le taux sans risque sera stale.

- **`generateOptimal` ne nettoie pas l'erreur** : Si une analyse précédente a échoué, `error` n'est pas reset dans `generateOptimal`. L'ancien message d'erreur reste affiché.

### Re-renders inutiles

- **AnimatedCounter** : Chaque KPI crée un `requestAnimationFrame` loop à chaque changement de valeur. Si les KPIs changent souvent (ex: résultats qui se mettent à jour), cela crée 5 animation loops simultanées. Le cleanup via `cancelAnimationFrame` est correct, mais le `ref.current` pattern signifie que si la valeur change pendant l'animation, l'animation suivante part de la valeur intermédiaire, pas de la valeur cible.

- **Ticker strip** : Le mapping `[...tickers, ...tickers]` duplique les éléments pour l'animation CSS. Si `quotes[t]` est undefined pour un ticker, `null` est retourné, créant des "trous" dans l'animation.

- **Re-renders au polling** : Le `setInterval` pour les quotes (L96-103) déclenche un re-render complet toutes les 30s, y compris des graphiques Recharts coûteux. Pas de `React.memo` sur aucun composant.

### Memory leaks potentiels

- **`useEffect` sur `tickers`** (L91-95) : Le `setTimeout` de 300ms est correctement nettoyé, mais si l'utilisateur ajoute/supprime des tickers rapidement, la requête précédente peut revenir après que les tickers ont changé, mettant à jour `quotes` avec des données stale.

- **`setInterval`** (L96-103) : Si le composant est démonté pendant un interval, le cleanup fonctionne via `return () => clearInterval(iv)`. Correct.

### Erreurs possibles

- **`benchData[0]`** (L195) : Si `benchData` est un array vide, `Object.keys(benchData[0])` throwera. Pas de vérification.

- **`result?.walk_forward_backtest?.summary.avg_return`** (L340) : Si `summary` n'existe pas (erreur dans le backtest), c'est un crash silencieux. Le `?.` protège partiellement, mais le rendu des KPIs montrera des valeurs vides sans explication.

- **Pas de validation côté client** : L'utilisateur peut envoyer 0 tickers, des tickers invalides, des valeurs négatives pour les vues BL, ou `min_weight > max_weight`. Le bouton est juste `disabled` si `tickers.length === 0`, mais aucune validation des autres inputs.

---

## 3. CSS

### Cohérence

- **Deux systèmes de variables** : `index.css` définit `--bg-primary`, `--accent`, etc. `App.css` définit `--vc-bg-primary`, `--vc-accent-gold`, etc. Les variables de `index.css` ne sont jamais utilisées dans `App.css`. Double définition, confusion potentielle.

- **Couleurs en dur dans App.tsx** : Les couleurs des graphiques (`isLight ? '#1f7a53' : '#2d9f6f'`) sont en dur dans le composant au lieu d'utiliser les CSS variables. Si on change les variables CSS, les graphiques ne suivront pas.

### Mode clair incomplet

- **Light mode : `--vc-accent-cyan` pas défini** dans `.vc-light`. La variable reste `#4a90d9` (dark), ce qui est trop sombre sur fond clair.

- **Light mode : `--vc-gradient-main`, `--vc-gradient-wine`, `--vc-gradient-card` pas définis** dans `.vc-light`. Les gradients restent sombres.

- **Light mode : pas de `--vc-accent-green` override**. `#2d9f6f` est OK mais aurait pu être ajusté.

- **`select option` dans `index.css`** : Hardcodé en `background: #111827` (sombre). En mode clair, les options de select seront sur fond noir. Le fix à la fin de `App.css` (`.vc-light select option`) est correct mais entre en conflit avec `index.css`.

- **Scrollbar** : `index.css` définit un scrollbar sombre qui ne change pas en mode clair.

### Responsive

- **Breakpoints corrects** : 1024px (sidebar), 768px (grilles), 480px (compact). Acceptable.

- **KPI grid en 5 colonnes** : Sur tablette (768-1024px), les 5 KPIs restent en ligne mais deviennent très étroits. Manque un breakpoint intermédiaire pour passer en 3 colonnes.

- **Correlation matrix** : Pas de scroll horizontal sur mobile. Avec 8+ actifs, la table déborde.

### Accessibilité

- **Aucun `aria-label`** sur les boutons de navigation, les inputs, les KPI cards.

- **Pas de `role`** sur les éléments interactifs personnalisés (modals, ticker strip).

- **Contraste insuffisant** en mode sombre : `--vc-text-muted: #5c6478` sur `--vc-bg-primary: #0a0f1e` donne un ratio de contraste ~4.2:1, en dessous du WCAG AA (4.5:1 pour texte normal).

- **Pas de `focus-visible`** styling sur les boutons et inputs. Navigation clavier impossible à distinguer.

- **`<h1>` dans la sidebar** : Il n'y a qu'un seul `<h1>` (le logo), et pas de `<h2>` dans le `<main>` pour les sections de page. La hiérarchie des titres est plate.

---

## 4. API

### Endpoints

- **`POST /analyze`** : Accepte n'importe quels symboles, n'importe quel nombre de vues, n'importe quelles valeurs. Pas de validation côté serveur. Un utilisateur peut envoyer 1000 tickers et planter le serveur.

- **`GET /quotes`** : Le paramètre `symbols` n'est pas limité en taille. Un utilisateur peut envoyer `?symbols=AAPL,MSFT,...` avec 500 tickers.

- **`GET /optimal-portfolio`** : Les portefeuilles sont hardcodés. Pas de validation que `region` est une clé valide (le fallback est `US`, ce qui est OK).

- **`GET /search`** : Appelle `yf.Ticker(q).info` pour n'importe quel input. Pas de rate limiting. Un attaquant peut flood le serveur de requêtes Yahoo Finance.

### Validation

- **Aucune validation** sur `AnalysisRequest.symbols` : pas de vérification de longueur, de format, de caractères spéciaux. Un symbole comme `; rm -rf /` passera directement dans yfinance.

- **`cov_method`** : Accepte n'importe quelle string. Si la valeur n'est pas `sample_cov`, `ledoit_wolf`, ou `oracle_approximating`, le code fallback sur `sample_cov` dans engine.py (L243), mais c'est implicite.

- **`tau`** : Pas de validation de range. Un utilisateur peut envoyer `tau=1000` ou `tau=-1`, ce qui peut causer des erreurs numériques dans Black-Litterman.

- **`min_weight` / `max_weight`** : Pas de validation que `min_weight < max_weight` ou que `min_weight * n_assets <= 1`. Le code Python (L262-263) a une garde conditionnelle mais silencieuse : si les contraintes sont irréalisables, il les ignore sans avertissement.

### Erreurs

- **`POST /analyze`** : En cas d'exception, le traceback complet est imprimé sur stdout (`traceback.print_exc()`) ET l'exception est renvoyée au client. Le traceback peut contenir des chemins de fichiers, des noms de modules, etc. Fuite d'information.

- **`GET /search`** : Si `yf.Ticker(q).info` lève une exception qui n'est pas `HTTPException`, elle est catchée par le second `except` et renvoie un 404 générique. Mais si c'est une erreur réseau/timeout, l'utilisateur ne sait pas que c'est un problème temporaire.

### CORS

- **`allow_origins=["*"]`** : Accepte n'importe quelle origine. En production, c'est une faille de sécurité. Seul le frontend devrait être autorisé.

- **`allow_credentials=True`** : Combiné avec `allow_origins=["*"]`, c'est techniquement interdit par la spec CORS (les navigateurs bloquent `credentials: true` avec `*`). Ça fonctionne parce que FastAPI ne vérifie pas cette incompatibilité, mais c'est incorrect.

### Rate limiting

- **Aucun rate limiting**. Un utilisateur peut spammer `/analyze` et saturer le serveur (chaque appel fait 8+ requêtes Yahoo Finance + calculs numPy).

---

## 5. Sécurité

### Injection

- **Pas d'injection SQL** (pas de base de données), mais les symboles passent directement dans `yf.Ticker()` et `yf.Tickers()`. Si yfinance construit des URLs à partir de ces strings sans sanitisation, c'est un vecteur d'attaque potentiel (SSRF).

- **`GET /quotes?symbols=...`** : Le split sur `,` et le strip sont OK, mais le symbole passe directement dans `yf.Ticker(t)`. Pas de whitelist ni de validation de format.

### XSS

- **React protège naturellement** contre le XSS dans le rendu JSX. Les données de l'API sont rendues comme du texte, pas comme du HTML. ✅

- **`dangerouslySetInnerHTML`** : Absent du code. ✅

### Exposition d'erreurs

- **`traceback.print_exc()`** dans `/analyze` : Imprime le traceback complet sur stdout. En production, les logs stdout peuvent être accessibles.

- **`raise HTTPException(status_code=500, detail=str(e))`** : Renvoie le message d'exception complet au client. Peut contenir des informations sensibles (chemins de fichiers, noms de modules, versions de libraries).

### Validation des inputs

- **Aucune validation Pydantic avancée** : `symbols` est `List[str]` mais pas de validation de longueur, de format regex, ou de valeurs autorisées.

- **`views`** : Les `BLView` ont des `Optional[int]` pour `asset`, `bull`, `bear`, mais pas de validation que ces index sont dans les bounds de `symbols`. Si `asset=100` avec 8 symboles, `engine.py` catche l'`IndexError` silencieusement (L254) et skip la vue.

- **`tau`** : Float non borné. Des valeurs extrêmes peuvent causer des erreurs numériques.

- **`risk_free_rate`** : Float non borné. Un taux de -100 ou 1000 passera directement.

---

## 6. Performance

### Backend

- **N+1 requêtes Yahoo Finance** : `fetch_stock_data` fait un appel par ticker (dans un ThreadPoolExecutor, OK). Mais `yf.Ticker(t).info` fait 2-3 requêtes HTTP par ticker. Pour 8 tickers = ~24 requêtes. Plus `yf.Tickers(symbols).history()` = 1 requête. Plus `yf.Ticker(benchmark).history()` = 1. Plus FX si nécessaire. Total : ~30 requêtes HTTP par analyse.

- **Pas de cache pour le benchmark** : Le benchmark est téléchargé à chaque analyse même s'il ne change pas.

- **Pas de cache pour le risk-free rate** : `^IRX` est téléchargé à chaque analyse.

- **Cache de 6h** : Correct pour les données de prix, mais les fundamentals changent rarement. Un cache plus long (24h) serait plus approprié pour les market caps.

### Frontend

- **Bundle size** : `recharts` est importé en entier (LineChart, ScatterChart, BarChart, RadarChart, etc.). Pas de lazy loading. Le bundle doit faire ~500KB+ gzippé. Pour une app avec 4 onglets, seul l'onglet actif est rendu, mais toutes les librairies sont chargées.

- **`mergedEvol` recalculé à chaque render** : `evolData.map(...)` + `benchData.find(...)` est O(n*m) et est recalculé à chaque re-render (pas de `useMemo`). Avec 1260 jours de données, c'est 1260 × len(benchData) comparaisons.

- **`efData` recalculé à chaque render** : Même pattern, pas de `useMemo`.

- **Pas de `React.memo`** sur aucun sous-composant. Chaque `setQuotes`, `setTickers`, `setTheme` déclenche un re-render complet de l'arbre, y compris tous les graphiques Recharts.

---

## 7. DevOps

### Process management

- **Pas de systemd** : L'application est lancée via `uvicorn` directement. Pas de service systemd, pas de restart automatique, pas de logs persistants.

- **Pas de process manager** : Pas de gunicorn, pas de supervisor, pas de Docker. Un seul worker uvicorn.

- **`host="0.0.0.0"`** : Écoute sur toutes les interfaces. En production, devrait être derrière un reverse proxy et n'écouter que sur localhost.

### HTTPS

- **Pas de HTTPS** : Aucun certificat TLS, pas de configuration pour Let's Encrypt, pas de reverse proxy Nginx.

### Auth

- **Pas d'authentification** : Aucun mécanisme d'auth. N'importe qui peut appeler `/analyze` et consommer des ressources serveur.

### Observabilité

- **Pas de logging structuré** : Seuls les `traceback.print_exc()` vont sur stdout.

- **Pas de health check avancé** : `/health` retourne `{"status": "healthy"}` sans vérifier quoi que ce soit (pas de vérification de connexion Yahoo Finance, pas de vérification de cache, pas de vérification mémoire).

### Configuration

- **Pas de variable d'environnement** pour le port, le host, le TTL du cache, les origines CORS. Tout est hardcodé.

---

## 8. Top 5 des améliorations techniques prioritaires

### 1. Sécurité : Validation des inputs + Rate limiting + CORS restreint

C'est le plus urgent. Sans validation ni rate limiting, n'importe qui peut DDOS le serveur via `/analyze`.  
Actions :
- Ajouter un Pydantic validator sur `symbols` (max 20, regex `[A-Z0-9.]{1,10}`), `tau` (0.001-1.0), `min_weight`/`max_weight` (0-1), `risk_free_rate` (-0.1 à 1.0).
- Ajouter un middleware de rate limiting (ex: `slowapi`).
- Restreindre CORS à l'origine du frontend uniquement.

### 2. Gestion d'erreurs : Éliminer tous les `except:` nus

Le code est truffé de `except:` silencieux qui masquent les erreurs et retournent des valeurs par défaut trompeuses.  
Actions :
- Remplacer tous les `except:` nus par des `except Exception as e:` avec logging.
- Pour les erreurs de data (benchmark, FX, fundamentals), propager l'information au frontend (ex: `benchmark_available: false`).
- Supprimer les valeurs par défaut trompeuses (`market_return = 0.12`, `beta = 1.0`).

### 3. Cache : Ajouter un cache pour benchmark + FX + risk-free rate + auto-cleanup

Chaque analyse fait ~30 requêtes HTTP dont la moitié sont redondantes.  
Actions :
- Cacher le benchmark, le risk-free rate, et les taux de change séparément avec des TTL adaptés.
- Appeler `cache_cleanup()` automatiquement (ex: au démarrage + toutes les heures).
- Ajouter un verrou fichier (ou utiliser SQLite) pour éviter les race conditions.

### 4. Frontend : Réparer le mode manuel + useMemo + React.memo

Le mode manuel est cassé (setter manquant), les re-renders sont excessifs.  
Actions :
- Ajouter le setter pour `manualWeights`.
- Envelopper les données calculées (`mergedEvol`, `efData`, `corrLabels`, etc.) dans `useMemo`.
- Ajouter `React.memo` sur les sous-composants (KPI cards, charts).
- Lazy-loader les composants Recharts par onglet.

### 5. DevOps : systemd + reverse proxy + variables d'environnement

L'app est actuellement non déployable en production.  
Actions :
- Créer un service systemd pour uvicorn avec auto-restart.
- Ajouter un reverse proxy Nginx avec HTTPS (Let's Encrypt).
- Extraire la configuration (port, host, CORS origins, cache TTL) dans des variables d'environnement.
- Ajouter un fichier `.env` et utiliser `pydantic-settings` pour la config.

---

## Résumé des bugs

| # | Sévérité | Fichier | Description |
|---|----------|---------|-------------|
| 1 | 🔴 Critique | App.tsx L82 | `manualWeights` setter manquant — mode manuel inopérant |
| 2 | 🔴 Critique | engine.py L108 | `future.result()` sans try/except — crash silencieux si un ticker échoue |
| 3 | 🔴 Critique | main.py L36 | CORS `allow_origins=["*"]` + `allow_credentials=True` — incorrect et dangereux |
| 4 | 🟠 Élevée | engine.py L168 | Appel réseau dans la boucle walk-forward — extrêmement lent |
| 5 | 🟠 Élevée | engine.py L278-300 | `except:` nus — erreurs masquées, valeurs par défaut trompeuses |
| 6 | 🟠 Élevée | main.py L28-33 | Aucune validation des inputs — DDOS possible |
| 7 | 🟡 Moyenne | engine.py L136 | Double `reset_index()` — gaspillage mémoire |
| 8 | 🟡 Moyenne | App.tsx L170 | `useCallback` dépendance manquante sur `region` |
| 9 | 🟡 Moyenne | cache.py | Pas de verrou fichier — race condition |
| 10 | 🟡 Moyenne | App.css | Mode clair incomplet (variables manquantes, scrollbar, select options) |
| 11 | 🟡 Moyenne | stress_test.py L76 | CVaR = VaR × 1.25 — approximation non documentée |
| 12 | 🟡 Moyenne | stress_test.py L85 | Drawdown = choc × correlation_floor — pas de fondement mathématique |
| 13 | 🟢 Basse | App.tsx | Pas de `useMemo` sur les données calculées — re-renders inutiles |
| 14 | 🟢 Basse | App.tsx | Pas de `React.memo` sur les sous-composants |
| 15 | 🟢 Basse | App.css/index.css | Deux systèmes de CSS variables qui se chevauchent |
| 16 | 🟢 Basse | engine.py L350 | Frontière efficace avec `try/except: pass` — résultats silencieusement vides |

---

*Review terminée. Le projet est fonctionnel pour un MVP, mais nécessite les 5 améliorations prioritaires avant une mise en production.*