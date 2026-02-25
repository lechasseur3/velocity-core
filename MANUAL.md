# Guide d'Utilisation - VELOCITY CORE (v2.0)

Bienvenue dans **VELOCITY CORE**, votre moteur d'optimisation quantitatif de portefeuille professionnel. Cette plateforme combine le modèle mathématique de Black-Litterman avec une interface premium et des analyses en temps réel.

## 🚀 Installation et Lancement

L'application est entièrement automatisée. Vous pouvez tout lancer en une seule commande depuis le dossier racine.

### Commande Unique (Recommandé)
Ouvrez votre terminal dans le dossier `velocity core` et exécutez :
```bash
npm run dev
```
*(Alternative : vous pouvez exécuter directement `./dev.sh`)*

Cela lancera simultanément le **Backend Python** (port 8000) et le **Frontend React/Vite** (port 5173).

---

## 📊 Utilisation du Moteur (Tableau de Bord)

### 1. Univers Actif (Ajout/Suppression d'Actifs)
- **Ajout de Ticker** : Utilisez la barre de recherche intelligente (`Add Ticker...`) et tapez le symbole d'une action ou d'une crypto. L'application valide le ticker via Yahoo Finance avant de l'ajouter.
- **Suppression** : Cliquez sur la croix rouge à côté d'un actif pour le retirer.

### 2. Configuration des Signaux Alpha (Market Views)
Injectez votre expertise dans le modèle mathématique de Black-Litterman :
- **Vues Absolues (ABS)** : Vous fixez un rendement attendu pour un actif spécifique (ex: l'action va performer à 15%).
- **Vues Relatives (REL)** : Vous définissez un écart entre deux actifs (ex: AAPL va surperformer MSFT).
**Pour ajouter un signal :** Cliquez sur le bouton `+` dans la section *Alpha Signals*.

### 3. Contrôles Avancés
- **Internationalisation** : Utilisez le sélecteur **FR / EN** en haut à droite.
- **Capital d'Investissement** : Saisissez votre capital total dans l'en-tête central.
- **Mode d'Allocation** : 
  - **Auto (Black-Litterman)** : Optimisation mathématique ajustée en fonction de vos signaux et de la matrice de corrélation.
  - **Manual** : Définissez vos propres pourcentages pour chaque actif.
- **Simulateur de Gains (Projected Gains)** : Ajustez le curseur d'horizon d'investissement (1 à 30 ans) pour visualiser vos profits projetés en fonction du capital et du rendement calculé.

---

### 4. Lancer l'Analyse
Une fois vos paramètres définis, cliquez sur le bouton central **RUN ANALYSIS**. Le moteur va instantanément :
1. Récupérer l'historique de marché via l'API Yahoo Finance.
2. Traiter les données et calculer les covariances et la frontière efficiente.
3. Afficher les métriques de risque, de performance ainsi que l'exposition sectorielle.

*(Le bouton **AUTO-OPTIMIZE** permet de réinitialiser vos vues manuelles et de forcer un calcul optimal par le modèle de base.)*

---

## 📈 Interprétation des Résultats

Le tableau de bord (DASHBOARD) centralise une multitude de données :

- **Expected Alpha / Return** : Le rendement annuel attendu de ce portefeuille.
- **Capital at Risk (VaR 99%)** : La perte potentielle maximale sur 1 an, exprimée en valeur (dollars/euros).
- **Sharpe Performance** : La performance ajustée au risque (plus elle est élevée, meilleur est le rendement par rapport au risque pris).
- **Market Beta** : Indice de sensibilité par rapport au marché (Nasdaq-100 ou S&P 500).
- **Top/Flop du Jour** : Affiche les meilleures et les pires performances quotidiennes au sein de votre univers sélectionné.
- **Currency Exposure (Forex Risk)** : Une analyse de votre exposition aux devises (USD, EUR, etc.) selon la pondération de vos actifs.
- **Poids de la Stratégie & Frontière Efficiente** : Visualisations dynamiques pour évaluer la diversification et l'optimisation risque/rendement.
- **Strategy Backtest vs S&P 500** : Graphique qui retrace la performance passée de votre stratégie par rapport à l'indice historique de référence.

---

## 🛠️ Dépannage
- **Données manquantes ou `Invalid Ticker`** : Assurez-vous que le ticker entré existe bien sur Yahoo Finance (ex: `MC.PA` pour LVMH, `BTC-USD` pour le Bitcoin).
- **Graphiques non affichés** : Vérifiez que les terminaux tournent toujours et que l'API est accessible au port 8000. Vous devez obligatoirement disposer d'une connexion Internet pour le téléchargement des historiques boursiers.
