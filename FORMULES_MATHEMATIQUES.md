# 🧮 Velocity Core : Documentation des Modèles Mathématiques

L'application **Velocity Core** s'appuie sur plusieurs modèles issus de la finance quantitative moderne. Ce document explique *pourquoi* et *comment* ces mathématiques sont appliquées, de l'intuition pour les débutants aux détails techniques pour les quants.

---

## 1. Modèle de Black-Litterman (Allocation Bayésienne)
**Pourquoi on l'utilise :** Les modèles classiques (comme Markowitz pur) sont très instables : un léger changement dans les prévisions bouleverse totalement le portefeuille. Black-Litterman règle ce problème en utilisant une approche "bayésienne". On part d'un équilibre sain (le marché mondial) et on n'ajuste cet équilibre *que* si l'investisseur a une opinion très forte (un "signal Alpha").

**Pour un débutant :** Imagine que le marché est une recette de gâteau parfaite. Au lieu de réinventer la recette de zéro, on prend la recette de base, et on rajoute juste un peu de chocolat si on est sûr que ça sera meilleur.

*   **Rendements d'équilibre (Le "Prior" / La recette de base) :**
    `Π = δ * Σ * W_mkt`
    *   **δ (Aversion au risque, fixée à 2.5) :** La prime que le marché exige pour compenser la volatilité globale.
    *   **Σ (Matrice de Covariance) :** Comment les actions bougent les unes par rapport aux autres.
    *   **W_mkt (Poids de Marché) :** La taille de chaque entreprise (Capitalisation boursière). Plus l'entreprise est grosse (ex: Apple), plus le marché s'attend à ce qu'elle soit stable.

*   **Rendements espérés (Le "Posterior" / La recette ajustée) :**
    `E[R] = [(τ * Σ)^-1 + P^T * Ω^-1 * P]^-1 * [(τ * Σ)^-1 * Π + P^T * Ω^-1 * Q]`
    *   **P et Q (La Matrice des Vues) :** C'est ta conviction logicielle (Ex: "Je suis Haussier sur l'Action A (+5%)").
    *   **Ω (Oméga - Incertitude) :** C'est ton niveau de certitude. Plus la covariance est élevée, moins ton signal modifiera le portefeuille.
    *   **En bref :** Cette formule complexe est juste une "moyenne pondérée" extrêmement intelligente entre la réalité du marché (Π) et tes opinions (Q).

---

## 2. Optimisation de Portefeuille de Markowitz (Frontière Efficiente)
**Pourquoi on l'utilise :** Une fois que Black-Litterman nous a donné les *bons* rendements attendus (E[R]), Markowitz calcule mathématiquement la meilleure façon de répartir ton argent pour gagner le plus d'argent en prenant le moins de risques possibles.

**Pour un débutant :** C'est comme construire une équipe de sport. On ne met pas que des attaquants (trop risqué). On mélange des attaquants, des défenseurs et des milieux qui "s'équilibrent" (covariance) pour faire la meilleure équipe possible.

*   **Fonction de Maximisation (Ratio de Sharpe) :**
    `Max [ (W^T * E[R] - R_f) / √(W^T * Σ * W) ]`
    *   **Le numérateur (W^T * E[R] - R_f) :** C'est le rendement "pur" de tes actions, moins ce que tu aurais gagné en ne prenant aucun risque (R_f = Bon du trésor à 4%).
    *   **Le dénominateur √(W^T * Σ * W) :** C'est ton risque total d'équipe (la volatilité). Le "Sigma" (Σ) s'assure qu'on récompense la diversification (si A baisse, B monte).
    *   **En bref :** L'algorithme teste des milliers de répartitions possibles (W) pour trouver celle qui donne le plus grand "Ratio" GAGNS / RISQUES.

---

## 3. Paramètres de Performance du Portefeuille (Les KPI du Tableau de Bord)

Ces calculs sont les compteurs de vitesse de ton tableau de bord.

*   **Rendement Espéré du Portefeuille (R_p) :**
    `E[R_p] = Σ (W_i * E[R_i])`
    *   *Explication :* Simplement la moyenne pondérée de ce que chaque action est censée rapporter. Si tu as 50% d'Apple (qui fait +10%) et 50% de Microsoft (qui fait +4%), ton rendement est de 7%.

*   **Volatilité ou Écart-Type (σ_p) :**
    `σ_p = √(W^T * Σ * W)`
    *   *Explication :* Mesure l'amplitude des montagnes russes de ton portefeuille. 10% signifie que le portefeuille peut facilement faire -10% ou +10% en une année de façon normale.

*   **Ratio de Sharpe :**
    `Sharpe = (E[R_p] - R_f) / σ_p`
    *   *Explication :* Au-dessus de 1.0, tu es un très bon investisseur. En dessous de 0.5, tu prends trop de risques pour pas grand-chose.

---

## 4. Modèle d'Évaluation des Actifs Financiers (CAPM / MEDAF)
**Pourquoi on l'utilise :** Pour comprendre si ton intelligence artificielle a *vraiment* battu le marché S&P 500, ou si elle a juste eu de la chance parce que tout montait.

*   **Bêta du Marché (β) :**
    `β = Cov(R_p, R_m) / Var(R_m)`
    *   *Pour un débutant :* Est-ce que mon portefeuille ressemble au S&P 500 ? Un bêta de 1 signifie que tu copies le marché. Un bêta de 1.5 signifie que tu bouges 50% plus fort que le marché (à la hausse comme à la baisse).

*   **Alpha de Jensen (α) :**
    `α = E[R_p] - [ R_f + β * (E[R_m] - R_f) ]`
    *   *Pour un débutant :* C'est le Graal. C'est l'intelligence "pure" de ton algorithme. L'équation retire la performance due à la chance du marché (le β), pour isoler ce que ta stratégie précise a réellement apporté en plus.

---

## 5. Gestion des Risques : Value at Risk (VaR Paramétrique)
**Pourquoi on l'utilise :** Les banques l'exigent pour la réglementation. C'est la réponse mathématique à la question du CEO : "Combien d'argent puis-je perdre au maximum cette année si le pire se produit ?"

**Pour un débutant :** C'est le filet de sécurité estimé sur les lois de la probabilité normale (la fameuse courbe en cloche / cloche de Gauss).

*   **Formule VaR Paramétrique (99% de confiance) :**
    `VaR_99% = - (E[R_p] - 2.326 * σ_p)`
    *   *Explication technique :* Sur une distribution gaussienne des rendements, 99% des événements se trouvent au-dessus de `-2.326 écarts-types` par rapport à la moyenne `E[R_p]`. Si le résultat est -12%, cela veut dire qu'il y a 99% de chances que tu ne perdes *pas plus* de 12% l'année prochaine.

*   **Contribution Marginale au Risque :**
    `RC_i = [ W_i * (Σ * W)_i ] / σ_p^2`
    *   *Explication :* Si Apple représente 20% de mon portefeuille monétaire, représente-t-elle 20% de mon *risque* total ? Souvent non. Cette équation traque le vrai coupable si ton portefeuille est trop risqué.

---

## 6. Régression Multifactorielle Fama-French (5 Facteurs)
**Pourquoi on l'utilise :** Inventé par des prix Nobel (Eugène Fama). C'est pour analyser pourquoi un portefeuille a gagné de l'argent. Souvent, les gens croient avoir été "intelligents" (Alpha), alors qu'ils ont simplement acheté de toutes petites entreprises très risquées (SMB), ou des entreprises très rentables (RMW).

**Pour un débutant :** C'est un radar à Rayons-X qui décompose ton portefeuille pour trouver son "ADN" caché.

*   **Modèle de Régression Linéaire OLS :**
    `(R_p - R_f) = α + β_mkt*(R_m - R_f) + β_smb*SMB + β_hml*HML + β_rmw*RMW + β_cma*CMA + ε`
    *   **Le Moteur :** Le serveur prend l'historique complet de ton portefeuille, et trace une ligne droite statistique avec la méthode des "Moindres Carrés Ordinaires" via `statsmodels` en Python.
    *   **Facteur SMB (Small Minus Big) :** Est-ce que tu es exposé aux PME à fort potentiel ?
    *   **Facteur HML (High Minus Low) :** Est-ce que tu es un chasseur de bonnes affaires (titres "Value" bon marché) ?
    *   **Facteur RMW (Robust Minus Weak) :** Est-ce que ton portefeuille privilégie les entreprises très rentables (Google, Apple) ?
    *   **Facteur CMA (Conservative Minus Aggressive) :** Es-tu investi dans des entreprises qui font attention à leurs dépenses plutôt qu'en dilapidant leurs liquidités ?
