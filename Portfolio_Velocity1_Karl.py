"""
==============================================================

  ██╗   ██╗███████╗██╗      ██████╗  ██████╗██╗████████╗██╗   ██╗
  ██║   ██║██╔════╝██║     ██╔═══██╗██╔════╝██║╚══██╔══╝╚██╗ ██╔╝
  ██║   ██║█████╗  ██║     ██║   ██║██║     ██║   ██║    ╚████╔╝
  ╚██╗ ██╔╝██╔══╝  ██║     ██║   ██║██║     ██║   ██║     ╚██╔╝
   ╚████╔╝ ███████╗███████╗╚██████╔╝╚██████╗██║   ██║      ██║
    ╚═══╝  ╚══════╝╚══════╝ ╚═════╝  ╚═════╝╚═╝   ╚═╝      ╚═╝

  VELOCITY PORTFOLIO — Optimiseur de portefeuille boursier
  ─────────────────────────────────────────────────────────
  Auteur  : Karl BAUJON
  Licence : Custom (Utilisation commerciale interdite sans accord)
  GitHub  : https://github.com/lechasseur3

  ⚠️  AVERTISSEMENT LÉGAL :
  Ce logiciel est fourni à titre ÉDUCATIF uniquement.
  Il ne constitue PAS un conseil en investissement.
  Les performances passées ne préjugent pas des futures.
  Investir comporte un risque de perte en capital.

  ✅ INSTALLATION :
  pip install yfinance pypfopt plotly scipy numpy pandas rich
  pip install pandas-datareader statsmodels     ← pour Fama-French (recommandé)

  ▶️  UTILISATION :
  Modifie SYMBOLS ci-dessous puis lance : python Portfolio_Velocity.py
==============================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import scipy.stats as stats
from pypfopt import BlackLittermanModel, risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import concurrent.futures
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Modifie uniquement cette section
# ─────────────────────────────────────────────────────────────────────────────

# Tes actions. Exemples :
#   🇺🇸 US     : AAPL, MSFT, GOOGL, NVDA, TSLA, JPM, V, META
#   🇫🇷 France : MC.PA, TTE.PA, BNP.PA, OR.PA, AI.PA
#   🇩🇪 Allem. : SAP.DE, BMW.DE, ALV.DE
#   🌍 ETFs    : SPY, QQQ, IWDA.AS
SYMBOLS = ['AAPL', 'MSFT', 'MC.PA', 'OR.PA', 'SAP.DE', 'NVDA', 'META', 'NFLX']

PERIOD         = '5y'       # Période d'analyse : '1y' | '2y' | '5y'
BENCHMARK      = '^IXIC'    # Indice de référence : '^GSPC' | '^IXIC' | '^FCHI'
RISK_AVERSION  = 2.5        # Aversion au risque : 1 (risqué) → 5 (prudent)
INVESTMENT     = 100_000    # Montant investi (€/$) pour le calcul VaR
VAR_CONFIDENCE = 0.99       # Niveau de confiance VaR (99% = standard pro)
VAR_HORIZON    = 10         # Horizon VaR en jours
USE_FAMA_FRENCH = True      # Activer l'analyse factorielle FF5 (recommandé)

# ─────────────────────────────────────────────────────────────────────────────

pio.renderers.default = 'browser'
console = Console()
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CHARGEMENT DES DONNÉES
# ═════════════════════════════════════════════════════════════════════════════

def fetch_stock_fundamentals(ticker):
    """
    Récupère market cap et dette totale pour un ticker.
    Exécuté en parallèle pour accélérer le chargement.
    """
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        mcap = info.get('marketCap', 1e9)
        currency = info.get('currency', 'USD')
        try:
            total_debt = stock.balance_sheet.loc['Total Debt'].iloc[0]
        except Exception:
            total_debt = 0
        return ticker, mcap, total_debt, currency
    except Exception:
        return ticker, 1e9, 0, 'USD'


def load_all_data(symbols):
    """
    Télécharge en parallèle :
      - 5 ans de prix de clôture (Yahoo Finance)
      - Capitalisations boursières + dettes (pour Black-Litterman et ratio d'endettement)
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
    ) as progress:

        progress.add_task(description="[cyan]Téléchargement des prix historiques...", total=None)
        df = yf.download(symbols, period=PERIOD, auto_adjust=True, progress=False)['Close'].dropna()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        task2 = progress.add_task(description="[cyan]Récupération des fondamentaux (parallèle)...", total=len(symbols))
        mcaps, debts, currencies = {}, {}, {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_stock_fundamentals, t): t for t in symbols}
            for future in concurrent.futures.as_completed(futures):
                ticker, mcap, debt, currency = future.result()
                mcaps[ticker] = mcap
                debts[ticker] = debt
                currencies[ticker] = currency
                progress.advance(task2)

    # Nettoyer les symboles sans données
    available = [s for s in symbols if s in df.columns]
    if len(available) < len(symbols):
        missing = set(symbols) - set(available)
        console.print(f"[yellow]⚠ Symboles exclus (données manquantes) : {missing}[/yellow]")

    # Conversion des devises en USD
    unique_currencies = set([currencies.get(s, 'USD') for s in available]) - {'USD'}
    if unique_currencies:
        console.print(f"  [dim]Conversion en USD requise pour : {', '.join(unique_currencies)}[/dim]")
        fx_pairs = [f"{cur}USD=X" for cur in unique_currencies]
        fx_data = yf.download(fx_pairs, period=PERIOD, auto_adjust=True, progress=False)['Close']
        if isinstance(fx_data, pd.Series):
            fx_data = fx_data.to_frame(fx_pairs[0])
            
        for ticker in available:
            cur = currencies.get(ticker, 'USD')
            if cur != 'USD':
                fx_pair = f"{cur}USD=X"
                if fx_pair in fx_data.columns:
                    fx = fx_data[fx_pair].ffill().bfill()
                    aligned_fx = fx.reindex(df.index).ffill().bfill()
                    df[ticker] = df[ticker] * aligned_fx

    return df[available], mcaps, debts, available


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FAMA-FRENCH 5 FACTEURS
# ═════════════════════════════════════════════════════════════════════════════

def run_fama_french(df, symbols, rf):
    """
    Calcule les betas Fama-French 5 facteurs pour chaque action.
    Ces betas permettent de construire des views Black-Litterman
    fondées sur un modèle académique — pas sur l'intuition.

    Formule :
        E(Ri) - Rf = β_mkt(Rm-Rf) + β_smb·SMB + β_hml·HML + β_rmw·RMW + β_cma·CMA

    Facteurs (Kenneth French, Université de Chicago — données gratuites) :
        Mkt-RF → Prime de risque du marché
        SMB    → Small Minus Big   : small caps surperforment les large caps historiquement
        HML    → High Minus Low    : actions value surperforment les actions growth
        RMW    → Robust Minus Weak : entreprises profitables surperforment
        CMA    → Conservative vs Aggressive : investissement discipliné surperforme

    Plus le R² est élevé, plus le modèle explique bien l'action
    → plus tes views seront fiables dans Black-Litterman.
    """
    try:
        import pandas_datareader.data as web
        import statsmodels.api as sm
    except ImportError:
        console.print("[yellow]⚠ pandas-datareader ou statsmodels non installé.[/yellow]")
        console.print("[dim]  → pip install pandas-datareader statsmodels[/dim]")
        return None, None, None

    FACTOR_NAMES = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    try:
        ff5 = web.DataReader(
            'Global_5_Factors_Daily',
            'famafrench',
            df.index[0], df.index[-1]
        )[0] / 100  # % → décimales
        if ff5.index.tz is not None:
            ff5.index = ff5.index.tz_localize(None)
    except Exception as e:
        console.print(f"[yellow]⚠ Impossible de charger les facteurs FF5 : {e}[/yellow]")
        return None, None, None

    # Régression OLS pour chaque action
    stock_ret = df.pct_change().dropna()
    common    = stock_ret.index.intersection(ff5.index)
    stock_ret = stock_ret.loc[common]
    factors   = ff5.loc[common]

    betas        = {}
    expected_ff5 = {}
    premia       = {f: ff5[f].mean() * 252 for f in FACTOR_NAMES}
    rf_annual    = ff5['RF'].mean() * 252

    for ticker in symbols:
        if ticker not in stock_ret.columns:
            continue
        y = stock_ret[ticker] - factors['RF']        # Excès de rendement
        X = sm.add_constant(factors[FACTOR_NAMES])
        model = sm.OLS(y, X, missing='drop').fit()

        b = {
            'alpha' : model.params['const'] * 252,   # Annualisé
            'mkt'   : model.params['Mkt-RF'],
            'smb'   : model.params['SMB'],
            'hml'   : model.params['HML'],
            'rmw'   : model.params['RMW'],
            'cma'   : model.params['CMA'],
            'r2'    : model.rsquared,
        }
        betas[ticker] = b

        # E(Ri) = Rf + Σ βi × prime_du_facteur_i
        expected_ff5[ticker] = (
            rf_annual
            + b['mkt'] * premia['Mkt-RF']
            + b['smb'] * premia['SMB']
            + b['hml'] * premia['HML']
            + b['rmw'] * premia['RMW']
            + b['cma'] * premia['CMA']
        )

    return betas, expected_ff5, ff5


def display_ff5_table(betas, symbols):
    """Affiche le tableau des betas FF5 avec Rich."""
    table = Table(
        title="[bold]Fama-French 5 Facteurs — Analyse par action[/bold]",
        border_style="bright_blue", show_header=True, header_style="bold cyan"
    )
    for col in ["Ticker", "Alpha", "β Marché", "β SMB", "β HML", "β RMW", "β CMA", "R²", "Fiabilité"]:
        table.add_column(col, justify="right")

    for t in symbols:
        if t not in betas:
            continue
        b  = betas[t]
        r2 = b['r2']
        if r2 > 0.6:
            rel = "[green]✓ Haute[/green]"
        elif r2 > 0.4:
            rel = "[yellow]~ Moyenne[/yellow]"
        else:
            rel = "[red]✗ Faible[/red]"

        table.add_row(
            f"[bold]{t}[/bold]",
            f"{b['alpha']:+.2%}",
            f"{b['mkt']:.2f}",
            f"{b['smb']:.2f}",
            f"{b['hml']:.2f}",
            f"{b['rmw']:.2f}",
            f"{b['cma']:.2f}",
            f"{r2:.3f}",
            rel
        )

    console.print(table)
    console.print(
        "[dim]  SMB > 0 → small cap  |  HML > 0 → value  |"
        "  RMW > 0 → profitable  |  CMA > 0 → prudent  |"
        "  R² > 0.6 → view fiable[/dim]\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VUES INVESTISSEUR (INTERACTIF)
# ═════════════════════════════════════════════════════════════════════════════

def get_views_interactive(symbols, expected_ff5=None):
    """
    Propose deux modes pour définir les views Black-Litterman :

    ── Mode AUTO (Fama-French) ── Recommandé ──
      Les views sont calculées automatiquement à partir des betas FF5.
      → Rendements attendus basés sur un modèle académique validé.
      → Aucune intuition requise, le modèle fait le travail.

    ── Mode MANUEL ──
      Tu entres tes propres convictions de marché :
      - Vue absolue  : "AAPL va rapporter 10% cette année"
      - Vue relative : "MSFT va surperformer GOOGL de 3%"
      → Utile si tu as des informations spécifiques sur un secteur.

    Dans les deux cas, Black-Litterman combine tes views avec
    l'équilibre du marché (pondéré par les market caps) pour produire
    des rendements ajustés plus robustes que les données historiques seules.
    """
    n = len(symbols)

    console.print(Panel(
        "[bold cyan]CONFIGURATION DES VIEWS BLACK-LITTERMAN[/bold cyan]\n\n"
        "Les views représentent TES convictions sur le marché.\n"
        "Black-Litterman les combine avec l'équilibre du marché (market caps).",
        border_style="bright_blue"
    ))

    if expected_ff5:
        console.print("[bold green]Modes disponibles :[/bold green]")
        console.print("  [bold]A[/bold] → [green]Auto Fama-French 5[/green]  ← Recommandé : views académiques")
        console.print("  [bold]M[/bold] → [yellow]Manuel[/yellow]              ← Tu entres tes propres convictions\n")
        choice = console.input("[bold blue]Ton choix (A/M) [A par défaut] : [/bold blue]").strip().upper()
    else:
        choice = 'M'

    # ── Mode automatique FF5 ──────────────────────────────────────────────
    if choice != 'M' and expected_ff5:
        P = np.eye(n)
        Q = np.array([[expected_ff5[t]] for t in symbols])

        view_table = Table(title="Views auto-générées (Fama-French 5)", border_style="green")
        view_table.add_column("Action")
        view_table.add_column("Rendement attendu FF5", justify="right")
        view_table.add_column("Type")
        for t, q in zip(symbols, Q.flatten()):
            view_table.add_row(t, f"{q:.2%}", "Absolue (modèle)")
        console.print(view_table)
        console.print("[dim]→ Rendements calculés à partir des expositions factorielles de chaque action.[/dim]\n")
        return P, Q

    # ── Mode manuel ───────────────────────────────────────────────────────
    console.print("\n[bold yellow]Tes actions disponibles :[/bold yellow]")
    asset_table = Table(box=None, show_header=False)
    asset_table.add_column("ID", justify="right", style="bold cyan")
    asset_table.add_column("Symbole")
    for i, s in enumerate(symbols):
        asset_table.add_row(str(i), s)
    console.print(asset_table)

    console.print("\n[bold green]Types de views :[/bold green]")
    console.print("  [bold]A[/bold] = Absolue  → ex : AAPL va rapporter 10%")
    console.print("  [bold]R[/bold] = Relative → ex : MSFT surperforme GOOGL de 3%\n")

    while True:
        try:
            val = console.input("[bold blue]Combien de views ? (1-10) : [/bold blue]").strip()
            n_views = int(val) if val else 1
            if 1 <= n_views <= 10:
                break
        except ValueError:
            pass

    P_rows, Q_rows = [], []
    for v in range(1, n_views + 1):
        console.print(f"\n[bold yellow]── View {v} / {n_views}[/bold yellow]")
        vtype = console.input("  Type (A/R) : ").strip().upper()
        row   = np.zeros(n)

        if vtype == 'A':
            idx = int(console.input(f"  ID de l'action (0-{n-1}) : "))
            pct = float(console.input(f"  Rendement attendu pour {symbols[idx]} (%) : ").replace(',', '.'))
            row[idx] = 1.0
            q = pct / 100.0
            console.print(f"  [dim]→ {symbols[idx]} = {pct:.1f}%[/dim]")
        else:
            bull   = int(console.input("  ID de l'action qui surperforme : "))
            bear   = int(console.input("  ID de l'action qui sous-performe : "))
            spread = float(console.input("  Spread (%) : ").replace(',', '.'))
            row[bull] = 1.0
            row[bear] = -1.0
            q = spread / 100.0
            console.print(f"  [dim]→ {symbols[bull]} > {symbols[bear]} de {spread:.1f}%[/dim]")

        P_rows.append(row)
        Q_rows.append(q)

    return np.array(P_rows), np.array(Q_rows).reshape(-1, 1)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OPTIMISATION BLACK-LITTERMAN
# ═════════════════════════════════════════════════════════════════════════════

def optimize_portfolio(df, mcaps, P, Q, rf, symbols):
    """
    Pipeline d'optimisation complet :

    1. Matrice de covariance
       → Mesure comment chaque paire d'actions évolue ensemble.
         Une bonne diversification = choisir des actifs peu corrélés.

    2. Black-Litterman
       → Combine l'équilibre du marché (market caps) avec tes views.
         Résultat : des rendements attendus plus robustes que le seul historique.

    3. Frontière efficiente (Markowitz, Prix Nobel 1990)
       → Trouve les poids qui maximisent le ratio de Sharpe :
         Sharpe = (Rendement - Taux sans risque) / Volatilité
         Plus le Sharpe est élevé, mieux tu es rémunéré pour le risque.
    """
    S  = risk_models.sample_cov(df)
    bl = BlackLittermanModel(
        S, pi="market", market_caps=mcaps,
        risk_aversion=RISK_AVERSION, P=P, Q=Q
    )
    bl_returns = bl.bl_returns()

    ef = EfficientFrontier(bl_returns, S)
    ef.max_sharpe(risk_free_rate=rf)
    weights = ef.clean_weights()
    perf    = ef.portfolio_performance(risk_free_rate=rf, verbose=False)

    return weights, bl_returns, S, perf


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MÉTRIQUES DE RISQUE
# ═════════════════════════════════════════════════════════════════════════════

def compute_risk_metrics(df, weights, S, perf, rf, symbols, debts):
    """
    Calcule l'ensemble des métriques de risque du portefeuille optimisé.

    Beta
      Sensibilité de ton portefeuille au marché.
      Beta = 1   → suit le marché parfaitement
      Beta > 1   → plus volatile (potentiel + rendement, + risque)
      Beta < 1   → défensif (moins de gains, moins de pertes)

    Alpha (Jensen)
      Surperformance nette après ajustement au risque pris.
      Alpha > 0  → tu bats le marché, excellent !
      Alpha < 0  → tu sous-perfornes malgré le risque

    VaR — Value at Risk (méthode paramétrique normale)
      Perte maximale avec X% de confiance sur Y jours.
      Ex : VaR(99%, 10j) = 8 000 € → tu as 99% de chances
           de ne pas perdre plus de 8 000 € sur 10 jours.

    Max Drawdown
      Pire baisse historique depuis un pic sur la période.
      Ex : -35% → à un moment, le portefeuille a perdu 35% depuis son sommet.

    Contribution au risque
      Part de la variance totale imputable à chaque actif.
      Un portefeuille diversifié = contributions équilibrées.
    """
    w = pd.Series(weights)

    # ── Rendements journaliers portefeuille ───────────────────────────────
    pf_ret = (df.pct_change().dropna()[list(weights.keys())] * w).sum(axis=1)

    # ── Benchmark ─────────────────────────────────────────────────────────
    bm = yf.Ticker(BENCHMARK).history(period=PERIOD, auto_adjust=True)['Close'].dropna()
    if bm.index.tz is not None:
        bm.index = bm.index.tz_localize(None)
    bm_ret = bm.pct_change().dropna()

    aligned    = pd.concat([pf_ret, bm_ret], axis=1).dropna()
    rp_a, rm_a = aligned.iloc[:, 0], aligned.iloc[:, 1]

    beta  = np.cov(rp_a, rm_a)[0, 1] / np.var(rm_a)
    alpha = perf[0] - (rf + beta * (rm_a.mean() * 252 - rf))

    # ── Contribution au risque ────────────────────────────────────────────
    mc       = S @ w
    rc       = w * mc
    risk_pct = rc / (perf[1] ** 2)

    # ── VaR paramétrique ──────────────────────────────────────────────────
    daily_ret = perf[0] / 252
    daily_vol = perf[1] / np.sqrt(252)
    z         = stats.norm.ppf(VAR_CONFIDENCE)
    var_1d    = max(0, INVESTMENT * (z * daily_vol - daily_ret))
    days_ax   = np.arange(1, VAR_HORIZON + 1)
    var_curve = var_1d * np.sqrt(days_ax)

    # ── Max Drawdown ──────────────────────────────────────────────────────
    cumret = (1 + pf_ret).cumprod()
    max_dd = ((cumret - cumret.cummax()) / cumret.cummax()).min()

    # ── Ratio d'endettement top 3 ─────────────────────────────────────────
    top3              = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    d_total, mc_total = 0, 0
    for t, wt in top3:
        d_total  += wt * debts.get(t, 0)
        mc_total += wt * yf.Ticker(t).info.get('marketCap', 0)
    debt_ratio = (d_total / (d_total + mc_total) * 100) if (d_total + mc_total) > 0 else None

    return {
        'beta'      : beta,
        'alpha'     : alpha,
        'risk_pct'  : risk_pct,
        'var_1d'    : var_1d,
        'var_curve' : var_curve,
        'days_ax'   : days_ax,
        'max_dd'    : max_dd,
        'debt_ratio': debt_ratio,
        'pf_ret'    : pf_ret,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — AFFICHAGE TERMINAL (RICH)
# ═════════════════════════════════════════════════════════════════════════════

def display_results(weights, perf, risk, symbols):
    """Affiche le récapitulatif complet dans le terminal avec Rich."""

    # ── Performance globale ───────────────────────────────────────────────
    perf_table = Table(
        title="[bold]Performance du Portefeuille Optimal[/bold]",
        border_style="green", header_style="bold cyan"
    )
    perf_table.add_column("Indicateur", style="cyan")
    perf_table.add_column("Valeur", style="bold white", justify="right")
    perf_table.add_column("Interprétation", style="dim")

    sh_lbl = "✓ Excellent (>1)" if perf[2] > 1 else ("~ Correct (>0.5)" if perf[2] > 0.5 else "✗ Faible")
    bt_lbl = "Plus volatile que le marché" if risk['beta'] > 1.1 else ("Défensif" if risk['beta'] < 0.9 else "Suit le marché")
    al_lbl = "[green]✓ Surperformance[/green]" if risk['alpha'] > 0 else "[red]✗ Sous-performance[/red]"
    dd_lbl = "[green]Maîtrisé[/green]" if risk['max_dd'] > -0.20 else ("[yellow]Modéré[/yellow]" if risk['max_dd'] > -0.35 else "[red]Sévère[/red]")

    perf_table.add_row("Rendement annuel attendu", f"{perf[0]:.2%}", "")
    perf_table.add_row("Volatilité annuelle",       f"{perf[1]:.2%}", "Mesure le risque global")
    perf_table.add_row("Ratio de Sharpe",           f"{perf[2]:.2f}", sh_lbl)
    perf_table.add_row("Beta (vs marché)",          f"{risk['beta']:.4f}", bt_lbl)
    perf_table.add_row("Alpha annualisé",           f"{risk['alpha']:.2%}", al_lbl)
    perf_table.add_row("Max Drawdown historique",   f"{risk['max_dd']:.2%}", dd_lbl)
    perf_table.add_row(
        f"VaR {int(VAR_CONFIDENCE*100)}% / {VAR_HORIZON}j",
        f"{risk['var_curve'][-1]:,.0f} €/$",
        f"Sur {INVESTMENT:,} €/$ investis"
    )
    if risk['debt_ratio'] is not None:
        d = risk['debt_ratio']
        dl = "[green]Raisonnable[/green]" if d < 40 else ("[yellow]Modéré[/yellow]" if d < 60 else "[red]Élevé[/red]")
        perf_table.add_row("Endettement (top 3 positions)", f"{d:.1f}%", dl)
    console.print(perf_table)

    # ── Poids optimaux ────────────────────────────────────────────────────
    w_table = Table(
        title="[bold]Allocation Optimale[/bold]",
        border_style="bright_blue", header_style="bold cyan"
    )
    w_table.add_column("Action")
    w_table.add_column("Poids", justify="right")
    w_table.add_column("Répartition visuelle")
    w_table.add_column("Contribution risque", justify="right")

    for t in symbols:
        wt = weights.get(t, 0)
        rc = risk['risk_pct'].get(t, 0)
        filled = int(wt * 40)
        bar    = "[green]" + "█" * filled + "[/green][dim]" + "░" * (40 - filled) + "[/dim]"
        color  = "green" if wt > 0.1 else ("yellow" if wt > 0.02 else "dim")
        w_table.add_row(
            f"[bold]{t}[/bold]",
            f"[{color}]{wt:.2%}[/{color}]",
            bar,
            f"{rc:.2%}"
        )
    console.print(w_table)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DASHBOARD PLOTLY
# ═════════════════════════════════════════════════════════════════════════════

def build_dashboard(df, symbols, weights, perf, risk, bl_returns, S, rf):
    """
    Dashboard interactif complet — 7 graphiques en une seule page :

    1. Évolution Base 100 (5 ans)        → performance relative de chaque action
    2. Distribution des rendements       → forme statistique des gains/pertes journaliers
    3. Allocation optimale (camembert)   → répartition recommandée
    4. Contribution au risque            → qui domine le risque total
    5. Frontière efficiente + CML        → ensemble des meilleurs portefeuilles possibles
    6. Value at Risk dans le temps       → perte maximale probable sur 10 jours
    7. Rendements journaliers du portef. → volatilité visible dans le temps
    """
    df100         = df / df.iloc[0] * 100
    daily_returns = np.log(df / df.shift(1)).dropna()

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '📊 Évolution Base 100 (5 ans)',
            '📈 Distribution des Rendements Journaliers',
            '🥧 Allocation Optimale',
            '⚖️ Contribution au Risque',
            '📉 Frontière Efficiente & CML',
            f'⚠️ Value at Risk ({int(VAR_CONFIDENCE*100)}%) sur {VAR_HORIZON} jours',
            '〰️ Rendements Journaliers du Portefeuille',
            '🔮 Projection des Gains (Scénarios)',
        ),
        specs=[
            [{"type": "xy"},     {"type": "xy"}],
            [{"type": "domain"}, {"type": "xy"}],
            [{"type": "xy"},     {"type": "xy"}],
            [{"type": "xy"},     {"type": "xy"}],
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    # ── 1. Base 100 ───────────────────────────────────────────────────────
    for i, s in enumerate(symbols):
        fig.add_trace(go.Scatter(
            x=df100.index, y=df100[s], name=s,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5)
        ), row=1, col=1)

    # ── 2. Distribution ───────────────────────────────────────────────────
    for i, s in enumerate(symbols):
        fig.add_trace(go.Histogram(
            x=daily_returns[s], name=s, opacity=0.55,
            marker_color=COLORS[i % len(COLORS)], nbinsx=50, showlegend=False
        ), row=1, col=2)

    # ── 3. Allocation pie ─────────────────────────────────────────────────
    active     = {k: v for k, v in weights.items() if v > 0.001}
    pie_colors = [COLORS[symbols.index(s) % len(COLORS)] for s in active.keys()]
    fig.add_trace(go.Pie(
        labels=list(active.keys()), values=list(active.values()),
        hole=0.45, marker=dict(colors=pie_colors),
        textposition='inside', textinfo='label+percent'
    ), row=2, col=1)

    # ── 4. Contribution risque ────────────────────────────────────────────
    rc         = risk['risk_pct']
    bar_colors = [COLORS[symbols.index(s) % len(COLORS)] for s in symbols if s in rc]
    fig.add_trace(go.Bar(
        x=[s for s in symbols if s in rc],
        y=[rc[s] for s in symbols if s in rc],
        marker_color=bar_colors,
        text=[f"{rc[s]:.1%}" for s in symbols if s in rc],
        textposition='outside', showlegend=False
    ), row=2, col=2)

    # ── 5. Frontière efficiente + CML ─────────────────────────────────────
    ef_rets, ef_vols = [], []
    for t in np.linspace(float(bl_returns.min()), float(bl_returns.max()), 50):
        try:
            ef_t = EfficientFrontier(bl_returns, S)
            ef_t.efficient_return(t)
            r, v, _ = ef_t.portfolio_performance(risk_free_rate=rf, verbose=False)
            ef_rets.append(r)
            ef_vols.append(v)
        except Exception:
            pass

    if ef_vols:
        cml_x = np.linspace(0, max(ef_vols) * 1.3, 100)
        cml_y = rf + (perf[0] - rf) / perf[1] * cml_x

        fig.add_trace(go.Scatter(
            x=ef_vols, y=ef_rets, mode='lines', name='Frontière efficiente',
            line=dict(color='#2196F3', width=2.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=cml_x, y=cml_y, mode='lines', name='CML',
            line=dict(color='#FF5722', dash='dash', width=1.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=[perf[1]], y=[perf[0]], mode='markers+text',
            marker=dict(size=14, color='#FFD700', symbol='star'),
            text=['  ⭐ Optimal'], textfont=dict(color='#FFD700', size=12),
            name='Optimal'
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=[0], y=[rf], mode='markers+text',
            marker=dict(size=9, color='#4CAF50'),
            text=['  Rf'], textfont=dict(color='#4CAF50', size=11),
            name='Taux sans risque'
        ), row=3, col=1)

    # ── 6. VaR dans le temps ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=risk['days_ax'], y=risk['var_curve'],
        mode='lines+markers',
        fill='tozeroy', fillcolor='rgba(255,87,34,0.15)',
        line=dict(color='#FF5722', width=2),
        marker=dict(size=7),
        name=f"VaR {int(VAR_CONFIDENCE*100)}%"
    ), row=3, col=2)

    # ── 7. Rendements journaliers ─────────────────────────────────────────
    pf_ret = risk['pf_ret']
    bar_col = ['#4CAF50' if v >= 0 else '#f44336' for v in pf_ret]
    fig.add_trace(go.Bar(
        x=pf_ret.index, y=pf_ret.values,
        marker_color=bar_col, showlegend=False, name='Rendements'
    ), row=4, col=1)

    # ── 8. Projection des gains ───────────────────────────────────────────
    v0 = INVESTMENT
    years = np.arange(0, 31)
    mu = perf[0] - (perf[1]**2) / 2
    vol = perf[1]
    
    proj_median = v0 * np.exp(mu * years)
    proj_upper = v0 * np.exp(mu * years + 1.645 * vol * np.sqrt(years))
    proj_lower = v0 * np.exp(mu * years - 1.645 * vol * np.sqrt(years))
    
    fig.add_trace(go.Scatter(
        x=years, y=proj_upper, mode='lines', line=dict(width=0), showlegend=False,
        name='+95% Confiance', hoverinfo='y'
    ), row=4, col=2)
    fig.add_trace(go.Scatter(
        x=years, y=proj_lower, mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(76, 175, 80, 0.2)', showlegend=False,
        name='-95% Confiance', hoverinfo='y'
    ), row=4, col=2)
    fig.add_trace(go.Scatter(
        x=years, y=proj_median, mode='lines', line=dict(color='#4CAF50', width=3),
        name='Gains Espérés'
    ), row=4, col=2)
    
    step_years = [1, 2, 3, 5, 10, 15, 20, 30]
    steps = [dict(
        method="relayout",
        args=[{"xaxis8.range": [0, y]}],
        label=f"{y} ans"
    ) for y in step_years]

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        height=1500,
        title=dict(
            text=(
                f"<b>Velocity Portfolio — Executive Dashboard (by Karl BAUJON)</b><br>"
                f"<sub>Rendement {perf[0]:.2%} · Volatilité {perf[1]:.2%} · "
                f"Sharpe {perf[2]:.2f} · Beta {risk['beta']:.2f} · "
                f"Alpha {risk['alpha']:.2%} · "
                f"VaR {int(VAR_CONFIDENCE*100)}%/{VAR_HORIZON}j : {risk['var_curve'][-1]:,.0f} €/$</sub>"
            ),
            font=dict(size=17)
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='gray', borderwidth=1),
        barmode='overlay',
        sliders=[dict(
            active=4,  # Défaut : 10 ans
            currentvalue={"prefix": "Horizon d'investissement (Curseur) : "},
            pad={"t": 50},
            steps=steps,
            x=0.5,
            xanchor="center",
            yanchor="top",
            y=-0.05
        )]
    )
    fig.update_xaxes(tickformat='.1%', row=3, col=1)
    fig.update_xaxes(range=[0, 10], row=4, col=2)
    fig.update_yaxes(tickformat=',.0f', row=4, col=2)
    fig.update_yaxes(tickformat='.1%', row=3, col=1)
    fig.update_yaxes(tickformat=',.0f', row=3, col=2)

    fig.show()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    console.print(Panel(
        "[bold cyan]VELOCITY PORTFOLIO — ENGINE (Crafted by Karl BAUJON)[/bold cyan]\n"
        "[dim]Black-Litterman · Fama-French 5F · Frontière Efficiente · VaR[/dim]",
        border_style="bright_blue", expand=False
    ))

    # ── Étape 1 : Données ─────────────────────────────────────────────────
    console.rule("[bold cyan]ÉTAPE 1 — Chargement des données[/bold cyan]")
    df, mcaps, debts, symbols = load_all_data(SYMBOLS)
    rf = yf.Ticker('^IRX').history(period='5d')['Close'].dropna().mean() / 100
    console.print(f"  Taux sans risque (^IRX) : [bold]{rf:.2%}[/bold]\n")

    # ── Étape 2 : Fama-French ─────────────────────────────────────────────
    console.rule("[bold cyan]ÉTAPE 2 — Analyse Fama-French 5 facteurs[/bold cyan]")
    betas, expected_ff5, ff5 = None, None, None

    if USE_FAMA_FRENCH:
        console.print("  Calcul des betas factoriels...\n")
        betas, expected_ff5, ff5 = run_fama_french(df, symbols, rf)
        if betas:
            display_ff5_table(betas, symbols)
        else:
            console.print("[yellow]  ⚠ FF5 indisponible → passage en mode manuel[/yellow]\n")
    else:
        console.print("  [dim]Fama-French désactivé (USE_FAMA_FRENCH = False)[/dim]\n")

    # ── Étape 3 : Views ───────────────────────────────────────────────────
    console.rule("[bold cyan]ÉTAPE 3 — Views Black-Litterman[/bold cyan]")
    P, Q = get_views_interactive(symbols, expected_ff5)

    # ── Étape 4 : Optimisation ────────────────────────────────────────────
    console.rule("[bold cyan]ÉTAPE 4 — Optimisation du portefeuille[/bold cyan]")
    console.print("  Calcul en cours (frontière efficiente + max Sharpe)...")
    weights, bl_returns, S, perf = optimize_portfolio(df, mcaps, P, Q, rf, symbols)
    console.print(f"  [green]✓ Terminé — Sharpe : {perf[2]:.2f}[/green]\n")

    # ── Étape 5 : Risque ──────────────────────────────────────────────────
    console.rule("[bold cyan]ÉTAPE 5 — Analyse du risque[/bold cyan]")
    console.print("  Calcul de Beta, Alpha, VaR, Drawdown, Contribution risque...")
    risk = compute_risk_metrics(df, weights, S, perf, rf, symbols, debts)
    console.print(f"  [green]✓ Terminé[/green]\n")

    # ── Étape 6 : Résultats ───────────────────────────────────────────────
    console.rule("[bold cyan]RÉSULTATS[/bold cyan]")
    display_results(weights, perf, risk, symbols)

    # ── Étape 7 : Dashboard ───────────────────────────────────────────────
    console.rule("[bold cyan]ÉTAPE 6 — Dashboard interactif[/bold cyan]")
    console.print("  Génération des graphiques...")
    build_dashboard(df, symbols, weights, perf, risk, bl_returns, S, rf)

    console.print(f"\n[dim]  Analyse complète en {time.time() - t0:.1f} secondes.[/dim]")
    console.print(Panel(
        "[bold green]✅ Analyse terminée ![/bold green]\n"
        "[dim]Le dashboard s'est ouvert dans ton navigateur.[/dim]",
        border_style="green", expand=False
    ))


if __name__ == "__main__":
    main()