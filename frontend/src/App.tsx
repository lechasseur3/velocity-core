import { useState, useCallback, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  LineChart, Line, ScatterChart, Scatter,
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, Brush
} from 'recharts';
import './App.css';

const COLORS = ['#00d4ff', '#8b5cf6', '#22c55e', '#eab308', '#ef4444', '#f97316', '#ec4899', '#06b6d4', '#a855f7', '#84cc16'];

interface AssetInfo {
  ticker: string;
  mcap: number;
  price: number;
  change_percent: number;
  pe_ratio: number | null;
  dividend_yield: number;
  currency: string;
}

interface BLView {
  type: 'A' | 'R';
  asset?: number;
  bull?: number;
  bear?: number;
  value: number;
}

interface QuoteData {
  price: number;
  change_pct: number;
  currency: string;
  name: string;
}

interface AnalysisResult {
  weights: Record<string, number>;
  performance: {
    expected_return: number;
    volatility: number;
    sharpe: number;
    beta: number;
    alpha: number;
    var_99_pct: number;
    var_historical: number;
    var_cornish_fisher: number;
    cvar_parametric: number;
    cvar_historical: number;
    cvar_cornish_fisher: number;
  };
  risk_contribution: Record<string, number>;
  historical_evolution: Array<Record<string, any>>;
  benchmark_evolution: Array<Record<string, any>>;
  efficient_frontier: { vols: number[]; rets: number[] };
  correlation_matrix: number[][];
  assets: string[];
  fama_french: Record<string, number> | null;
  assets_info: Record<string, AssetInfo>;
  walk_forward_backtest: any;
}

function formatPrice(v: number, cur: string) {
  return new Intl.NumberFormat('fr-FR', { style: 'currency', currency: cur, minimumFractionDigits: 2 }).format(v);
}
function formatMcap(v: number) {
  if (v >= 1e12) return (v / 1e12).toFixed(2) + 'T';
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  return v.toLocaleString();
}

const REGIONS = [
  { code: 'US', label: '🇺🇸 USA', rf: 0.036 },
  { code: 'EU', label: '🇪🇺 Europe', rf: 0.028 },
  { code: 'FR', label: '🇫🇷 France', rf: 0.034 },
  { code: 'ASIA', label: '🇯🇵 Asie', rf: 0.022 },
  { code: 'GLOBAL', label: '🌍 Mondial', rf: 0.035 },
];

/* ─── Animated Counter ─────────────────────────────────── */
function AnimatedCounter({ value, decimals = 2, suffix = '' }: { value: number; decimals?: number; suffix?: string }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef(value);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    const start = ref.current;
    const end = value;
    const duration = 600;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out
      const current = start + (end - start) * eased;
      setDisplay(current);
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(animate);
      } else {
        ref.current = end;
      }
    };

    frameRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameRef.current);
  }, [value]);

  return <span className="vc-counter">{display.toFixed(decimals)}{suffix}</span>;
}

/* ─── Skeleton Card ────────────────────────────────────── */
function SkeletonCards() {
  return (
    <div className="vc-skeleton-cards">
      {[1, 2, 3, 4, 5].map(i => <div key={i} className="vc-skeleton-card" />)}
    </div>
  );
}

/* ─── Empty State ──────────────────────────────────────── */
function EmptyState({ onGoConfig }: { onGoConfig: () => void }) {
  return (
    <div className="vc-empty-state">
      <div className="vc-empty-icon">◈</div>
      <div className="vc-empty-title">Aucun résultat pour le moment</div>
      <div className="vc-empty-desc">Configurez vos actifs et lancez une analyse dans l'onglet Configuration.</div>
      <button onClick={onGoConfig} className="vc-cta vc-cta-primary" style={{ marginTop: 24, width: 'auto', display: 'inline-block' }}>
        ▸ Aller à la configuration
      </button>
    </div>
  );
}

/* ─── Gold Separator ───────────────────────────────────── */
function GoldSeparator() {
  return <hr className="vc-separator" />;
}

export default function App() {
  const [tickers, setTickers] = useState<string[]>(['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM']);
  const [tickerInput, setTickerInput] = useState('');
  const [isAuto, setIsAuto] = useState(true);
  const [tau, setTau] = useState(0.05);
  const [covMethod, setCovMethod] = useState('sample_cov');
  const [minWeight, setMinWeight] = useState(2);
  const [maxWeight, setMaxWeight] = useState(25);
  const [views, setViews] = useState<BLView[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [manualWeights] = useState<Record<string, string>>({});
  const [quotes, setQuotes] = useState<Record<string, QuoteData>>({});
  const [region, setRegion] = useState('US');
  const [activeTab, setActiveTab] = useState<'config' | 'results' | 'risk' | 'backtest'>('config');
  const [selectedAsset, setSelectedAsset] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const isLight = theme === 'light';
  const tickerRef = useRef<HTMLDivElement>(null);

  // Fetch real-time quotes
  useEffect(() => {
    if (tickers.length === 0) return;
    const timer = setTimeout(() => {
      axios.get(`/quotes?symbols=${tickers.join(',')}`).then(r => setQuotes(r.data)).catch(() => {});
    }, 300);
    return () => clearTimeout(timer);
  }, [tickers]);

  // Refresh quotes every 30s
  useEffect(() => {
    const iv = setInterval(() => {
      if (tickers.length > 0) {
        axios.get(`/quotes?symbols=${tickers.join(',')}`).then(r => setQuotes(r.data)).catch(() => {});
      }
    }, 30000);
    return () => clearInterval(iv);
  }, [tickers]);

  const addTicker = () => {
    const t = tickerInput.trim().toUpperCase();
    if (t && !tickers.includes(t)) {
      setTickers([...tickers, t]);
      setTickerInput('');
    }
  };

  const removeTicker = (t: string) => setTickers(tickers.filter(x => x !== t));

  const addView = () => setViews([...views, { type: 'A', value: 5 }]);

  const updateView = (i: number, field: string, val: any) => {
    const nv = [...views];
    (nv[i] as any)[field] = val;
    setViews(nv);
  };

  const removeView = (i: number) => setViews(views.filter((_, j) => j !== i));

  const getRf = () => REGIONS.find(r => r.code === region)?.rf || 0.04;

  const analyze = useCallback(async () => {
    setLoading(true);
    setError('');
    setActiveTab('results');
    try {
      const res = await axios.post<AnalysisResult>('/analyze', {
        symbols: tickers,
        views,
        is_auto: isAuto,
        manual_weights: isAuto ? null : Object.fromEntries(Object.entries(manualWeights).map(([k, v]) => [k, parseFloat(v) || 0])),
        cov_method: covMethod,
        tau,
        min_weight: minWeight / 100,
        max_weight: maxWeight / 100,
        risk_free_rate: getRf(),
      });
      setResult(res.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [tickers, views, isAuto, manualWeights, covMethod, tau, minWeight, maxWeight, region]);

  const generateOptimal = useCallback(async () => {
    setLoading(true);
    setError('');
    setActiveTab('results');
    try {
      const res = await axios.get<{ tickers: string[], result: AnalysisResult }>(`/optimal-portfolio?region=${region}`);
      setTickers(res.data.tickers);
      setResult(res.data.result);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Failed');
    } finally {
      setLoading(false);
    }
  }, [region]);

  const exportPDF = () => window.print();

  const p = result?.performance;
  const efData = result ? result.efficient_frontier.vols.map((v, i) => ({ vol: v * 100, ret: result.efficient_frontier.rets[i] * 100 })) : [];

  const evolData = result?.historical_evolution?.map((r: any) => {
    const d: any = { Date: r.Date?.slice(0, 10) };
    result.assets.forEach(a => d[a] = r[a]);
    d['Portefeuille'] = r['Portefeuille'];
    return d;
  }) || [];

  const benchData = result?.benchmark_evolution || [];
  const benchKeys = benchData.length > 0 ? Object.keys(benchData[0]).filter(k => k !== 'Date' && typeof benchData[0][k] === 'number') : [];
  const benchValueKey = benchKeys[0] || null;

  const mergedEvol = evolData.map((d: any) => {
    const bd = benchData.find((b: any) => b.Date?.slice(0, 10) === d.Date);
    return { ...d, Benchmark: bd && benchValueKey ? bd[benchValueKey] : undefined };
  });

  const corrLabels = result?.assets || [];
  const ffData = result?.fama_french ? Object.entries(result.fama_french).map(([k, v]) => ({ factor: k, exposure: v })) : [];
  const wfResults = result?.walk_forward_backtest?.results || [];
  const selectedAssetInfo = selectedAsset && result?.assets_info?.[selectedAsset] ? result.assets_info[selectedAsset] : null;

  // Theme-aware chart helpers
  const gridStroke = isLight ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.15)';
  const axisFill = isLight ? '#4a4a6a' : '#64748b';
  const labelFill = isLight ? '#8a8aa0' : '#475569';
  const chartTooltipStyle = {
    background: isLight ? 'rgba(255,255,255,0.98)' : 'rgba(11,15,25,0.96)',
    border: isLight ? '1px solid rgba(0,0,0,0.1)' : '1px solid rgba(201,168,76,0.2)',
    borderRadius: 12,
    padding: '12px 16px',
    color: isLight ? '#1a1a2e' : '#e8e8e8',
    fontSize: 12,
    boxShadow: isLight ? '0 8px 24px rgba(0,0,0,0.1)' : '0 8px 24px rgba(0,0,0,0.4)',
  };

  const navItems = [
    { key: 'config' as const, label: 'Configuration', icon: '▸' },
    { key: 'results' as const, label: 'Résultats', icon: '◆' },
    { key: 'risk' as const, label: 'Risque', icon: '●' },
    { key: 'backtest' as const, label: 'Backtest', icon: '◈' },
  ];

  return (
    <div className={`vc-layout ${theme === 'light' ? 'vc-light' : ''}`}>
      {/* Mobile sidebar overlay */}
      <div className={`vc-sidebar-overlay ${sidebarOpen ? 'open' : ''}`} onClick={() => setSidebarOpen(false)} />

      {/* Mobile toggle */}
      <button className="vc-mobile-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>☰</button>

      {/* ─── Sidebar ─────────────────────────────────────── */}
      <aside className={`vc-sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="vc-sidebar-logo">
          <h1>VELOCITY CORE</h1>
          <p>Portfolio Optimization Engine</p>
        </div>

        <nav className="vc-nav">
          {navItems.map(item => (
            <button key={item.key}
              className={`vc-nav-item ${activeTab === item.key ? 'active' : ''}`}
              onClick={() => { setActiveTab(item.key); setSidebarOpen(false); }}>
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>

        {/* Region in sidebar */}
        <div className="vc-sidebar-section">
          <div className="vc-sidebar-section-title">Région</div>
          <div className="vc-region-bar">
            {REGIONS.map(r => (
              <button key={r.code} onClick={() => setRegion(r.code)} className={`vc-region-btn ${region === r.code ? 'active' : ''}`}>
                {r.label}
              </button>
            ))}
          </div>
        </div>

        {/* Theme toggle at bottom */}
        <div className="vc-sidebar-bottom">
          <button className="vc-theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            <span className="toggle-icon">{theme === 'dark' ? '☀' : '☾'}</span>
            {theme === 'dark' ? 'Mode clair' : 'Mode sombre'}
          </button>
        </div>

        <div className="vc-sidebar-footer">
          Velocity Core v2.0 · Black-Litterman · Markowitz · Fama-French
        </div>
      </aside>

      {/* ─── Main Content ────────────────────────────────── */}
      <main className="vc-main">

        {/* ─── CONFIG TAB ─────────────────────────────────── */}
        {activeTab === 'config' && (
          <div className="vc-page">
            <div className="vc-section-header">
              <h2>Configuration</h2>
              <p>Construisez votre portefeuille, paramétrez l'optimisation et lancez l'analyse.</p>
            </div>

            <div className="vc-card">
              <div className="vc-card-title"><span className="title-icon">▸</span> Actifs sélectionnés</div>
              <div className="vc-chips">
                {tickers.map(t => (
                  <span key={t} className="vc-chip">
                    {t}
                    <button className="chip-remove" onClick={() => removeTicker(t)}>✕</button>
                  </span>
                ))}
              </div>
              <div className="vc-chip-input-row">
                <input value={tickerInput} onChange={e => setTickerInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && addTicker()}
                  placeholder="Ajouter un ticker (ex: MC.PA, SAP.DE)" />
                <button className="vc-chip-add-btn" onClick={addTicker}>+ Ajouter</button>
              </div>
            </div>

            {/* Parameters */}
            <div className="vc-card">
              <div className="vc-card-title"><span className="title-icon">◆</span> Paramètres d'optimisation</div>
              <div className="vc-config-grid">
                <div className="vc-config-group">
                  <div className="vc-config-group-title">Mode</div>
                  <div className="vc-field">
                    <label>Stratégie</label>
                    <select value={isAuto ? 'auto' : 'manual'} onChange={e => setIsAuto(e.target.value === 'auto')}>
                      <option value="auto">Auto — Sharpe maximum</option>
                      <option value="manual">Manuel — Poids définis</option>
                    </select>
                  </div>
                  <div className="vc-field">
                    <label>Covariance</label>
                    <select value={covMethod} onChange={e => setCovMethod(e.target.value)}>
                      <option value="sample_cov">Sample Covariance</option>
                      <option value="ledoit_wolf">Ledoit-Wolf Shrinkage</option>
                      <option value="oracle_approximating">Oracle Approximating</option>
                    </select>
                  </div>
                </div>

                <div className="vc-config-group">
                  <div className="vc-config-group-title">Contraintes</div>
                  <div className="vc-field">
                    <label>Poids minimum (%)</label>
                    <input type="number" value={minWeight} onChange={e => setMinWeight(parseInt(e.target.value) || 0)} />
                  </div>
                  <div className="vc-field">
                    <label>Poids maximum (%)</label>
                    <input type="number" value={maxWeight} onChange={e => setMaxWeight(parseInt(e.target.value) || 25)} />
                  </div>
                </div>

                <div className="vc-config-group">
                  <div className="vc-config-group-title">Black-Litterman</div>
                  <div className="vc-field">
                    <label>Tau (incertitude)</label>
                    <div className="vc-range-wrapper">
                      <input type="range" min="0.01" max="0.5" step="0.01" value={tau}
                        onChange={e => setTau(parseFloat(e.target.value))} />
                      <span className="vc-range-value">{tau.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* BL Views */}
            <div className="vc-card">
              <div className="vc-card-title"><span className="title-icon">●</span> Vues Black-Litterman</div>
              {views.length === 0 && (
                <p style={{ fontSize: '13px', color: 'var(--vc-text-muted)', marginBottom: 12 }}>Aucune vue définie. Ajoutez des vues pour exprimer vos convictions.</p>
              )}
              {views.map((v, i) => (
                <div key={i} className="vc-views-row">
                  <select value={v.type} onChange={e => updateView(i, 'type', e.target.value)}>
                    <option value="A">Absolue</option>
                    <option value="R">Relative</option>
                  </select>
                  {v.type === 'A' ? (
                    <select value={v.asset ?? 0} onChange={e => updateView(i, 'asset', parseInt(e.target.value))}>
                      {tickers.map((t, j) => <option key={j} value={j}>{t}</option>)}
                    </select>
                  ) : (
                    <>
                      <select value={v.bull ?? 0} onChange={e => updateView(i, 'bull', parseInt(e.target.value))}>
                        {tickers.map((t, j) => <option key={j} value={j}>{t} ↑</option>)}
                      </select>
                      <select value={v.bear ?? 1} onChange={e => updateView(i, 'bear', parseInt(e.target.value))}>
                        {tickers.map((t, j) => <option key={j} value={j}>{t} ↓</option>)}
                      </select>
                    </>
                  )}
                  <input type="number" value={v.value} onChange={e => updateView(i, 'value', parseFloat(e.target.value) || 0)}
                    step="0.5" placeholder="Rdt %" style={{ width: 80 }} />
                  <button className="vc-views-remove" onClick={() => removeView(i)}>✕</button>
                </div>
              ))}
              <button className="vc-views-add" onClick={addView}>+ Ajouter une vue</button>
            </div>

            {/* CTA Buttons */}
            <div className="vc-cta-row no-print">
              <button onClick={analyze} disabled={loading || tickers.length === 0}
                className="vc-cta vc-cta-primary">
                {loading ? '⟳ Analyse en cours...' : '▶ Analyser le portefeuille'}
              </button>
              <button onClick={generateOptimal} disabled={loading}
                className="vc-cta vc-cta-secondary">
                {loading ? '⟳ Génération...' : `✨ Portefeuille optimal — ${REGIONS.find(r => r.code === region)?.label}`}
              </button>
            </div>

            {error && <div className="vc-error">{error}</div>}
          </div>
        )}

        {/* ─── LOADER ─────────────────────────────────────── */}
        {loading && (
          <div className="vc-loader no-print">
            <div className="vc-spinner"></div>
            <p style={{ fontSize: '13px', fontWeight: 600, color: 'var(--vc-accent-gold)', marginBottom: 8 }}>Analyse en cours...</p>
            <SkeletonCards />
          </div>
        )}

        {/* ─── RESULTS TAB ────────────────────────────────── */}
        {activeTab === 'results' && result && p && !loading && (
          <div className="vc-page">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 0 }}>
              <div className="vc-section-header" style={{ marginBottom: 0, flex: 1 }}>
                <h2>Résultats</h2>
                <p>Performance, allocation et évolution du portefeuille optimisé.</p>
              </div>
              <button onClick={exportPDF} className="vc-export-btn no-print" title="Exporter PDF">⬙</button>
            </div>

            <GoldSeparator />

            {/* Live Ticker Strip */}
            {Object.keys(quotes).length > 0 && (
              <div className="vc-ticker-strip no-print" ref={tickerRef}>
                <div className="vc-ticker-scroll">
                  {[...tickers, ...tickers].map((t, idx) => {
                    const q = quotes[t];
                    if (!q || !q.price) return null;
                    const up = q.change_pct >= 0;
                    return (
                      <div key={`${t}-${idx}`} className="vc-ticker-item">
                        <div className="vc-ticker-name">{t}</div>
                        <div className="vc-ticker-price">{formatPrice(q.price, q.currency)}</div>
                        <div className={`vc-ticker-change ${up ? 'up' : 'down'}`}>
                          {up ? '▲' : '▼'} {q.change_pct.toFixed(2)}%
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* KPI Cards */}
            <div className="vc-kpi-grid">
              {[
                { label: 'Rendement Espéré', value: p.expected_return * 100, decimals: 2, suffix: '%', color: '#22c55e', icon: '▸' },
                { label: 'Volatilité', value: p.volatility * 100, decimals: 2, suffix: '%', color: '#eab308', icon: '◆' },
                { label: 'Ratio de Sharpe', value: p.sharpe, decimals: 3, suffix: '', color: '#00d4ff', icon: '⚡' },
                { label: 'Beta', value: p.beta, decimals: 3, suffix: '', color: '#8b5cf6', icon: '●' },
                { label: 'Alpha (Jensen)', value: p.alpha * 100, decimals: 2, suffix: '%', color: p.alpha >= 0 ? '#22c55e' : '#ef4444', icon: p.alpha >= 0 ? '◈' : '◈' },
              ].map(k => (
                <div key={k.label} className="vc-kpi" style={{ '--vc-glow': k.color + '20' } as any}>
                  <span className="vc-kpi-icon">{k.icon}</span>
                  <div className="vc-kpi-label">{k.label}</div>
                  <div className="vc-kpi-value" style={{ color: k.color }}>
                    <AnimatedCounter value={k.value} decimals={k.decimals} suffix={k.suffix} />
                  </div>
                </div>
              ))}
            </div>

            <GoldSeparator />

            {/* Portfolio Evolution vs Benchmark */}
            <div className="vc-card">
              <div className="vc-card-title"><span className="title-icon">▸</span> Évolution vs Benchmark</div>
              <div className="vc-chart-scroll">
                <div style={{ minWidth: Math.max(600, mergedEvol.length * 2) }}>
                  <ResponsiveContainer width="100%" height={380}>
                    <LineChart data={mergedEvol}>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                      <XAxis dataKey="Date" tick={{ fill: axisFill, fontSize: 10 }} interval="preserveStartEnd" />
                      <YAxis tick={{ fill: axisFill, fontSize: 10 }} />
                      <Tooltip contentStyle={chartTooltipStyle} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Brush dataKey="Date" height={30} stroke={isLight ? '#b8941f' : '#c9a84c'} fill={isLight ? 'rgba(184,148,31,0.05)' : 'rgba(201,168,76,0.05)'}
                        tickFormatter={(v: string) => v?.slice(0, 7)} />
                      {result.assets.map((a, i) => (
                        <Line key={a} type="monotone" dataKey={a} stroke={COLORS[i % COLORS.length]} strokeWidth={1} dot={false} strokeOpacity={0.35} />
                      ))}
                      <Line type="monotone" dataKey="Portefeuille" stroke={isLight ? '#b8941f' : '#c9a84c'} strokeWidth={3} dot={false} />
                      <Line type="monotone" dataKey="Benchmark" stroke={isLight ? '#4a4a6a' : '#94a3b8'} strokeWidth={2} strokeDasharray="8 4" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Efficient Frontier */}
            <div className="vc-card">
              <div className="vc-card-title"><span className="title-icon">◆</span> Frontière Efficiente</div>
              <div className="vc-chart">
                <ResponsiveContainer width="100%" height={350}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                    <XAxis type="number" dataKey="vol" name="Volatilité" tick={{ fill: axisFill, fontSize: 10 }} unit="%"
                      label={{ value: 'Volatilité (%)', position: 'bottom', fill: labelFill, fontSize: 10 }} />
                    <YAxis type="number" dataKey="ret" name="Rendement" tick={{ fill: axisFill, fontSize: 10 }} unit="%"
                      label={{ value: 'Rendement (%)', angle: -90, position: 'insideLeft', fill: labelFill, fontSize: 10 }} />
                    <Tooltip contentStyle={chartTooltipStyle}
                      formatter={(value: any, name: any) => [Number(value).toFixed(2) + '%', name]} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Scatter data={efData} fill={isLight ? '#b8941f' : '#c9a84c'} name="Frontière" />
                    {p && efData.length > 0 && (
                      <Scatter data={[{ vol: p.volatility * 100, ret: p.expected_return * 100 }]} fill="#22c55e" name="Portefeuille optimal" />
                    )}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            <GoldSeparator />

            {/* Weights */}
            <div className="vc-card">
              <div className="vc-card-title"><span className="title-icon">●</span> Poids Optimaux</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {result.assets.map((a, i) => {
                  const w = (result.weights[a] || 0) * 100;
                  const q = quotes[a];
                  const rc = (result.risk_contribution?.[a] || 0) * 100;
                  const isNarrow = w < 15;
                  return (
                    <div key={a} className="vc-weight-row" onClick={() => setSelectedAsset(a)}>
                      <div className="vc-weight-label" style={{ color: COLORS[i % COLORS.length] }}>{a}</div>
                      <div className="vc-weight-bar-bg">
                        <div className={`vc-weight-bar-fill ${isNarrow ? 'vc-bar-narrow' : ''}`}
                          style={{ width: `${Math.max(w, 2)}%`, background: `linear-gradient(90deg, ${COLORS[i % COLORS.length]}44, ${COLORS[i % COLORS.length]})` }}>
                          <span>{w.toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="vc-weight-tooltip">
                        <strong>{a}</strong> — Poids: {w.toFixed(2)}%<br />
                        Contribution risque: {rc.toFixed(2)}%
                        {q?.price && <><br />Prix: {formatPrice(q.price, q.currency)}<br />Change: {q.change_pct >= 0 ? '+' : ''}{q.change_pct.toFixed(2)}%</>}
                      </div>
                    </div>
                  );
                })}
              </div>
              <p style={{ fontSize: '12px', color: 'var(--vc-text-muted)', marginTop: 12 }}>Cliquez sur un ticker pour les détails</p>
            </div>

            {/* Asset Modal */}
            {selectedAsset && selectedAssetInfo && (
              <div className="vc-modal-overlay" onClick={() => setSelectedAsset(null)}>
                <div className="vc-modal" onClick={e => e.stopPropagation()}>
                  <div className="vc-modal-header">
                    <h3>{selectedAsset}</h3>
                    <button className="vc-modal-close" onClick={() => setSelectedAsset(null)}>✕</button>
                  </div>
                  <div className="vc-modal-row"><span className="label">Prix</span><span className="value">{formatPrice(selectedAssetInfo.price, selectedAssetInfo.currency)}</span></div>
                  <div className="vc-modal-row"><span className="label">Change</span><span className="value" style={{ color: selectedAssetInfo.change_percent >= 0 ? '#22c55e' : '#dc2626' }}>{selectedAssetInfo.change_percent >= 0 ? '+' : ''}{selectedAssetInfo.change_percent.toFixed(2)}%</span></div>
                  <div className="vc-modal-row"><span className="label">P/E Ratio</span><span className="value">{selectedAssetInfo.pe_ratio != null ? selectedAssetInfo.pe_ratio.toFixed(1) : 'N/A'}</span></div>
                  <div className="vc-modal-row"><span className="label">Dividend Yield</span><span className="value">{selectedAssetInfo.dividend_yield ? (selectedAssetInfo.dividend_yield * 100).toFixed(2) + '%' : 'N/A'}</span></div>
                  <div className="vc-modal-row"><span className="label">Market Cap</span><span className="value">{selectedAssetInfo.mcap ? formatMcap(selectedAssetInfo.mcap) : 'N/A'}</span></div>
                </div>
              </div>
            )}

            {/* Correlation Matrix */}
            {result.correlation_matrix && result.correlation_matrix.length > 0 && (
              <div className="vc-card">
                <div className="vc-card-title"><span className="title-icon">◈</span> Matrice de Corrélation</div>
                <div style={{ overflowX: 'auto' }}>
                  <table className="vc-corr-table">
                    <thead>
                      <tr>
                        <th></th>
                        {corrLabels.map(l => <th key={l}>{l}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {result.correlation_matrix.map((row, i) => (
                        <tr key={i}>
                          <td style={{ fontWeight: 700, color: 'var(--vc-text-muted)' }}>{corrLabels[i]}</td>
                          {row.map((v, j) => {
                            const abs = Math.abs(v);
                            const bg = i === j ? 'var(--vc-corr-self-bg)' : v > 0 ? `rgba(0,212,255,${abs * 0.2})` : `rgba(239,68,68,${abs * 0.2})`;
                            return <td key={j} style={{ background: bg }}>{v.toFixed(2)}</td>;
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Fama-French */}
            {ffData.length > 0 && (
              <div className="vc-card">
                <div className="vc-card-title"><span className="title-icon">◆</span> Exposition Fama-French 5 Facteurs</div>
                <div className="vc-chart">
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={ffData}>
                      <PolarGrid stroke={gridStroke} />
                      <PolarAngleAxis dataKey="factor" tick={{ fill: axisFill, fontSize: 11 }} />
                      <PolarRadiusAxis tick={{ fill: labelFill, fontSize: 9 }} />
                      <Radar name="Exposition" dataKey="exposure" stroke={isLight ? '#b8941f' : '#00d4ff'} fill={isLight ? '#b8941f' : '#00d4ff'} fillOpacity={0.15} strokeWidth={2} />
                      <Tooltip contentStyle={chartTooltipStyle} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            <GoldSeparator />

            {/* Footer */}
            <div className="vc-footer no-print">
              <div className="vc-footer-line1">Velocity Core · Karl BAUJON</div>
              <div className="vc-footer-line2">Black-Litterman · Markowitz · Fama-French · VaR/CVaR · Walk-Forward</div>
            </div>
          </div>
        )}

        {/* ─── RISK TAB ──────────────────────────────────── */}
        {activeTab === 'risk' && result && p && !loading && (
          <div className="vc-page">
            <div className="vc-section-header">
              <h2>Analyse de Risque</h2>
              <p>VaR, CVaR et métriques de risque avancées.</p>
            </div>

            <GoldSeparator />

            <div className="vc-card">
              <div className="vc-card-title danger"><span className="title-icon">●</span> VaR & CVaR</div>
              <div className="vc-risk-grid">
                {[
                  { label: 'VaR Paramétrique 99%', value: p.var_99_pct },
                  { label: 'VaR Historique 99%', value: p.var_historical },
                  { label: 'VaR Cornish-Fisher 99%', value: p.var_cornish_fisher },
                  { label: 'CVaR Paramétrique', value: p.cvar_parametric },
                  { label: 'CVaR Historique', value: p.cvar_historical },
                  { label: 'CVaR Cornish-Fisher', value: p.cvar_cornish_fisher },
                ].map(r => {
                  const isPositive = r.value >= 0;
                  return (
                    <div key={r.label} className={`vc-risk-item ${isPositive ? 'positive' : ''}`}>
                      <div className="vc-risk-label">{r.label}</div>
                      <div className={`vc-risk-value ${isPositive ? 'positive' : 'negative'}`}>
                        <AnimatedCounter value={r.value * 100} decimals={2} suffix="%" />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Risk contribution */}
            {result.risk_contribution && (
              <div className="vc-card">
                <div className="vc-card-title"><span className="title-icon">▸</span> Contribution au Risque</div>
                <div className="vc-chart">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={result.assets.map(a => ({
                      name: a,
                      contribution: (result.risk_contribution[a] || 0) * 100
                    }))} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                      <XAxis type="number" tick={{ fill: axisFill, fontSize: 10 }} unit="%" />
                      <YAxis type="category" dataKey="name" tick={{ fill: axisFill, fontSize: 11 }} width={70} />
                      <Tooltip contentStyle={chartTooltipStyle} formatter={(v: any) => [Number(v).toFixed(2) + '%', 'Contribution']} />
                      <Bar dataKey="contribution" name="Contribution Risque" radius={[0, 6, 6, 0]}>
                        {result.assets.map((_: any, i: number) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            <GoldSeparator />

            <div className="vc-footer no-print">
              <div className="vc-footer-line1">Velocity Core · Karl BAUJON</div>
              <div className="vc-footer-line2">Analyse de risque quantitatif</div>
            </div>
          </div>
        )}

        {/* ─── BACKTEST TAB ───────────────────────────────── */}
        {activeTab === 'backtest' && result && p && !loading && (
          <div className="vc-page">
            <div className="vc-section-header">
              <h2>Walk-Forward Backtest</h2>
              <p>Validation hors-échantillon par fenêtres glissantes.</p>
            </div>

            <GoldSeparator />

            {wfResults.length > 0 ? (
              <>
                <div className="vc-kpi-grid" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
                  {[
                    { label: 'Rendement Moyen', value: result.walk_forward_backtest.summary.avg_return * 100, suffix: '%', color: '#22c55e', icon: '▸' },
                    { label: 'Volatilité Moyenne', value: result.walk_forward_backtest.summary.avg_volatility * 100, suffix: '%', color: '#eab308', icon: '◆' },
                    { label: 'Sharpe Moyen', value: result.walk_forward_backtest.summary.avg_sharpe, suffix: '', color: '#00d4ff', icon: '⚡' },
                    { label: 'VaR 99% Moyen', value: result.walk_forward_backtest.summary.avg_var_99 * 100, suffix: '%', color: '#ef4444', icon: '●' },
                    { label: 'Max Drawdown', value: result.walk_forward_backtest.summary.max_drawdown * 100, suffix: '%', color: '#ef4444', icon: '◈' },
                  ].map(s => (
                    <div key={s.label} className="vc-kpi" style={{ '--vc-glow': s.color + '20' } as any}>
                      <span className="vc-kpi-icon">{s.icon}</span>
                      <div className="vc-kpi-label">{s.label}</div>
                      <div className="vc-kpi-value" style={{ color: s.color }}>
                        <AnimatedCounter value={s.value} decimals={2} suffix={s.suffix} />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="vc-card">
                  <div className="vc-card-title"><span className="title-icon">◆</span> Rendement par période</div>
                  <div className="vc-chart">
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={wfResults.map((r: any) => ({
                        period: r.period_start?.slice(0, 10) || '',
                        return: r.return * 100,
                        sharpe: r.sharpe
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                        <XAxis dataKey="period" tick={{ fill: axisFill, fontSize: 9 }} />
                        <YAxis tick={{ fill: axisFill, fontSize: 10 }} />
                        <Tooltip contentStyle={chartTooltipStyle}
                          formatter={(v: any) => [Number(v).toFixed(2) + '%', 'Rendement']} />
                        <Bar dataKey="return" name="Rendement %" radius={[6, 6, 0, 0]}>
                          {wfResults.map((r: any, i: number) => (
                            <Cell key={i} fill={r.return >= 0 ? '#22c55e' : '#ef4444'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            ) : (
              <div className="vc-card">
                <p style={{ color: 'var(--vc-text-muted)', fontSize: '13px' }}>Aucun résultat de backtest disponible. Lancez une analyse d'abord.</p>
              </div>
            )}

            <GoldSeparator />

            <div className="vc-footer no-print">
              <div className="vc-footer-line1">Velocity Core · Karl BAUJON</div>
              <div className="vc-footer-line2">Walk-Forward Out-of-Sample Validation</div>
            </div>
          </div>
        )}

        {/* Empty state for results/risk/backtest when no data */}
        {['results', 'risk', 'backtest'].includes(activeTab) && !result && !loading && (
          <div className="vc-page">
            <EmptyState onGoConfig={() => setActiveTab('config')} />
          </div>
        )}

      </main>
    </div>
  );
}