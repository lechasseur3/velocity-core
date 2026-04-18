import { useState, useCallback, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  AreaChart, Area, LineChart, Line, ScatterChart, Scatter,
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell
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

function formatPct(v: number) { return (v * 100).toFixed(2) + '%'; }
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
  const [manualWeights, setManualWeights] = useState<Record<string, string>>({});
  const [quotes, setQuotes] = useState<Record<string, QuoteData>>({});
  const [region, setRegion] = useState('US');
  const [activeTab, setActiveTab] = useState<'config' | 'results'>('config');
  const [selectedAsset, setSelectedAsset] = useState<string | null>(null);
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

  // Merge evolution data with benchmark
  const evolData = result?.historical_evolution?.map((r: any) => {
    const d: any = { Date: r.Date?.slice(0, 10) };
    result.assets.forEach(a => d[a] = r[a]);
    d['Portefeuille'] = r['Portefeuille'];
    return d;
  }) || [];

  const benchData = result?.benchmark_evolution || [];

  // Better benchmark merge: normalize by ticker name from benchData keys
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

  return (
    <div className="min-h-screen" style={{ background: '#0a0e1a' }}>
      <div className="max-w-7xl mx-auto px-4 py-6 md:px-8">

        {/* Header */}
        <div className="flex items-center justify-between mb-8 flex-wrap gap-4 no-print">
          <div>
            <h1 className="text-3xl md:text-4xl font-black tracking-tight" style={{ color: '#00d4ff' }}>
              ⚡ VELOCITY CORE
            </h1>
            <p className="text-xs mt-1" style={{ color: '#64748b' }}>
              Black-Litterman · Markowitz · Fama-French · VaR
            </p>
          </div>
          <div className="flex gap-2 region-buttons flex-wrap">
            {REGIONS.map(r => (
              <button key={r.code} onClick={() => setRegion(r.code)}
                className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                style={{
                  background: region === r.code ? 'rgba(0,212,255,0.2)' : 'rgba(255,255,255,0.03)',
                  color: region === r.code ? '#00d4ff' : '#64748b',
                  border: region === r.code ? '1px solid rgba(0,212,255,0.4)' : '1px solid rgba(255,255,255,0.06)'
                }}>
                {r.label}
              </button>
            ))}
          </div>
        </div>

        {/* Live Quotes Ticker — smooth horizontal scroll */}
        {Object.keys(quotes).length > 0 && (
          <div className="mb-4 overflow-hidden no-print" ref={tickerRef}>
            <div className="ticker-scroll">
              {[...tickers, ...tickers].map((t, idx) => {
                const q = quotes[t];
                if (!q || !q.price) return null;
                const up = q.change_pct >= 0;
                return (
                  <div key={`${t}-${idx}`} className="flex-shrink-0 px-3 py-2 rounded-lg"
                    style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <div className="text-xs font-bold" style={{ color: '#94a3b8' }}>{t}</div>
                    <div className="text-sm font-bold" style={{ color: '#e2e8f0' }}>{formatPrice(q.price, q.currency)}</div>
                    <div className="text-xs font-medium" style={{ color: up ? '#22c55e' : '#ef4444' }}>
                      {up ? '▲' : '▼'} {q.change_pct.toFixed(2)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Loader */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-16 no-print">
            <div className="loader-spinner mb-4"></div>
            <p className="text-sm font-medium" style={{ color: '#00d4ff' }}>Analyse en cours...</p>
            <div className="mt-4 space-y-2 w-64">
              <div className="skeleton-line" style={{ width: '80%' }}></div>
              <div className="skeleton-line" style={{ width: '60%' }}></div>
              <div className="skeleton-line" style={{ width: '90%' }}></div>
            </div>
          </div>
        )}

        {/* Config Card */}
        {!loading && (
          <div className="rounded-xl p-6 mb-6 no-print" style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)', backdropFilter: 'blur(20px)' }}>
            <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: '#00d4ff' }}>Configuration</h2>

            {/* Tickers */}
            <div className="mb-4">
              <label className="block text-xs mb-2" style={{ color: '#64748b' }}>Actifs</label>
              <div className="flex gap-2 mb-2 flex-wrap">
                {tickers.map(t => (
                  <span key={t} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold"
                    style={{ background: 'rgba(0,212,255,0.08)', color: '#00d4ff', border: '1px solid rgba(0,212,255,0.2)' }}>
                    {t}
                    <button onClick={() => removeTicker(t)} className="hover:opacity-70" style={{ color: '#ef4444' }}>✕</button>
                  </span>
                ))}
              </div>
              <div className="flex gap-2">
                <input value={tickerInput} onChange={e => setTickerInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && addTicker()}
                  placeholder="Ajouter (ex: MC.PA, SAP.DE)"
                  className="flex-1 px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }} />
                <button onClick={addTicker} className="px-4 py-2 rounded-lg text-sm font-bold"
                  style={{ background: 'rgba(0,212,255,0.15)', color: '#00d4ff', border: '1px solid rgba(0,212,255,0.3)' }}>+</button>
              </div>
            </div>

            {/* Params */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4 config-grid">
              <div>
                <label className="block text-xs mb-1" style={{ color: '#64748b' }}>Mode</label>
                <select value={isAuto ? 'auto' : 'manual'} onChange={e => setIsAuto(e.target.value === 'auto')}
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}>
                  <option value="auto">Auto (Sharpe max)</option>
                  <option value="manual">Manuel</option>
                </select>
              </div>
              <div>
                <label className="block text-xs mb-1" style={{ color: '#64748b' }}>Covariance</label>
                <select value={covMethod} onChange={e => setCovMethod(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}>
                  <option value="sample_cov">Sample</option>
                  <option value="ledoit_wolf">Ledoit-Wolf</option>
                  <option value="oracle_approximating">Oracle Approx.</option>
                </select>
              </div>
              <div>
                <label className="block text-xs mb-1" style={{ color: '#64748b' }}>Tau (BL): {tau}</label>
                <input type="range" min="0.01" max="0.5" step="0.01" value={tau}
                  onChange={e => setTau(parseFloat(e.target.value))} className="w-full mt-2 accent-cyan-400" />
              </div>
              <div>
                <label className="block text-xs mb-1" style={{ color: '#64748b' }}>Min poids %</label>
                <input type="number" value={minWeight} onChange={e => setMinWeight(parseInt(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }} />
              </div>
              <div>
                <label className="block text-xs mb-1" style={{ color: '#64748b' }}>Max poids %</label>
                <input type="number" value={maxWeight} onChange={e => setMaxWeight(parseInt(e.target.value) || 25)}
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }} />
              </div>
            </div>

            {/* BL Views */}
            <div className="mb-4">
              <div className="flex items-center gap-2 mb-2">
                <label className="text-xs font-bold uppercase" style={{ color: '#64748b' }}>Vues Black-Litterman</label>
                <button onClick={addView} className="text-xs px-2 py-1 rounded-lg" style={{ background: 'rgba(0,212,255,0.1)', color: '#00d4ff' }}>+ Ajouter</button>
              </div>
              {views.map((v, i) => (
                <div key={i} className="flex gap-2 mb-2 items-center flex-wrap">
                  <select value={v.type} onChange={e => updateView(i, 'type', e.target.value)}
                    className="px-2 py-1 rounded text-sm"
                    style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}>
                    <option value="A">Absolue</option>
                    <option value="R">Relative</option>
                  </select>
                  {v.type === 'A' ? (
                    <select value={v.asset ?? 0} onChange={e => updateView(i, 'asset', parseInt(e.target.value))}
                      className="px-2 py-1 rounded text-sm"
                      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}>
                      {tickers.map((t, j) => <option key={j} value={j}>{t}</option>)}
                    </select>
                  ) : (
                    <>
                      <select value={v.bull ?? 0} onChange={e => updateView(i, 'bull', parseInt(e.target.value))}
                        className="px-2 py-1 rounded text-sm"
                        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}>
                        {tickers.map((t, j) => <option key={j} value={j}>{t} ↑</option>)}
                      </select>
                      <select value={v.bear ?? 1} onChange={e => updateView(i, 'bear', parseInt(e.target.value))}
                        className="px-2 py-1 rounded text-sm"
                        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }}>
                        {tickers.map((t, j) => <option key={j} value={j}>{t} ↓</option>)}
                      </select>
                    </>
                  )}
                  <input type="number" value={v.value} onChange={e => updateView(i, 'value', parseFloat(e.target.value) || 0)}
                    step="0.5" placeholder="Rendement %"
                    className="w-20 px-2 py-1 rounded text-sm"
                    style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#e2e8f0' }} />
                  <button onClick={() => removeView(i)} style={{ color: '#ef4444' }}>✕</button>
                </div>
              ))}
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <button onClick={analyze} disabled={loading || tickers.length === 0}
                className="py-3 rounded-xl font-bold text-sm transition-all"
                style={{
                  background: loading ? 'rgba(255,255,255,0.05)' : 'linear-gradient(135deg, #00d4ff, #0099cc)',
                  color: loading ? '#64748b' : '#0a0e1a',
                  cursor: loading ? 'not-allowed' : 'pointer'
                }}>
                {loading ? '⟳ Analyse en cours...' : '▶ Analyser le portefeuille'}
              </button>
              <button onClick={generateOptimal} disabled={loading}
                className="py-3 rounded-xl font-bold text-sm transition-all"
                style={{
                  background: loading ? 'rgba(255,255,255,0.05)' : 'linear-gradient(135deg, #22c55e, #16a34a)',
                  color: loading ? '#64748b' : '#0a0e1a',
                  cursor: loading ? 'not-allowed' : 'pointer'
                }}>
                {loading ? '⟳ Génération...' : `✨ Portefeuille optimal — ${REGIONS.find(r => r.code === region)?.label}`}
              </button>
            </div>
            {error && <p className="mt-3 text-sm text-center" style={{ color: '#ef4444' }}>{error}</p>}
          </div>
        )}

        {/* Results */}
        {result && p && !loading && (
          <>
            {/* Export PDF button */}
            <div className="flex justify-end mb-4 no-print">
              <button onClick={exportPDF}
                className="px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2"
                style={{ background: 'rgba(255,255,255,0.05)', color: '#94a3b8', border: '1px solid rgba(255,255,255,0.1)' }}>
                📄 Exporter PDF
              </button>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6 kpi-grid">
              {[
                { label: 'Rendement Espéré', value: formatPct(p.expected_return), color: '#22c55e', icon: '📈' },
                { label: 'Volatilité', value: formatPct(p.volatility), color: '#eab308', icon: '📊' },
                { label: 'Ratio de Sharpe', value: p.sharpe.toFixed(3), color: '#00d4ff', icon: '⚡' },
                { label: 'Beta', value: p.beta.toFixed(3), color: '#8b5cf6', icon: '🎯' },
                { label: 'Alpha (Jensen)', value: formatPct(p.alpha), color: p.alpha >= 0 ? '#22c55e' : '#ef4444', icon: p.alpha >= 0 ? '✅' : '❌' },
              ].map(k => (
                <div key={k.label} className="kpi-card rounded-xl p-4 text-center"
                  style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div className="text-lg mb-1">{k.icon}</div>
                  <div className="text-xs mb-1" style={{ color: '#64748b' }}>{k.label}</div>
                  <div className="text-xl font-black" style={{ color: k.color }}>{k.value}</div>
                </div>
              ))}
            </div>

            {/* Risk Cards */}
            <div className="rounded-xl p-6 mb-6" style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: '#ef4444' }}>⚠️ Risque — VaR & CVaR</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[
                  { label: 'VaR Paramétrique 99%', value: formatPct(p.var_99_pct) },
                  { label: 'VaR Historique 99%', value: formatPct(p.var_historical) },
                  { label: 'VaR Cornish-Fisher 99%', value: formatPct(p.var_cornish_fisher) },
                  { label: 'CVaR Paramétrique', value: formatPct(p.cvar_parametric) },
                  { label: 'CVaR Historique', value: formatPct(p.cvar_historical) },
                  { label: 'CVaR Cornish-Fisher', value: formatPct(p.cvar_cornish_fisher) },
                ].map(r => (
                  <div key={r.label} className="p-3 rounded-lg" style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)' }}>
                    <div className="text-xs" style={{ color: '#64748b' }}>{r.label}</div>
                    <div className="text-lg font-bold" style={{ color: '#ef4444' }}>{r.value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Portfolio Evolution vs Benchmark */}
            <div className="rounded-xl p-6 mb-6" style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: '#00d4ff' }}>📈 Évolution du Portefeuille vs Benchmark</h2>
              <ResponsiveContainer width="100%" height={380}>
                <LineChart data={mergedEvol}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="Date" tick={{ fill: '#64748b', fontSize: 9 }} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: 8, color: '#e2e8f0' }} />
                  <Legend />
                  {result.assets.map((a, i) => (
                    <Line key={a} type="monotone" dataKey={a} stroke={COLORS[i % COLORS.length]} strokeWidth={1} dot={false} strokeOpacity={0.35} />
                  ))}
                  <Line type="monotone" dataKey="Portefeuille" stroke="#00d4ff" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="Benchmark" stroke="#ffffff" strokeWidth={2} strokeDasharray="8 4" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Efficient Frontier */}
            <div className="rounded-xl p-6 mb-6" style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: '#00d4ff' }}>🎯 Frontière Efficiente</h2>
              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" dataKey="vol" name="Volatilité" tick={{ fill: '#64748b' }} unit="%" label={{ value: 'Volatilité (%)', position: 'bottom', fill: '#64748b' }} />
                  <YAxis type="number" dataKey="ret" name="Rendement" tick={{ fill: '#64748b' }} unit="%" label={{ value: 'Rendement (%)', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                  <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: 8, color: '#e2e8f0' }} />
                  <Scatter data={efData} fill="#00d4ff" />
                  {p && efData.length > 0 && (
                    <Scatter data={[{ vol: p.volatility * 100, ret: p.expected_return * 100 }]} fill="#22c55e" />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Weights — clickable for asset info modal */}
            <div className="rounded-xl p-6 mb-6" style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: '#00d4ff' }}>⚖️ Poids Optimaux</h2>
              <div className="space-y-3">
                {result.assets.map((a, i) => {
                  const w = (result.weights[a] || 0) * 100;
                  const q = quotes[a];
                  return (
                    <div key={a} className="flex items-center gap-3 cursor-pointer group" onClick={() => setSelectedAsset(a)}>
                      <div className="w-24 flex items-center gap-2">
                        <span className="text-sm font-bold group-hover:underline" style={{ color: COLORS[i % COLORS.length] }}>{a}</span>
                        {q?.price && <span className="text-xs" style={{ color: '#64748b' }}>{q.price.toFixed(0)}</span>}
                      </div>
                      <div className="flex-1 h-7 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.04)' }}>
                        <div className="weight-bar h-full rounded-full flex items-center justify-end pr-2"
                          style={{ width: `${w}%`, background: `linear-gradient(90deg, ${COLORS[i % COLORS.length]}33, ${COLORS[i % COLORS.length]})`, minWidth: w > 0 ? '24px' : '0' }}>
                          <span className="text-xs font-bold" style={{ color: '#e2e8f0' }}>{w.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="text-xs mt-3" style={{ color: '#475569' }}>Cliquez sur un ticker pour voir les détails</p>
            </div>

            {/* Asset Info Modal */}
            {selectedAsset && selectedAssetInfo && (
              <div className="asset-modal-overlay" onClick={() => setSelectedAsset(null)}>
                <div className="asset-modal" onClick={e => e.stopPropagation()}>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-black" style={{ color: '#00d4ff' }}>{selectedAsset}</h3>
                    <button onClick={() => setSelectedAsset(null)} className="text-lg" style={{ color: '#64748b' }}>✕</button>
                  </div>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between"><span style={{ color: '#64748b' }}>Prix</span><span style={{ color: '#e2e8f0' }}>{formatPrice(selectedAssetInfo.price, selectedAssetInfo.currency)}</span></div>
                    <div className="flex justify-between"><span style={{ color: '#64748b' }}>Change</span><span style={{ color: selectedAssetInfo.change_percent >= 0 ? '#22c55e' : '#ef4444' }}>{selectedAssetInfo.change_percent.toFixed(2)}%</span></div>
                    <div className="flex justify-between"><span style={{ color: '#64748b' }}>P/E Ratio</span><span style={{ color: '#e2e8f0' }}>{selectedAssetInfo.pe_ratio != null ? selectedAssetInfo.pe_ratio.toFixed(1) : 'N/A'}</span></div>
                    <div className="flex justify-between"><span style={{ color: '#64748b' }}>Dividend Yield</span><span style={{ color: '#e2e8f0' }}>{selectedAssetInfo.dividend_yield ? (selectedAssetInfo.dividend_yield * 100).toFixed(2) + '%' : 'N/A'}</span></div>
                    <div className="flex justify-between"><span style={{ color: '#64748b' }}>Market Cap</span><span style={{ color: '#e2e8f0' }}>{selectedAssetInfo.mcap ? formatMcap(selectedAssetInfo.mcap) : 'N/A'}</span></div>
                  </div>
                </div>
              </div>
            )}

            {/* Correlation Matrix */}
            {result.correlation_matrix && result.correlation_matrix.length > 0 && (
              <div className="rounded-xl p-6 mb-6" style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}>
                <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: '#00d4ff' }}>🔢 Matrice de Corrélation</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr>
                        <th className="p-2" style={{ color: '#64748b' }}></th>
                        {corrLabels.map(l => <th key={l} className="p-2" style={{ color: '#64748b' }}>{l}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {result.correlation_matrix.map((row, i) => (
                        <tr key={i}>
                          <td className="p-2 font-bold" style={{ color: '#64748b' }}>{corrLabels[i]}</td>
                          {row.map((v, j) => {
                            const abs = Math.abs(v);
                            const bg = i === j ? 'rgba(0,212,255,0.15)' : v > 0 ? `rgba(0,212,255,${abs * 0.3})` : `rgba(239,68,68,${abs * 0.3})`;
                            return <td key={j} className="p-2 text-center rounded" style={{ background: bg, color: '#e2e8f0' }}>{v.toFixed(2)}</td>;
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
              <div className="rounded-xl p-6 mb-6" style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}>
                <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: '#00d4ff' }}>🧬 Exposition Fama-French 5 Facteurs</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={ffData}>
                    <PolarGrid stroke="#1e293b" />
                    <PolarAngleAxis dataKey="factor" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <PolarRadiusAxis tick={{ fill: '#475569', fontSize: 9 }} />
                    <Radar name="Exposition" dataKey="exposure" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.2} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Walk-Forward Backtest */}
            {wfResults.length > 0 && (
              <div className="rounded-xl p-6 mb-6" style={{ background: 'rgba(15,20,35,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}>
                <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: '#00d4ff' }}>🔄 Walk-Forward Backtest</h2>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
                  {[
                    { label: 'Rendement Moyen', value: formatPct(result.walk_forward_backtest.summary.avg_return), color: '#22c55e' },
                    { label: 'Volatilité Moyenne', value: formatPct(result.walk_forward_backtest.summary.avg_volatility), color: '#eab308' },
                    { label: 'Sharpe Moyen', value: result.walk_forward_backtest.summary.avg_sharpe.toFixed(3), color: '#00d4ff' },
                    { label: 'VaR 99% Moyen', value: formatPct(result.walk_forward_backtest.summary.avg_var_99), color: '#ef4444' },
                    { label: 'Max Drawdown', value: formatPct(result.walk_forward_backtest.summary.max_drawdown), color: '#ef4444' },
                  ].map(s => (
                    <div key={s.label} className="text-center p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)' }}>
                      <div className="text-xs" style={{ color: '#64748b' }}>{s.label}</div>
                      <div className="text-lg font-bold" style={{ color: s.color }}>{s.value}</div>
                    </div>
                  ))}
                </div>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={wfResults.map(r => ({
                    period: r.period_start?.slice(0, 10) || '',
                    return: r.return * 100,
                    sharpe: r.sharpe
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="period" tick={{ fill: '#64748b', fontSize: 9 }} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: 8, color: '#e2e8f0' }} />
                    <Bar dataKey="return" name="Rendement %" radius={[4, 4, 0, 0]}>
                      {wfResults.map((r, i) => (
                        <Cell key={i} fill={r.return >= 0 ? '#22c55e' : '#ef4444'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Footer */}
            <div className="text-center py-4 text-xs" style={{ color: '#475569' }}>
              Velocity Core v1.0 — Karl BAUJON — Black-Litterman · Markowitz · Fama-French
            </div>
          </>
        )}
      </div>
    </div>
  );
}