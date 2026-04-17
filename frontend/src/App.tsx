import { useState, useCallback } from 'react';
import axios from 'axios';
import {
  AreaChart, Area, ScatterChart, Scatter,
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell
} from 'recharts';

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
function formatNum(v: number) { return v.toFixed(4); }

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

  const analyze = useCallback(async () => {
    setLoading(true);
    setError('');
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
      });
      setResult(res.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [tickers, views, isAuto, manualWeights, covMethod, tau, minWeight, maxWeight]);

  const p = result?.performance;
  const efData = result ? result.efficient_frontier.vols.map((v, i) => ({ vol: v, ret: result.efficient_frontier.rets[i] })) : [];
  const evolData = result?.historical_evolution?.map((r: any) => {
    const d: any = { Date: r.Date?.slice(0, 10) };
    result.assets.forEach(a => d[a] = r[a]);
    return d;
  }) || [];
  const benchData = result?.benchmark_evolution || [];
  const mergedEvol = evolData.map((d: any) => {
    const bd = benchData.find((b: any) => b.Date?.slice(0, 10) === d.Date);
    return { ...d, SPY: bd?.SPY };
  });

  const corrLabels = result?.assets || [];
  const ffData = result?.fama_french ? Object.entries(result.fama_french).map(([k, v]) => ({ factor: k, exposure: v })) : [];

  const wfResults = result?.walk_forward_backtest?.results || [];

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight" style={{ color: 'var(--accent)' }}>
          VELOCITY CORE
        </h1>
        <p className="text-sm mt-2" style={{ color: 'var(--text-secondary)' }}>
          Quantitative Portfolio Optimization Engine — Black-Litterman · Markowitz · Fama-French
        </p>
      </div>

      {/* Config */}
      <div className="card glow mb-6">
        <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--accent)' }}>Configuration</h2>

        {/* Tickers */}
        <div className="mb-4">
          <label className="block text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Actifs</label>
          <div className="flex gap-2 mb-2 flex-wrap">
            {tickers.map(t => (
              <span key={t} className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium"
                style={{ background: 'rgba(0,212,255,0.15)', color: 'var(--accent)', border: '1px solid rgba(0,212,255,0.3)' }}>
                {t}
                <button onClick={() => removeTicker(t)} className="ml-1 hover:opacity-70" style={{ color: 'var(--red)' }}>✕</button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <input value={tickerInput} onChange={e => setTickerInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addTicker()}
              placeholder="Ajouter un ticker (ex: MC.PA)"
              className="flex-1 px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }} />
            <button onClick={addTicker} className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}>+</button>
          </div>
        </div>

        {/* Params */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Mode</label>
            <select value={isAuto ? 'auto' : 'manual'} onChange={e => setIsAuto(e.target.value === 'auto')}
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }}>
              <option value="auto">Auto (Sharpe max)</option>
              <option value="manual">Manuel</option>
            </select>
          </div>
          <div>
            <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Covariance</label>
            <select value={covMethod} onChange={e => setCovMethod(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }}>
              <option value="sample_cov">Sample</option>
              <option value="ledoit_wolf">Ledoit-Wolf</option>
              <option value="oracle_approximating">Oracle Approximating</option>
            </select>
          </div>
          <div>
            <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Tau (BL)</label>
            <input type="range" min="0.01" max="0.5" step="0.01" value={tau}
              onChange={e => setTau(parseFloat(e.target.value))} className="w-full mt-2" />
            <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{tau}</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs" style={{ color: 'var(--text-secondary)' }}>Min %</label>
              <input type="number" value={minWeight} onChange={e => setMinWeight(parseInt(e.target.value) || 0)}
                className="w-full px-2 py-1 rounded text-sm"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }} />
            </div>
            <div>
              <label className="block text-xs" style={{ color: 'var(--text-secondary)' }}>Max %</label>
              <input type="number" value={maxWeight} onChange={e => setMaxWeight(parseInt(e.target.value) || 25)}
                className="w-full px-2 py-1 rounded text-sm"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }} />
            </div>
          </div>
        </div>

        {/* BL Views */}
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <label className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Vues Black-Litterman</label>
            <button onClick={addView} className="text-xs px-2 py-1 rounded" style={{ background: 'rgba(0,212,255,0.15)', color: 'var(--accent)' }}>+ Ajouter</button>
          </div>
          {views.map((v, i) => (
            <div key={i} className="flex gap-2 mb-2 items-center flex-wrap">
              <select value={v.type} onChange={e => updateView(i, 'type', e.target.value)}
                className="px-2 py-1 rounded text-sm"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }}>
                <option value="A">Absolue</option>
                <option value="R">Relative</option>
              </select>
              {v.type === 'A' ? (
                <select value={v.asset ?? 0} onChange={e => updateView(i, 'asset', parseInt(e.target.value))}
                  className="px-2 py-1 rounded text-sm"
                  style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }}>
                  {tickers.map((t, j) => <option key={j} value={j}>{t}</option>)}
                </select>
              ) : (
                <>
                  <select value={v.bull ?? 0} onChange={e => updateView(i, 'bull', parseInt(e.target.value))}
                    className="px-2 py-1 rounded text-sm"
                    style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }}>
                    {tickers.map((t, j) => <option key={j} value={j}>{t} ↑</option>)}
                  </select>
                  <select value={v.bear ?? 1} onChange={e => updateView(i, 'bear', parseInt(e.target.value))}
                    className="px-2 py-1 rounded text-sm"
                    style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }}>
                    {tickers.map((t, j) => <option key={j} value={j}>{t} ↓</option>)}
                  </select>
                </>
              )}
              <input type="number" value={v.value} onChange={e => updateView(i, 'value', parseFloat(e.target.value) || 0)}
                step="0.5" placeholder="Rendement %"
                className="w-20 px-2 py-1 rounded text-sm"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-card)', color: 'var(--text-primary)' }} />
              <button onClick={() => removeView(i)} style={{ color: 'var(--red)' }}>✕</button>
            </div>
          ))}
        </div>

        <button onClick={analyze} disabled={loading || tickers.length === 0}
          className="w-full py-3 rounded-lg font-semibold text-sm transition-all"
          style={{
            background: loading ? 'var(--border-card)' : 'var(--accent)',
            color: loading ? 'var(--text-secondary)' : 'var(--bg-primary)',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}>
          {loading ? '⟳ Analyse en cours...' : '▶ Analyser le portefeuille'}
        </button>
        {error && <p className="mt-2 text-sm text-center" style={{ color: 'var(--red)' }}>{error}</p>}
      </div>

      {/* Results */}
      {result && p && (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
            {[
              { label: 'Rendement Espéré', value: formatPct(p.expected_return), color: 'var(--green)' },
              { label: 'Volatilité', value: formatPct(p.volatility), color: 'var(--yellow)' },
              { label: 'Ratio de Sharpe', value: p.sharpe.toFixed(3), color: 'var(--accent)' },
              { label: 'Beta', value: p.beta.toFixed(3), color: '#8b5cf6' },
              { label: 'Alpha (Jensen)', value: formatPct(p.alpha), color: p.alpha >= 0 ? 'var(--green)' : 'var(--red)' },
            ].map(k => (
              <div key={k.label} className="card text-center">
                <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{k.label}</div>
                <div className="text-xl font-bold" style={{ color: k.color }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Risk Cards */}
          <div className="card glow mb-6">
            <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--accent)' }}>Risque — VaR & CVaR</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {[
                { label: 'VaR Paramétrique 99%', value: formatPct(p.var_99_pct) },
                { label: 'VaR Historique 99%', value: formatPct(p.var_historical) },
                { label: 'VaR Cornish-Fisher 99%', value: formatPct(p.var_cornish_fisher) },
                { label: 'CVaR Paramétrique', value: formatPct(p.cvar_parametric) },
                { label: 'CVaR Historique', value: formatPct(p.cvar_historical) },
                { label: 'CVaR Cornish-Fisher', value: formatPct(p.cvar_cornish_fisher) },
              ].map(r => (
                <div key={r.label} className="p-3 rounded-lg" style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}>
                  <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{r.label}</div>
                  <div className="text-lg font-bold" style={{ color: 'var(--red)' }}>{r.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Portfolio Evolution vs Benchmark */}
          <div className="card glow mb-6">
            <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--accent)' }}>Évolution du Portefeuille vs Benchmark</h2>
            <ResponsiveContainer width="100%" height={350}>
              <AreaChart data={mergedEvol}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="Date" tick={{ fill: '#94a3b8', fontSize: 10 }} interval="preserveStartEnd" />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: 8 }} />
                <Legend />
                {result.assets.map((a, i) => (
                  <Area key={a} type="monotone" dataKey={a} stroke={COLORS[i % COLORS.length]} fill={COLORS[i % COLORS.length]} fillOpacity={0.1} />
                ))}
                <Area type="monotone" dataKey="SPY" stroke="#ffffff" fill="none" strokeWidth={2} strokeDasharray="5 5" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Efficient Frontier */}
          <div className="card glow mb-6">
            <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--accent)' }}>Frontière Efficiente</h2>
            <ResponsiveContainer width="100%" height={350}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" dataKey="vol" name="Volatilité" tick={{ fill: '#94a3b8' }} unit="%" />
                <YAxis type="number" dataKey="ret" name="Rendement" tick={{ fill: '#94a3b8' }} unit="%" />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: 8 }} />
                <Scatter data={efData} fill="#00d4ff" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Weights Table */}
          <div className="card glow mb-6">
            <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--accent)' }}>Poids Optimaux</h2>
            <div className="space-y-3">
              {result.assets.map((a, i) => {
                const w = (result.weights[a] || 0) * 100;
                return (
                  <div key={a} className="flex items-center gap-3">
                    <span className="w-16 text-sm font-medium">{a}</span>
                    <div className="flex-1 h-6 rounded-full overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
                      <div className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${w}%`, background: COLORS[i % COLORS.length], minWidth: w > 0 ? '4px' : '0' }} />
                    </div>
                    <span className="w-16 text-right text-sm font-medium" style={{ color: COLORS[i % COLORS.length] }}>{w.toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Correlation Matrix */}
          {result.correlation_matrix && result.correlation_matrix.length > 0 && (
            <div className="card glow mb-6">
              <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--accent)' }}>Matrice de Corrélation</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="p-2" style={{ color: 'var(--text-secondary)' }}></th>
                      {corrLabels.map(l => <th key={l} className="p-2" style={{ color: 'var(--text-secondary)' }}>{l}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {result.correlation_matrix.map((row, i) => (
                      <tr key={i}>
                        <td className="p-2 font-medium" style={{ color: 'var(--text-secondary)' }}>{corrLabels[i]}</td>
                        {row.map((v, j) => {
                          const intensity = Math.abs(v);
                          const color = v > 0 ? `rgba(0,212,255,${intensity * 0.6})` : `rgba(239,68,68,${intensity * 0.6})`;
                          return (
                            <td key={j} className="p-2 text-center" style={{ background: color, borderRadius: 2 }}>{v.toFixed(2)}</td>
                          );
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
            <div className="card glow mb-6">
              <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--accent)' }}>Exposition Fama-French 5 Facteurs</h2>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={ffData}>
                  <PolarGrid stroke="#1e293b" />
                  <PolarAngleAxis dataKey="factor" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 9 }} />
                  <Radar name="Exposition" dataKey="exposure" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.3} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Walk-Forward Backtest */}
          {wfResults.length > 0 && (
            <div className="card glow mb-6">
              <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--accent)' }}>Walk-Forward Backtest</h2>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
                {[
                  { label: 'Rendement Moyen', value: formatPct(result.walk_forward_backtest.summary.avg_return) },
                  { label: 'Volatilité Moyenne', value: formatPct(result.walk_forward_backtest.summary.avg_volatility) },
                  { label: 'Sharpe Moyen', value: result.walk_forward_backtest.summary.avg_sharpe.toFixed(3) },
                  { label: 'VaR 99% Moyen', value: formatPct(result.walk_forward_backtest.summary.avg_var_99) },
                  { label: 'Max Drawdown', value: formatPct(result.walk_forward_backtest.summary.max_drawdown) },
                ].map(s => (
                  <div key={s.label} className="text-center">
                    <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{s.label}</div>
                    <div className="text-lg font-bold" style={{ color: 'var(--accent)' }}>{s.value}</div>
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
                  <XAxis dataKey="period" tick={{ fill: '#94a3b8', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: 8 }} />
                  <Bar dataKey="return" name="Rendement %" fill="#00d4ff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  );
}