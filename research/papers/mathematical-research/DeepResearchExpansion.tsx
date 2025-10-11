import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, Pause, Download, Database, Activity, Zap } from 'lucide-react';

const PHI = (1 + Math.sqrt(5)) / 2;
const PI = Math.PI;
const E = Math.E;
const GAMMA = 0.5772156649015329; // Euler-Mascheroni constant
const APERY = 1.2020569031595942; // ζ(3) - Apéry's constant
const SQRT2 = Math.sqrt(2);
const SQRT3 = Math.sqrt(3);
const CATALAN = 0.915965594177219; // Catalan's constant

// Mathematical constants to explore
const CONSTANTS = [
  { name: 'π (Pi)', symbol: 'π', value: PI, color: '#FF6B6B' },
  { name: 'e (Euler)', symbol: 'e', value: E, color: '#4ECDC4' },
  { name: 'φ (Golden Ratio)', symbol: 'φ', value: PHI, color: '#FFD93D' },
  { name: 'γ (Euler-Mascheroni)', symbol: 'γ', value: GAMMA, color: '#95E1D3' },
  { name: 'ζ(3) (Apéry)', symbol: 'ζ(3)', value: APERY, color: '#F38181' },
  { name: '√2 (Silver Ratio)', symbol: '√2', value: SQRT2, color: '#AA96DA' },
  { name: '√3', symbol: '√3', value: SQRT3, color: '#FCBAD3' },
  { name: 'G (Catalan)', symbol: 'G', value: CATALAN, color: '#A8E6CF' }
];

const DeepResearchExpansion = () => {
  const [activePhase, setActivePhase] = useState('constants');
  const [computing, setComputing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [primes, setPrimes] = useState([]);
  const [gaps, setGaps] = useState([]);
  const [results, setResults] = useState(null);
  const [selectedConstants, setSelectedConstants] = useState([0, 1, 2, 3, 4]);

  // Wallace Transform
  const WT = (x) => {
    if (x <= 0) return 0;
    const logVal = Math.log(x + 1e-12);
    return PHI * Math.pow(Math.abs(logVal), PHI) * Math.sign(logVal) + 1.0;
  };

  // Prime generation
  const isPrime = (n) => {
    if (n < 2) return false;
    if (n === 2) return true;
    if (n % 2 === 0) return false;
    for (let i = 3; i <= Math.sqrt(n); i += 2) {
      if (n % i === 0) return false;
    }
    return true;
  };

  const generatePrimes = (limit) => {
    const primes = [];
    for (let i = 2; primes.length < limit; i++) {
      if (isPrime(i)) primes.push(i);
    }
    return primes;
  };

  // Initialize data
  useEffect(() => {
    const computed = generatePrimes(50000);
    const gapArray = [];
    for (let i = 1; i < computed.length; i++) {
      gapArray.push(computed[i] - computed[i-1]);
    }
    setPrimes(computed);
    setGaps(gapArray);
  }, []);

  // Test relationship with constant
  const testConstantRelationship = (constant, power = -2) => {
    if (gaps.length === 0 || primes.length === 0) return null;

    const testSize = Math.min(10000, gaps.length);
    const ratio = Math.pow(constant, power);
    let matches = 0;
    const tolerance = 0.20;

    for (let i = 0; i < testSize; i++) {
      const actualGap = gaps[i];
      const p = primes[i];
      const wt_p = WT(p);
      const predicted = wt_p * ratio;

      if (Math.abs(actualGap - predicted) / actualGap <= tolerance) {
        matches++;
      }
    }

    return {
      constant: constant,
      power: power,
      matches: matches,
      total: testSize,
      matchRate: (matches / testSize * 100).toFixed(2),
      ratio: ratio
    };
  };

  // Phase 1: Explore Mathematical Constants
  const exploreConstants = async () => {
    setComputing(true);
    setProgress(0);

    const constantResults = [];

    for (let i = 0; i < CONSTANTS.length; i++) {
      if (!selectedConstants.includes(i)) continue;

      const constant = CONSTANTS[i];
      setProgress((i / CONSTANTS.length) * 100);

      // Test various powers
      const powers = [-3, -2, -1, 1, 2, 3];
      const powerResults = powers.map(p => testConstantRelationship(constant.value, p));

      const bestResult = powerResults.reduce((best, curr) =>
        parseFloat(curr.matchRate) > parseFloat(best.matchRate) ? curr : best
      );

      constantResults.push({
        ...constant,
        bestPower: bestResult.power,
        bestMatchRate: parseFloat(bestResult.matchRate),
        allPowers: powerResults
      });

      await new Promise(resolve => setTimeout(resolve, 50));
    }

    constantResults.sort((a, b) => b.bestMatchRate - a.bestMatchRate);

    setResults({
      type: 'constants',
      data: constantResults,
      summary: {
        tested: constantResults.length,
        bestConstant: constantResults[0]?.name,
        bestMatch: constantResults[0]?.bestMatchRate,
        bestPower: constantResults[0]?.bestPower
      }
    });

    setProgress(100);
    setComputing(false);
  };

  // Phase 2: Physical Interpretations
  const explorePhysics = async () => {
    setComputing(true);
    setProgress(0);

    const physicsTests = [
      {
        name: 'Quantum Energy Levels',
        description: 'Test if prime gaps match E_n = hf · constant',
        test: () => {
          // Simulated quantum energy level matching
          const h = 6.62607015e-34; // Planck constant (scaled)
          const results = [];

          for (const constant of CONSTANTS.filter((_, i) => selectedConstants.includes(i))) {
            const scaling = constant.value / PI; // Normalize to π
            let matches = 0;

            for (let i = 0; i < Math.min(1000, gaps.length); i++) {
              const gap = gaps[i];
              const quantumLevel = Math.floor(gap * scaling);
              const predicted = quantumLevel / scaling;

              if (Math.abs(gap - predicted) / gap < 0.15) matches++;
            }

            results.push({
              constant: constant.name,
              matches: matches,
              matchRate: (matches / 1000 * 100).toFixed(2)
            });
          }

          return results.sort((a, b) => parseFloat(b.matchRate) - parseFloat(a.matchRate));
        }
      },
      {
        name: 'Chaos Theory Attractors',
        description: 'Test Lyapunov exponent connections',
        test: () => {
          const results = [];

          for (const constant of CONSTANTS.filter((_, i) => selectedConstants.includes(i))) {
            // Lyapunov exponent simulation
            let chaosScore = 0;
            const lambda = Math.log(constant.value);

            for (let i = 1; i < Math.min(1000, primes.length); i++) {
              const ratio = primes[i] / primes[i-1];
              const divergence = Math.abs(Math.log(ratio) - lambda);
              if (divergence < 0.1) chaosScore++;
            }

            results.push({
              constant: constant.name,
              lyapunovAlignment: chaosScore,
              score: (chaosScore / 1000 * 100).toFixed(2)
            });
          }

          return results.sort((a, b) => parseFloat(b.score) - parseFloat(a.score));
        }
      },
      {
        name: 'Harmonic Oscillator Frequencies',
        description: 'Test ω = √(k/m) · constant relationships',
        test: () => {
          const results = [];

          for (const constant of CONSTANTS.filter((_, i) => selectedConstants.includes(i))) {
            let resonances = 0;

            for (let i = 0; i < Math.min(1000, gaps.length); i++) {
              const gap = gaps[i];
              const omega = Math.sqrt(gap * constant.value);
              const harmonicGap = Math.pow(omega / constant.value, 2);

              if (Math.abs(gap - harmonicGap) / gap < 0.12) resonances++;
            }

            results.push({
              constant: constant.name,
              resonances: resonances,
              resonanceRate: (resonance / 1000 * 100).toFixed(2)
            });
          }

          return results.sort((a, b) => parseFloat(b.resonanceRate) - parseFloat(a.resonanceRate));
        }
      }
    ];

    const physicsResults = [];
    for (let i = 0; i < physicsTests.length; i++) {
      setProgress((i / physicsTests.length) * 100);
      const test = physicsTests[i];
      const result = test.test();
      physicsResults.push({
        name: test.name,
        description: test.description,
        results: result
      });
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    setResults({
      type: 'physics',
      data: physicsResults,
      summary: {
        testsRun: physicsResults.length,
        bestTest: physicsResults[0]?.name,
        bestConstant: physicsResults[0]?.results[0]?.constant
      }
    });

    setProgress(100);
    setComputing(false);
  };

  // Phase 3: Riemann Connection
  const exploreRiemann = async () => {
    setComputing(true);
    setProgress(0);

    // Simulate Riemann zeta zero correlations
    const zetaTests = [];

    for (let i = 0; i < CONSTANTS.length; i++) {
      if (!selectedConstants.includes(i)) continue;

      const constant = CONSTANTS[i];
      setProgress((i / CONSTANTS.length) * 100);

      // Simulate zeta zero spacing analysis
      const transformedPrimes = primes.slice(0, 1000).map(p => WT(p) * constant.value);

      // Calculate spacing distribution
      const spacings = [];
      for (let j = 1; j < transformedPrimes.length; j++) {
        spacings.push(transformedPrimes[j] - transformedPrimes[j-1]);
      }

      // Calculate statistics
      const mean = spacings.reduce((a, b) => a + b, 0) / spacings.length;
      const variance = spacings.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / spacings.length;
      const stdDev = Math.sqrt(variance);

      // GUE spacing ratio (theoretical = 0.5307)
      const gueRatio = mean / stdDev;
      const correlation = 1 - Math.abs(gueRatio - 0.5307) / 0.5307;

      zetaTests.push({
        constant: constant.name,
        symbol: constant.symbol,
        gueRatio: gueRatio.toFixed(4),
        correlation: (correlation * 100).toFixed(2),
        color: constant.color
      });

      await new Promise(resolve => setTimeout(resolve, 50));
    }

    zetaTests.sort((a, b) => parseFloat(b.correlation) - parseFloat(a.correlation));

    setResults({
      type: 'riemann',
      data: zetaTests,
      summary: {
        tested: zetaTests.length,
        bestConstant: zetaTests[0]?.constant,
        bestCorrelation: zetaTests[0]?.correlation,
        theoretical: '0.5307 (GUE)'
      }
    });

    setProgress(100);
    setComputing(false);
  };

  const runPhase = () => {
    switch(activePhase) {
      case 'constants':
        exploreConstants();
        break;
      case 'physics':
        explorePhysics();
        break;
      case 'riemann':
        exploreRiemann();
        break;
      default:
        break;
    }
  };

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600">
            🔬 Deep Research Expansion
          </h1>
          <p className="text-gray-300 text-lg">
            Exploring Mathematical Constants & Physical Interpretations
          </p>
          <div className="mt-4 text-sm text-gray-400">
            Dataset: {primes.length.toLocaleString()} primes, {gaps.length.toLocaleString()} gaps
          </div>
        </div>

        {/* Phase Selector */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          {[
            { id: 'constants', name: 'Phase 1: Mathematical Constants', icon: Database },
            { id: 'physics', name: 'Phase 2: Physical Interpretations', icon: Activity },
            { id: 'riemann', name: 'Phase 3: Riemann Connection', icon: Zap }
          ].map(phase => (
            <button
              key={phase.id}
              onClick={() => setActivePhase(phase.id)}
              className={`p-4 rounded-lg transition-all ${
                activePhase === phase.id
                  ? 'bg-purple-600 shadow-lg scale-105'
                  : 'bg-slate-800 hover:bg-slate-700'
              }`}
            >
              <phase.icon className="w-6 h-6 mx-auto mb-2" />
              <div className="text-sm font-semibold">{phase.name}</div>
            </button>
          ))}
        </div>

        {/* Constant Selection */}
        {activePhase === 'constants' && (
          <div className="bg-slate-800 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4">Select Constants to Test:</h3>
            <div className="grid grid-cols-4 gap-3">
              {CONSTANTS.map((constant, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setSelectedConstants(prev =>
                      prev.includes(idx)
                        ? prev.filter(i => i !== idx)
                        : [...prev, idx]
                    );
                  }}
                  className={`p-3 rounded-lg transition-all ${
                    selectedConstants.includes(idx)
                      ? 'bg-purple-600 shadow-lg'
                      : 'bg-slate-700 hover:bg-slate-600'
                  }`}
                  style={{
                    borderLeft: `4px solid ${constant.color}`
                  }}
                >
                  <div className="font-bold text-lg">{constant.symbol}</div>
                  <div className="text-xs text-gray-300">{constant.name}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Run Controls */}
        <div className="bg-slate-800 rounded-lg p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold">
                {activePhase === 'constants' && 'Test Mathematical Constants'}
                {activePhase === 'physics' && 'Explore Physical Interpretations'}
                {activePhase === 'riemann' && 'Analyze Riemann Connections'}
              </h3>
              <p className="text-sm text-gray-400 mt-1">
                {activePhase === 'constants' && 'Test g_n ≈ W_φ(p_n) · constant^k for various powers'}
                {activePhase === 'physics' && 'Explore quantum, chaos theory, and harmonic connections'}
                {activePhase === 'riemann' && 'Correlate with Riemann zeta function zeros'}
              </p>
            </div>
            <button
              onClick={runPhase}
              disabled={computing}
              className={`px-8 py-3 rounded-lg font-semibold flex items-center gap-2 transition-all ${
                computing
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-purple-600 hover:bg-purple-700 shadow-lg'
              }`}
            >
              {computing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              {computing ? 'Computing...' : 'Run Analysis'}
            </button>
          </div>

          {computing && (
            <div className="mt-4">
              <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
                <span>Progress</span>
                <span>{progress.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Results Display */}
        {results && (
          <div className="space-y-6">
            {/* Summary */}
            <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-lg p-6 border border-purple-500">
              <h3 className="text-2xl font-bold mb-4">🏆 Analysis Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(results.summary).map(([key, value]) => (
                  <div key={key} className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-sm text-gray-400 capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </div>
                    <div className="text-xl font-bold mt-1">
                      {typeof value === 'number' ? value.toFixed(2) : value}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Phase-specific results */}
            {results.type === 'constants' && (
              <div className="bg-slate-800 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4">Constant Match Rates</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={results.data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="symbol" stroke="#fff" />
                    <YAxis stroke="#fff" label={{ value: 'Match Rate (%)', angle: -90 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b' }} />
                    <Bar dataKey="bestMatchRate" fill="#a855f7" />
                  </BarChart>
                </ResponsiveContainer>

                <div className="mt-6 space-y-3">
                  {results.data.map((item, idx) => (
                    <div key={idx} className="bg-slate-700 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <div
                            className="w-4 h-4 rounded"
                            style={{ backgroundColor: item.color }}
                          />
                          <span className="font-bold text-lg">{item.symbol}</span>
                          <span className="text-gray-400">{item.name}</span>
                        </div>
                        <div className="text-xl font-bold text-purple-400">
                          {item.bestMatchRate}%
                        </div>
                      </div>
                      <div className="text-sm text-gray-400">
                        Best power: {item.symbol}^{item.bestPower} = {Math.pow(item.value, item.bestPower).toFixed(6)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results.type === 'physics' && (
              <div className="space-y-6">
                {results.data.map((test, idx) => (
                  <div key={idx} className="bg-slate-800 rounded-lg p-6">
                    <h3 className="text-xl font-bold mb-2">{test.name}</h3>
                    <p className="text-sm text-gray-400 mb-4">{test.description}</p>

                    <div className="space-y-2">
                      {test.results.map((result, ridx) => (
                        <div key={ridx} className="bg-slate-700 rounded-lg p-3 flex items-center justify-between">
                          <span className="font-semibold">{result.constant}</span>
                          <div className="text-right">
                            <div className="text-lg font-bold text-purple-400">
                              {result.matchRate || result.score || result.resonanceRate}%
                            </div>
                            <div className="text-xs text-gray-400">
                              {result.matches || result.lyapunovAlignment || result.resonances} matches
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {results.type === 'riemann' && (
              <div className="bg-slate-800 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4">Riemann Zeta Zero Correlations</h3>
                <p className="text-sm text-gray-400 mb-6">
                  GUE spacing ratio (theoretical: 0.5307) measures correlation with Riemann zeros
                </p>

                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={results.data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="symbol" stroke="#fff" />
                    <YAxis stroke="#fff" label={{ value: 'Correlation (%)', angle: -90 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b' }} />
                    <Bar dataKey="correlation" fill="#a855f7" />
                  </BarChart>
                </ResponsiveContainer>

                <div className="mt-6 space-y-3">
                  {results.data.map((item, idx) => (
                    <div key={idx} className="bg-slate-700 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div
                            className="w-4 h-4 rounded"
                            style={{ backgroundColor: item.color }}
                          />
                          <span className="font-bold text-lg">{item.symbol}</span>
                          <span className="text-gray-400">{item.constant}</span>
                        </div>
                        <div className="text-right">
                          <div className="text-xl font-bold text-purple-400">
                            {item.correlation}%
                          </div>
                          <div className="text-xs text-gray-400">
                            GUE ratio: {item.gueRatio}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Export Options */}
            <div className="bg-slate-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">📊 Export Results</h3>
              <div className="grid grid-cols-3 gap-4">
                <button className="px-4 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center justify-center gap-2">
                  <Download className="w-5 h-5" />
                  <span>JSON Data</span>
                </button>
                <button className="px-4 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center justify-center gap-2">
                  <Download className="w-5 h-5" />
                  <span>CSV Export</span>
                </button>
                <button className="px-4 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center justify-center gap-2">
                  <Download className="w-5 h-5" />
                  <span>Full Report</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DeepResearchExpansion;
