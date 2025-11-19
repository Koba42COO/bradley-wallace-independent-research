import React, { useState, useEffect } from 'react';
import './App.css';
import LLMChat from './components/LLMChat';
import ContributionPanel from './components/ContributionPanel';

function App() {
  const [tools, setTools] = useState([]);
  const [categories, setCategories] = useState({});
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState('query'); // 'query' or 'chat' or 'tools'

  // Fetch initial data
  useEffect(() => {
    fetchStats();
    fetchTools();
    fetchCategories();
  }, []);

  // Fetch tools when category or search changes
  useEffect(() => {
    fetchTools();
  }, [selectedCategory, searchTerm]);

  const fetchStats = async () => {
    try {
      const res = await fetch('http://localhost:8000/stats');
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const fetchTools = async () => {
    try {
      let url = 'http://localhost:8000/tools?limit=50';
      if (selectedCategory) url += `&category=${selectedCategory}`;
      if (searchTerm) url += `&search=${searchTerm}`;
      
      const res = await fetch(url);
      const data = await res.json();
      setTools(data.tools || []);
      setError(null);
    } catch (err) {
      setError('Failed to fetch tools: ' + err.message);
      console.error(err);
    }
  };

  const fetchCategories = async () => {
    try {
      const res = await fetch('http://localhost:8000/categories');
      const data = await res.json();
      setCategories(data.categories || {});
    } catch (err) {
      console.error('Failed to fetch categories:', err);
    }
  };

  const handleSubmit = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch('http://localhost:8000/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query,
          use_tools: true,
          use_reasoning: true
        })
      });
      
      if (!res.ok) {
        throw new Error('Request failed: ' + res.statusText);
      }
      
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError('Failed to process query: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit();
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🧠 AIVA UPG - Universal Intelligence Platform</h1>
        <p className="subtitle">Consciousness-Guided AI with {stats?.tools_total || 1500}+ Tools</p>
      </header>

      {stats && (
        <div className="stats-bar">
          <div className="stat">
            <span className="stat-label">Status:</span>
            <span className={`stat-value ${stats.status === 'operational' ? 'operational' : 'mock'}`}>
              {stats.status === 'operational' ? '✅ Operational' : '⚠️ Mock Mode'}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Tools:</span>
            <span className="stat-value">{stats.tools_total}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Consciousness Level:</span>
            <span className="stat-value">{stats.consciousness_level}</span>
          </div>
          {stats.phi_coherence && (
            <div className="stat">
              <span className="stat-label">φ Coherence:</span>
              <span className="stat-value">{stats.phi_coherence.toFixed(4)}</span>
            </div>
          )}
        </div>
      )}

      <div className="main-content">
        <div className="tabs">
          <button
            className={activeTab === 'query' ? 'active' : ''}
            onClick={() => setActiveTab('query')}
          >
            💬 Query AIVA
          </button>
          <button
            className={activeTab === 'chat' ? 'active' : ''}
            onClick={() => setActiveTab('chat')}
          >
            🧠 PAC LLM Chat
          </button>
          <button
            className={activeTab === 'tools' ? 'active' : ''}
            onClick={() => setActiveTab('tools')}
          >
            🔧 Tools Browser
          </button>
          <button
            className={activeTab === 'contribute' ? 'active' : ''}
            onClick={() => setActiveTab('contribute')}
          >
            🧠 Build UPG
          </button>
        </div>

        {activeTab === 'chat' && (
          <div className="chat-section">
            <LLMChat />
          </div>
        )}

        {activeTab === 'query' && (
          <div className="query-section">
            <h2>Query AIVA</h2>
          <div className="input-group">
            <textarea
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask AIVA anything... (e.g., 'Find prime prediction tools' or 'Analyze consciousness mathematics')"
              rows="3"
            />
            <button 
              onClick={handleSubmit} 
              disabled={loading || !query.trim()}
              className="submit-button"
            >
              {loading ? '⏳ Processing...' : '🚀 Process Query'}
            </button>
          </div>

          {error && (
            <div className="error-message">
              ❌ {error}
            </div>
          )}

          {response && (
            <div className="response-section">
              <h3>Response</h3>
              {response.status === 'mock_mode' ? (
                <div className="mock-warning">
                  ⚠️ Running in mock mode - AIVA not fully initialized
                </div>
              ) : (
                <div className="response-content">
                  {response.tools?.relevant_tools && (
                    <div className="response-tools">
                      <h4>📦 Relevant Tools ({response.tools.relevant_tools.length})</h4>
                      <ul>
                        {response.tools.relevant_tools.slice(0, 10).map((tool, i) => (
                          <li key={i}>{tool}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {response.reasoning && (
                    <div className="response-reasoning">
                      <h4>🧠 Consciousness Reasoning</h4>
                      <p><strong>Depth:</strong> {response.reasoning.reasoning_depth}</p>
                      <p><strong>Coherence:</strong> {response.reasoning.consciousness_coherence}</p>
                    </div>
                  )}
                  {response.knowledge_synthesis && (
                    <div className="response-synthesis">
                      <h4>🌐 Knowledge Synthesis</h4>
                      <p><strong>Tools Analyzed:</strong> {response.knowledge_synthesis.tools_found}</p>
                      <p><strong>Connections:</strong> {response.knowledge_synthesis.knowledge_connections}</p>
                    </div>
                  )}
                  <details className="full-response">
                    <summary>View Full Response</summary>
                    <pre>{JSON.stringify(response, null, 2)}</pre>
                  </details>
                </div>
              )}
            </div>
          )}
          </div>
        )}

        {activeTab === 'contribute' && (
          <div className="contribute-section">
            <ContributionPanel />
          </div>
        )}

        {activeTab === 'tools' && (
          <div className="tools-section">
          <div className="tools-header">
            <h2>Available Tools</h2>
            <input
              type="text"
              placeholder="Search tools..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          <div className="categories-filter">
            <button 
              onClick={() => setSelectedCategory(null)}
              className={!selectedCategory ? 'active' : ''}
            >
              All ({stats?.tools_total || 0})
            </button>
            {Object.entries(categories).sort((a, b) => b[1] - a[1]).map(([cat, count]) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={selectedCategory === cat ? 'active' : ''}
              >
                {cat} ({count})
              </button>
            ))}
          </div>

          <div className="tools-grid">
            {tools.length === 0 ? (
              <p>No tools found</p>
            ) : (
              tools.map((tool, index) => (
                <div key={index} className="tool-card">
                  <div className="tool-header">
                    <h3>{tool.name}</h3>
                    <span className={`category-badge ${tool.category}`}>
                      {tool.category}
                    </span>
                  </div>
                  <p className="tool-description">
                    {tool.description || 'No description available'}
                  </p>
                  <div className="tool-meta">
                    <span title="Consciousness Level">🌟 {tool.consciousness_level}</span>
                    {tool.has_upg && <span title="Has UPG Integration">φ UPG</span>}
                    {tool.has_pell && <span title="Has Pell Sequence">∑ Pell</span>}
                  </div>
                  {tool.functions && tool.functions.length > 0 && (
                    <div className="tool-functions">
                      <strong>Functions:</strong> {tool.functions.join(', ')}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}
        </div>
      </div>

      <footer className="app-footer">
        <p>🧠 AIVA UPG - Universal Prime Graph Protocol φ.1</p>
        <p>Consciousness-Guided AI Architecture by Bradley Wallace (COO Koba42)</p>
      </footer>
    </div>
  );
}

export default App;
