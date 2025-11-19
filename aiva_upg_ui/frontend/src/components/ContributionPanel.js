import React, { useState, useEffect } from 'react';
import './ContributionPanel.css';

function ContributionPanel() {
  const [question, setQuestion] = useState('');
  const [userId, setUserId] = useState('user_' + Math.random().toString(36).substr(2, 9));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [userStats, setUserStats] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);
  const [upgStats, setUpgStats] = useState(null);

  useEffect(() => {
    fetchUserStats();
    fetchLeaderboard();
    fetchUPGStats();
  }, []);

  const fetchUserStats = async () => {
    try {
      const res = await fetch(`http://localhost:8000/contribute/user/${userId}/stats`);
      const data = await res.json();
      setUserStats(data);
    } catch (err) {
      console.error('Failed to fetch user stats:', err);
    }
  };

  const fetchLeaderboard = async () => {
    try {
      const res = await fetch('http://localhost:8000/contribute/leaderboard?limit=10');
      const data = await res.json();
      setLeaderboard(data.leaderboard || []);
    } catch (err) {
      console.error('Failed to fetch leaderboard:', err);
    }
  };

  const fetchUPGStats = async () => {
    try {
      const res = await fetch('http://localhost:8000/contribute/upg/stats');
      const data = await res.json();
      setUpgStats(data.upg_statistics);
    } catch (err) {
      console.error('Failed to fetch UPG stats:', err);
    }
  };

  const handleContribute = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:8000/contribute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          question: question,
          reward_pool: 100.0
        })
      });

      if (!res.ok) {
        throw new Error('Request failed: ' + res.statusText);
      }

      const data = await res.json();
      setResult(data);
      setQuestion('');

      // Refresh stats
      fetchUserStats();
      fetchLeaderboard();
      fetchUPGStats();
    } catch (err) {
      setError('Failed to contribute: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="contribution-panel">
      <div className="contribution-header">
        <h3>🧠 Build UPG Through Questions</h3>
        <p className="subtitle">Your questions create the knowledge graph and earn rewards!</p>
      </div>

      {userStats && (
        <div className="user-stats-bar">
          <div className="stat-item">
            <span className="stat-label">Your Rewards:</span>
            <span className="stat-value">{userStats.total_rewards?.toFixed(2) || 0}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Contributions:</span>
            <span className="stat-value">{userStats.contribution_count || 0}</span>
          </div>
        </div>
      )}

      <div className="contribution-input-section">
        <h4>Ask a Question</h4>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about consciousness mathematics, PAC compression, prime topology, or anything related to AIVA/UPG..."
          rows="4"
          disabled={loading}
        />
        <button
          onClick={handleContribute}
          disabled={loading || !question.trim()}
          className="contribute-button"
        >
          {loading ? '⏳ Contributing...' : '🚀 Contribute & Build UPG'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {result && (
        <div className="contribution-result">
          <h4>✅ Contribution Processed!</h4>
          
          <div className="scores-grid">
            <div className="score-card">
              <div className="score-label">Quality</div>
              <div className="score-value">{result.contribution?.quality_score?.toFixed(1) || 0}</div>
            </div>
            <div className="score-card">
              <div className="score-label">Metrics</div>
              <div className="score-value">{result.contribution?.metric_score?.toFixed(1) || 0}</div>
            </div>
            <div className="score-card">
              <div className="score-label">Demand</div>
              <div className="score-value">{result.contribution?.demand_score?.toFixed(1) || 0}</div>
            </div>
            <div className="score-card">
              <div className="score-label">Novelty</div>
              <div className="score-value">{result.contribution?.novelty_score?.toFixed(1) || 0}</div>
            </div>
            <div className="score-card total">
              <div className="score-label">Total Score</div>
              <div className="score-value">{result.contribution?.total_score?.toFixed(1) || 0}</div>
            </div>
          </div>

          <div className="upg-impact">
            <h5>🧠 UPG Impact</h5>
            <div className="impact-stats">
              <div>
                <strong>Nodes Created:</strong> {result.upg_impact?.nodes_created || 0}
              </div>
              <div>
                <strong>Connections:</strong> {result.upg_impact?.connections_created || 0}
              </div>
              <div>
                <strong>Total UPG Nodes:</strong> {result.upg_impact?.total_nodes || 0}
              </div>
            </div>
          </div>

          <div className="reward-display">
            <h5>💰 Reward Earned</h5>
            <div className="reward-amount">
              {result.reward?.amount?.toFixed(2) || 0}
            </div>
            <div className="reward-total">
              Total Rewards: {result.reward?.total_user_rewards?.toFixed(2) || 0}
            </div>
          </div>

          <p className="success-message">{result.message}</p>
        </div>
      )}

      <div className="stats-sections">
        <div className="upg-stats-section">
          <h4>📊 UPG Statistics</h4>
          {upgStats ? (
            <div className="stats-grid">
              <div>
                <strong>Total Nodes:</strong> {upgStats.total_nodes || 0}
              </div>
              <div>
                <strong>Total Connections:</strong> {upgStats.total_connections || 0}
              </div>
              <div>
                <strong>Avg Connections:</strong> {upgStats.avg_connections_per_node?.toFixed(2) || 0}
              </div>
              <div>
                <strong>Prime Coordinates:</strong> {upgStats.prime_coordinates_used || 0}
              </div>
            </div>
          ) : (
            <p>Loading UPG statistics...</p>
          )}
        </div>

        <div className="leaderboard-section">
          <h4>🏆 Top Contributors</h4>
          {leaderboard.length > 0 ? (
            <div className="leaderboard-list">
              {leaderboard.map((user, idx) => (
                <div key={idx} className="leaderboard-item">
                  <span className="rank">#{user.rank}</span>
                  <span className="user-id">{user.user_id}</span>
                  <span className="rewards">{user.total_rewards?.toFixed(2) || 0}</span>
                  <span className="contributions">({user.contribution_count} contributions)</span>
                </div>
              ))}
            </div>
          ) : (
            <p>No contributors yet. Be the first!</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default ContributionPanel;

