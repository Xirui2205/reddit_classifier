import React, { useEffect, useState } from 'react';
import { fetchLeaderboard } from '../api/leaderboard';

const LeaderboardPage: React.FC = () => {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        await fetchLeaderboard();
      } catch (err: any) {
        setError(err?.message || 'Leaderboard not available');
      }
    };
    void load();
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Leaderboard</h1>
      {error && <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 rounded">{error}</div>}
      <p className="text-gray-600">Ranking data will appear here when backend support is available.</p>
    </div>
  );
};

export default LeaderboardPage;
