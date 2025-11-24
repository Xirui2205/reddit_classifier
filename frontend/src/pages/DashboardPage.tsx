import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { getMyStats } from '../api/stats';
import { AnnotatorStats } from '../types/api';

const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState<AnnotatorStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getMyStats();
        setStats(data);
      } catch (err: any) {
        setError(err?.response?.data?.detail || 'Unable to load stats');
      }
    };
    void load();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Welcome, {user?.full_name || user?.email}</h1>
        <p className="text-gray-600">Role: {user?.role}</p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h2 className="text-sm text-gray-500">Total tokens</h2>
          <div className="text-2xl font-semibold">{stats ? stats.total_tokens : '—'}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h2 className="text-sm text-gray-500">Topics covered</h2>
          <div className="text-2xl font-semibold">{stats ? Object.keys(stats.tokens_per_topic).length : '—'}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h2 className="text-sm text-gray-500">Daily activity</h2>
          <div className="text-2xl font-semibold">{stats ? Object.keys(stats.tokens_per_day).length : '—'}</div>
        </div>
      </div>

      {error && <div className="text-red-600">{error}</div>}

      <div className="grid gap-4 md:grid-cols-2">
        <div className="bg-white p-5 rounded-lg border shadow-sm">
          <h3 className="text-lg font-semibold mb-2">Start annotating</h3>
          <p className="text-sm text-gray-600 mb-4">Choose a mode to begin contributing.</p>
          <div className="flex flex-wrap gap-2">
            <Link to="/annotate/chat" className="bg-indigo-600 text-white px-4 py-2 rounded-md">
              Chat annotation
            </Link>
            <Link to="/annotate/translation" className="bg-indigo-100 text-indigo-700 px-4 py-2 rounded-md">
              Translation mode
            </Link>
            <Link to="/annotate/persona" className="bg-indigo-100 text-indigo-700 px-4 py-2 rounded-md">
              Persona chats
            </Link>
          </div>
        </div>
        <div className="bg-white p-5 rounded-lg border shadow-sm">
          <h3 className="text-lg font-semibold mb-2">Community</h3>
          <p className="text-sm text-gray-600 mb-4">Track progress against peers.</p>
          <Link to="/leaderboard" className="text-indigo-600 font-medium">
            View leaderboard
          </Link>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
