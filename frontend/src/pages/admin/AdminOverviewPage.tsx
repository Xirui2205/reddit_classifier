import React, { useEffect, useState } from 'react';
import { getAdminOverview } from '../../api/stats';
import { AdminOverview } from '../../types/api';

const AdminOverviewPage: React.FC = () => {
  const [data, setData] = useState<AdminOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await getAdminOverview();
        setData(res);
      } catch (err: any) {
        setError(err?.response?.data?.detail || 'Unable to load admin overview');
      }
    };
    void load();
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Admin Overview</h1>
      {error && <div className="text-red-600">{error}</div>}
      {data && (
        <div className="grid gap-4 md:grid-cols-3">
          <div className="bg-white border p-4 rounded shadow-sm">
            <div className="text-sm text-gray-500">Global tokens</div>
            <div className="text-2xl font-semibold">{data.global_tokens}</div>
          </div>
          <div className="bg-white border p-4 rounded shadow-sm">
            <div className="text-sm text-gray-500">Topics</div>
            <div className="text-2xl font-semibold">{Object.keys(data.tokens_per_topic).length}</div>
          </div>
          <div className="bg-white border p-4 rounded shadow-sm">
            <div className="text-sm text-gray-500">Annotators ranked</div>
            <div className="text-2xl font-semibold">{data.annotator_rankings.length}</div>
          </div>
        </div>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="bg-white border rounded p-4 shadow-sm">
          <h2 className="text-lg font-semibold mb-2">Tokens per topic</h2>
          <ul className="space-y-1 text-sm text-gray-700">
            {data &&
              Object.entries(data.tokens_per_topic).map(([topic, tokens]) => (
                <li key={topic} className="flex justify-between">
                  <span>{topic}</span>
                  <span className="font-medium">{tokens}</span>
                </li>
              ))}
          </ul>
        </div>
        <div className="bg-white border rounded p-4 shadow-sm">
          <h2 className="text-lg font-semibold mb-2">Sessions per topic</h2>
          <ul className="space-y-1 text-sm text-gray-700">
            {data &&
              Object.entries(data.sessions_per_topic).map(([topic, count]) => (
                <li key={topic} className="flex justify-between">
                  <span>{topic}</span>
                  <span className="font-medium">{count}</span>
                </li>
              ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AdminOverviewPage;
