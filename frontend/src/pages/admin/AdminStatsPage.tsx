import React, { useEffect, useState } from 'react';
import { getAdminOverview } from '../../api/stats';
import { AdminOverview } from '../../types/api';

const AdminStatsPage: React.FC = () => {
  const [data, setData] = useState<AdminOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await getAdminOverview();
        setData(res);
      } catch (err: any) {
        setError(err?.response?.data?.detail || 'Unable to load stats');
      }
    };
    void load();
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Admin analytics</h1>
      {error && <div className="text-red-600">{error}</div>}
      {data ? (
        <div className="space-y-3">
          <div className="bg-white border rounded p-4 shadow-sm">
            <h2 className="text-lg font-semibold mb-2">Tokens per day</h2>
            <ul className="space-y-1 text-sm text-gray-700">
              {Object.entries(data.tokens_per_day).map(([day, tokens]) => (
                <li key={day} className="flex justify-between">
                  <span>{day}</span>
                  <span className="font-medium">{tokens}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="bg-white border rounded p-4 shadow-sm">
            <h2 className="text-lg font-semibold mb-2">Annotator rankings</h2>
            <ul className="space-y-1 text-sm text-gray-700">
              {data.annotator_rankings.map((row, idx) => (
                <li key={row.annotator_id} className="flex justify-between">
                  <span>
                    #{idx + 1} – {row.annotator_id}
                  </span>
                  <span className="font-medium">{row.total_tokens} tokens</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : (
        <p className="text-gray-600">Loading admin analytics...</p>
      )}
    </div>
  );
};

export default AdminStatsPage;
