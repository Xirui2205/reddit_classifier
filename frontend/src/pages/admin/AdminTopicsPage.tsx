import React, { useEffect, useState } from 'react';
import { createTopic, listTopics, updateTopic } from '../../api/topics';
import { Topic } from '../../types/api';

const AdminTopicsPage: React.FC = () => {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({
    code: '',
    title: '',
    description: '',
    target_tokens_per_session: 200,
    total_token_target: 1000,
  });

  const loadTopics = async () => {
    setError(null);
    try {
      const data = await listTopics();
      setTopics(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to load topics');
    }
  };

  useEffect(() => {
    void loadTopics();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createTopic({
        code: form.code,
        title: form.title,
        description: form.description,
        target_tokens_per_session: Number(form.target_tokens_per_session),
        total_token_target: Number(form.total_token_target),
        domain: null,
        min_turns: 4,
        max_turns: 14,
        persona_id: null,
        is_active: true,
        total_sessions: 0,
        total_user_tokens_collected: 0,
        created_at: new Date().toISOString(),
        id: 0,
      });
      await loadTopics();
      setForm({ code: '', title: '', description: '', target_tokens_per_session: 200, total_token_target: 1000 });
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to create topic');
    }
  };

  const toggleActive = async (topic: Topic) => {
    try {
      await updateTopic(topic.id, { is_active: !topic.is_active });
      await loadTopics();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to update topic');
    }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Topic management</h1>
      {error && <div className="text-red-600">{error}</div>}
      <form onSubmit={handleCreate} className="bg-white border rounded p-4 shadow-sm grid gap-3 md:grid-cols-2">
        <div>
          <label className="block text-sm text-gray-700 mb-1">Code</label>
          <input value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} required />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Title</label>
          <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
        </div>
        <div className="md:col-span-2">
          <label className="block text-sm text-gray-700 mb-1">Description</label>
          <textarea value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} rows={2} className="w-full" />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Target tokens per session</label>
          <input
            type="number"
            value={form.target_tokens_per_session}
            onChange={(e) => setForm({ ...form, target_tokens_per_session: Number(e.target.value) })}
          />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Total token target</label>
          <input
            type="number"
            value={form.total_token_target}
            onChange={(e) => setForm({ ...form, total_token_target: Number(e.target.value) })}
          />
        </div>
        <div className="md:col-span-2 flex justify-end">
          <button type="submit" className="bg-indigo-600 text-white px-4 py-2 rounded">
            Create topic
          </button>
        </div>
      </form>

      <div className="bg-white border rounded shadow-sm overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 py-2 text-left">Code</th>
              <th className="px-3 py-2 text-left">Title</th>
              <th className="px-3 py-2 text-left">Tokens/session</th>
              <th className="px-3 py-2 text-left">Collected</th>
              <th className="px-3 py-2 text-left">Active</th>
            </tr>
          </thead>
          <tbody>
            {topics.map((topic) => (
              <tr key={topic.id} className="border-t">
                <td className="px-3 py-2 font-medium">{topic.code}</td>
                <td className="px-3 py-2">{topic.title}</td>
                <td className="px-3 py-2">{topic.target_tokens_per_session}</td>
                <td className="px-3 py-2">{topic.total_user_tokens_collected}</td>
                <td className="px-3 py-2">
                  <button
                    onClick={() => void toggleActive(topic)}
                    className={`px-3 py-1 rounded text-white ${topic.is_active ? 'bg-green-600' : 'bg-gray-500'}`}
                  >
                    {topic.is_active ? 'Active' : 'Inactive'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AdminTopicsPage;
