import React, { useEffect, useState } from 'react';
import { createPersona, listPersonas, updatePersona } from '../../api/personas';
import { Persona } from '../../types/api';

const AdminPersonasPage: React.FC = () => {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ code: '', name: '', description: '', prompt_text: '' });

  const load = async () => {
    setError(null);
    try {
      const data = await listPersonas();
      setPersonas(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to load personas');
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createPersona({ ...form, is_active: true, id: 0, created_at: new Date().toISOString() });
      await load();
      setForm({ code: '', name: '', description: '', prompt_text: '' });
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to create persona');
    }
  };

  const toggleActive = async (persona: Persona) => {
    try {
      await updatePersona(persona.id, { is_active: !persona.is_active });
      await load();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to update persona');
    }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Persona management</h1>
      {error && <div className="text-red-600">{error}</div>}
      <form onSubmit={handleCreate} className="bg-white border rounded p-4 shadow-sm space-y-3">
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <label className="block text-sm text-gray-700 mb-1">Code</label>
            <input value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} required />
          </div>
          <div>
            <label className="block text-sm text-gray-700 mb-1">Name</label>
            <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </div>
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Description</label>
          <textarea value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} rows={2} className="w-full" />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">System prompt</label>
          <textarea value={form.prompt_text} onChange={(e) => setForm({ ...form, prompt_text: e.target.value })} rows={3} className="w-full" required />
        </div>
        <div className="flex justify-end">
          <button type="submit" className="bg-indigo-600 text-white px-4 py-2 rounded">
            Create persona
          </button>
        </div>
      </form>

      <div className="bg-white border rounded shadow-sm overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 py-2 text-left">Code</th>
              <th className="px-3 py-2 text-left">Name</th>
              <th className="px-3 py-2 text-left">Active</th>
            </tr>
          </thead>
          <tbody>
            {personas.map((persona) => (
              <tr key={persona.id} className="border-t">
                <td className="px-3 py-2 font-medium">{persona.code}</td>
                <td className="px-3 py-2">{persona.name}</td>
                <td className="px-3 py-2">
                  <button
                    onClick={() => void toggleActive(persona)}
                    className={`px-3 py-1 rounded text-white ${persona.is_active ? 'bg-green-600' : 'bg-gray-500'}`}
                  >
                    {persona.is_active ? 'Active' : 'Inactive'}
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

export default AdminPersonasPage;
