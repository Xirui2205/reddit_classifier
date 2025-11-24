import React, { useState } from 'react';
import { createUser, listUsers } from '../../api/users';

const AdminUsersPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('annotator');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);
    setError(null);
    try {
      await createUser({ email, password, role });
      setMessage('User created (requires admin privileges).');
      setEmail('');
      setPassword('');
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to create user');
    }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">User management</h1>
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 rounded">
        Listing users is not supported by the backend yet. You can create users if authenticated as admin.
      </div>
      <form onSubmit={handleCreate} className="bg-white border rounded p-4 shadow-sm space-y-3 max-w-lg">
        <div>
          <label className="block text-sm text-gray-700 mb-1">Email</label>
          <input className="w-full" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Password</label>
          <input className="w-full" type="password" required value={password} onChange={(e) => setPassword(e.target.value)} />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Role</label>
          <select className="w-full" value={role} onChange={(e) => setRole(e.target.value)}>
            <option value="annotator">Annotator</option>
            <option value="admin">Admin</option>
          </select>
        </div>
        {message && <div className="text-green-700">{message}</div>}
        {error && <div className="text-red-600">{error}</div>}
        <button type="submit" className="bg-indigo-600 text-white px-4 py-2 rounded">Create user</button>
      </form>
    </div>
  );
};

export default AdminUsersPage;
