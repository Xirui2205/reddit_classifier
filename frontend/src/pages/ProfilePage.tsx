import React from 'react';
import { useAuth } from '../context/AuthContext';

const ProfilePage: React.FC = () => {
  const { user } = useAuth();

  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Profile</h1>
      <div className="bg-white border rounded-lg shadow-sm p-4 space-y-2">
        <div className="text-sm text-gray-500">Email</div>
        <div className="font-medium">{user?.email}</div>
        <div className="text-sm text-gray-500">Full name</div>
        <div className="font-medium">{user?.full_name || '—'}</div>
        <div className="text-sm text-gray-500">Role</div>
        <div className="font-medium">{user?.role}</div>
        <div className="text-sm text-gray-500">Status</div>
        <div className="font-medium">{user?.is_active ? 'Active' : 'Inactive'}</div>
      </div>
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-3 rounded">
        Account management actions (password change, preferences) are not exposed by the backend yet.
      </div>
    </div>
  );
};

export default ProfilePage;
