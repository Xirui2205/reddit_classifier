import React from 'react';
import { Link } from 'react-router-dom';

const NotFoundPage: React.FC = () => (
  <div className="text-center py-20">
    <h1 className="text-3xl font-semibold mb-2">Page not found</h1>
    <p className="text-gray-600 mb-4">The page you are looking for does not exist.</p>
    <Link to="/dashboard" className="text-indigo-600">Return to dashboard</Link>
  </div>
);

export default NotFoundPage;
