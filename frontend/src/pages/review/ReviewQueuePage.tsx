import React from 'react';

const ReviewQueuePage: React.FC = () => {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Review queue</h1>
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 rounded">
        Review workflow is not implemented in the backend. This page is read-only until endpoints are available.
      </div>
    </div>
  );
};

export default ReviewQueuePage;
