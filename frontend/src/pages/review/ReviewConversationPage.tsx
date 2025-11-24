import React from 'react';

const ReviewConversationPage: React.FC = () => {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Conversation review</h1>
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 rounded">
        Backend support missing for review actions. Conversation transcripts would appear here when available.
      </div>
    </div>
  );
};

export default ReviewConversationPage;
