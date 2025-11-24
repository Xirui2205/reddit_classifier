import React from 'react';
import { Link } from 'react-router-dom';

const modes = [
  {
    title: 'Standard Chat Annotation',
    description: 'AI in English, respond in Swahili/Sheng. Tokens tracked per topic.',
    href: '/annotate/chat',
  },
  {
    title: 'Translation Annotation',
    description: 'Provide Swahili and Sheng translations for English text.',
    href: '/annotate/translation',
  },
  {
    title: 'Persona Conversations',
    description: 'Chat with assigned AI personas with multi-turn prompts.',
    href: '/annotate/persona',
  },
];

const AnnotationHubPage: React.FC = () => {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Choose an annotation mode</h1>
      <div className="grid gap-4 md:grid-cols-3">
        {modes.map((mode) => (
          <div key={mode.title} className="bg-white border rounded-lg p-4 shadow-sm flex flex-col">
            <h2 className="text-lg font-semibold mb-1">{mode.title}</h2>
            <p className="text-sm text-gray-600 flex-1">{mode.description}</p>
            <Link to={mode.href} className="mt-4 inline-flex text-indigo-600 font-medium">
              Open
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AnnotationHubPage;
