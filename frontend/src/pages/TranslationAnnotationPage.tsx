import React from 'react';

const TranslationAnnotationPage: React.FC = () => {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Translation Annotation</h1>
      <p className="text-gray-600">Provide Swahili and Sheng translations for English prompts.</p>
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 rounded">
        Backend support missing for translation tasks. This mode is a placeholder until the API is available.
      </div>
      <div className="bg-white border rounded-lg shadow-sm p-6 space-y-4">
        <div>
          <div className="text-sm text-gray-500">English source</div>
          <div className="mt-2 p-3 bg-gray-50 border rounded">No task loaded.</div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="block text-sm text-gray-700 mb-1">Swahili translation</label>
            <textarea className="w-full" rows={4} disabled placeholder="Waiting for backend support" />
          </div>
          <div>
            <label className="block text-sm text-gray-700 mb-1">Sheng translation</label>
            <textarea className="w-full" rows={4} disabled placeholder="Waiting for backend support" />
          </div>
        </div>
        <button className="bg-indigo-600 text-white px-4 py-2 rounded-md opacity-60 cursor-not-allowed">Submit</button>
      </div>
    </div>
  );
};

export default TranslationAnnotationPage;
