import React, { useEffect, useState } from 'react';
import { startSession, replyToSession } from '../api/conversations';
import { Message, SessionSummary } from '../types/api';

const PersonaChatPage: React.FC = () => {
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSession = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await startSession();
      setSession(data.session);
      setMessages([data.first_message]);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to start persona conversation');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadSession();
  }, []);

  const handleSend = async () => {
    if (!session || !input.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await replyToSession(session.id, { text: input.trim() });
      setSession(res.session);
      setMessages((prev) => [...prev, res.user_message, ...(res.assistant_message ? [res.assistant_message] : [])]);
      setInput('');
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to send reply');
    } finally {
      setLoading(false);
    }
  };

  const persona = session?.topic.persona;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Persona Conversation</h1>
          <p className="text-gray-600">Converse with AI personas while replying in Swahili/Sheng.</p>
          {persona ? (
            <div className="mt-2 text-sm text-gray-700">
              <div className="font-semibold">Persona: {persona.name || persona.code}</div>
              <div>{persona.description}</div>
            </div>
          ) : (
            <p className="text-sm text-gray-500">No persona configured for this topic.</p>
          )}
        </div>
        <button onClick={loadSession} className="bg-indigo-600 text-white px-4 py-2 rounded-md" disabled={loading}>
          Load new session
        </button>
      </div>

      {error && <div className="text-red-600">{error}</div>}

      <div className="bg-white border rounded-lg shadow-sm p-4 h-[70vh] flex flex-col">
        <div className="flex-1 overflow-y-auto space-y-3 pr-2">
          {messages.map((msg) => (
            <div key={msg.id} className={`max-w-3xl ${msg.sender_role === 'assistant' ? 'text-left' : 'text-right ml-auto'}`}>
              <div
                className={`inline-block px-4 py-2 rounded-lg shadow ${
                  msg.sender_role === 'assistant' ? 'bg-gray-100 text-gray-900' : 'bg-indigo-600 text-white'
                }`}
              >
                <div className="text-xs uppercase tracking-wide opacity-70 mb-1">{msg.sender_role}</div>
                <p className="whitespace-pre-wrap text-sm">{msg.text}</p>
              </div>
            </div>
          ))}
          {!messages.length && <div className="text-gray-500">No messages yet.</div>}
        </div>
        <div className="mt-4">
          <label className="block text-sm text-gray-600 mb-1">Your reply (Swahili/Sheng)</label>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={3}
            className="w-full"
            placeholder="Jibu kwa Kiswahili au Sheng..."
          />
          <div className="flex justify-end mt-2">
            <button
              disabled={loading || !session}
              onClick={handleSend}
              className="bg-indigo-600 text-white px-4 py-2 rounded-md disabled:opacity-60"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PersonaChatPage;
