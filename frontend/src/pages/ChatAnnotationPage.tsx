import React, { useEffect, useState } from 'react';
import { startSession, replyToSession } from '../api/conversations';
import { Message, SessionSummary } from '../types/api';

const ChatAnnotationPage: React.FC = () => {
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);

  const loadSession = async () => {
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const data = await startSession();
      setSession(data.session);
      setMessages([data.first_message]);
      setFinished(false);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to start session');
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
      const response = await replyToSession(session.id, { text: input.trim() });
      setSession(response.session);
      setMessages((prev) => [...prev, response.user_message, ...(response.assistant_message ? [response.assistant_message] : [])]);
      setInput('');
      if (response.session_finished) {
        setFinished(true);
        if (response.new_session) {
          if (response.new_session.new_session_started && response.new_session.session && response.new_session.first_message) {
            setNotice('Previous conversation submitted. A new one is ready when you click "Load next".');
            setSession(response.new_session.session);
            setMessages([response.new_session.first_message]);
            setFinished(false);
          } else if (!response.new_session.new_session_started) {
            setNotice('Conversation submitted. No further topics available.');
          }
        }
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to send reply');
    } finally {
      setLoading(false);
    }
  };

  const progress = session
    ? Math.min(100, Math.round((session.user_token_count / session.user_token_target) * 100))
    : 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Standard Chat Annotation</h1>
          <p className="text-gray-600">AI speaks English, reply in Swahili or Sheng. Tokens tracked per conversation.</p>
          {session && (
            <p className="text-sm text-gray-500 mt-1">
              Topic: <span className="font-medium">{session.topic.title || session.topic.code}</span> — Target tokens: {session.user_token_target}
            </p>
          )}
        </div>
        <div className="w-64">
          <div className="text-sm text-gray-600 mb-1">
            Progress: {session?.user_token_count ?? 0}/{session?.user_token_target ?? 0} tokens
          </div>
          <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-indigo-600" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </div>

      {error && <div className="text-red-600">{error}</div>}
      {notice && <div className="text-indigo-700 bg-indigo-50 border border-indigo-200 p-3 rounded">{notice}</div>}

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
          <div className="flex justify-between items-center mt-2">
            <div className="text-xs text-gray-500">Aim for natural, detailed answers.</div>
            <button
              disabled={loading || !session}
              onClick={handleSend}
              className="bg-indigo-600 text-white px-4 py-2 rounded-md disabled:opacity-60"
            >
              {finished ? 'Submit & next' : 'Send reply'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatAnnotationPage;
