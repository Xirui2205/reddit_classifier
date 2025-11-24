import client from './client';
import {
  SessionDetail,
  SessionReplyPayload,
  SessionReplyResponse,
  SessionStartResponse,
} from '../types/api';

export async function startSession(): Promise<SessionStartResponse> {
  const { data } = await client.post<SessionStartResponse>('/api/sessions/start');
  return data;
}

export async function replyToSession(
  sessionId: number,
  payload: SessionReplyPayload
): Promise<SessionReplyResponse> {
  const { data } = await client.post<SessionReplyResponse>(`/api/sessions/${sessionId}/reply`, payload);
  return data;
}

export async function getSession(sessionId: number): Promise<SessionDetail> {
  const { data } = await client.get<SessionDetail>(`/api/sessions/${sessionId}`);
  return data;
}
