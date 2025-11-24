import client from './client';
import { Topic } from '../types/api';

export async function listTopics(): Promise<Topic[]> {
  const { data } = await client.get<Topic[]>('/api/topics');
  return data;
}

export async function createTopic(topic: Partial<Topic> & { code: string; target_tokens_per_session: number; total_token_target: number }): Promise<Topic> {
  const { data } = await client.post<Topic>('/api/topics', topic);
  return data;
}

export async function updateTopic(topicId: number, topic: Partial<Topic>): Promise<Topic> {
  const { data } = await client.put<Topic>(`/api/topics/${topicId}`, topic);
  return data;
}
