import client from './client';
import { AdminOverview, AnnotatorStats } from '../types/api';

export async function getMyStats(): Promise<AnnotatorStats> {
  const { data } = await client.get<AnnotatorStats>('/api/stats/me');
  return data;
}

export async function getAdminOverview(): Promise<AdminOverview> {
  const { data } = await client.get<AdminOverview>('/api/stats/admin/overview');
  return data;
}
