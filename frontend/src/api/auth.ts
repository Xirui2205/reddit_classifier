import client from './client';
import { TokenResponse, User } from '../types/api';

export async function login(email: string, password: string): Promise<TokenResponse> {
  const { data } = await client.post<TokenResponse>('/api/auth/login', { email, password });
  return data;
}

export async function getCurrentUser(): Promise<User> {
  const { data } = await client.get<User>('/api/auth/me');
  return data;
}

export async function register(user: { email: string; password: string; full_name?: string; role?: string }): Promise<User> {
  const { data } = await client.post<User>('/api/auth/register', user);
  return data;
}
