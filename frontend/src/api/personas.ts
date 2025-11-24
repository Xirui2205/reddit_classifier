import client from './client';
import { Persona } from '../types/api';

export async function listPersonas(): Promise<Persona[]> {
  const { data } = await client.get<Persona[]>('/api/personas');
  return data;
}

export async function createPersona(persona: Partial<Persona> & { code: string; prompt_text: string }): Promise<Persona> {
  const { data } = await client.post<Persona>('/api/personas', persona);
  return data;
}

export async function updatePersona(personaId: number, persona: Partial<Persona>): Promise<Persona> {
  const { data } = await client.put<Persona>(`/api/personas/${personaId}`, persona);
  return data;
}
