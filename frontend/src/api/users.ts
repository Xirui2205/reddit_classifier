import { register } from './auth';

export const createUser = register;

export async function listUsers() {
  throw new Error('User listing endpoint not implemented by backend');
}
