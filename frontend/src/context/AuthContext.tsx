import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import * as authApi from '../api/auth';
import { tokenStorage } from '../api/client';
import { User } from '../types/api';

interface AuthContextState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  role: string | null;
  initializing: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  initializeFromStorage: () => Promise<void>;
}

const AuthContext = createContext<AuthContextState | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [initializing, setInitializing] = useState(true);

  const logout = () => {
    setUser(null);
    setToken(null);
    tokenStorage.clear();
  };

  const initializeFromStorage = async () => {
    const stored = tokenStorage.get();
    if (stored) {
      setToken(stored);
      try {
        const profile = await authApi.getCurrentUser();
        setUser(profile);
      } catch (error) {
        logout();
      }
    }
    setInitializing(false);
  };

  const login = async (email: string, password: string) => {
    const tokenResp = await authApi.login(email, password);
    setToken(tokenResp.access_token);
    tokenStorage.set(tokenResp.access_token);
    const profile = await authApi.getCurrentUser();
    setUser(profile);
  };

  useEffect(() => {
    void initializeFromStorage();
  }, []);

  const value = useMemo(
    () => ({
      user,
      token,
      isAuthenticated: Boolean(user && token),
      role: user?.role || null,
      initializing,
      login,
      logout,
      initializeFromStorage,
    }),
    [user, token, initializing]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextState => {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return ctx;
};
