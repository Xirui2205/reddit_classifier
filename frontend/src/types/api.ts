export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  id: number;
  email: string;
  full_name?: string | null;
  role?: string | null;
  is_active?: boolean | null;
  created_at: string;
}

export interface Topic {
  id: number;
  code: string;
  title?: string | null;
  description?: string | null;
  domain?: string | null;
  target_tokens_per_session: number;
  total_token_target: number;
  total_sessions: number;
  total_user_tokens_collected: number;
  min_turns?: number | null;
  max_turns?: number | null;
  persona_id?: number | null;
  is_active?: boolean | null;
  created_at: string;
  persona?: Persona | null;
}

export interface Persona {
  id: number;
  code: string;
  name?: string | null;
  description?: string | null;
  prompt_text: string;
  is_active?: boolean | null;
  created_at: string;
}

export interface Message {
  id: number;
  sender_role: string;
  lang: string;
  text: string;
  token_count: number;
  created_at: string;
}

export interface SessionSummary {
  id: number;
  status: string;
  user_token_target: number;
  user_token_count: number;
  turn_count: number;
  started_at: string;
  completed_at?: string | null;
  topic: Topic;
}

export interface SessionDetail extends SessionSummary {
  messages: Message[];
}

export interface SessionStartResponse {
  session: SessionSummary;
  first_message: Message;
}

export interface SessionReplyPayload {
  text: string;
}

export interface SessionReplyResponse {
  session: SessionSummary;
  user_message: Message;
  assistant_message?: Message | null;
  session_finished: boolean;
  new_session?:
    | {
        session_finished: boolean;
        new_session_started: boolean;
        session?: SessionSummary;
        first_message?: Message;
      }
    | null;
}

export interface AnnotatorStats {
  total_tokens: number;
  tokens_per_topic: Record<string, number>;
  tokens_per_day: Record<string, number>;
}

export interface AdminOverview {
  global_tokens: number;
  sessions_per_topic: Record<string, number>;
  tokens_per_topic: Record<string, number>;
  annotator_rankings: Array<{ annotator_id: number | string; total_tokens: number }>;
  tokens_per_day: Record<string, number>;
}

export interface ApiError {
  detail?: string | Record<string, string[]> | null;
}
