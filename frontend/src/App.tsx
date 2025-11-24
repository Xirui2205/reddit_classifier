import React from 'react';
import { Routes, Route, Navigate, Outlet } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import AnnotationHubPage from './pages/AnnotationHubPage';
import ChatAnnotationPage from './pages/ChatAnnotationPage';
import TranslationAnnotationPage from './pages/TranslationAnnotationPage';
import PersonaChatPage from './pages/PersonaChatPage';
import LeaderboardPage from './pages/LeaderboardPage';
import ProfilePage from './pages/ProfilePage';
import AdminOverviewPage from './pages/admin/AdminOverviewPage';
import AdminUsersPage from './pages/admin/AdminUsersPage';
import AdminTopicsPage from './pages/admin/AdminTopicsPage';
import AdminPersonasPage from './pages/admin/AdminPersonasPage';
import AdminStatsPage from './pages/admin/AdminStatsPage';
import ReviewQueuePage from './pages/review/ReviewQueuePage';
import ReviewConversationPage from './pages/review/ReviewConversationPage';
import NotFoundPage from './pages/NotFoundPage';
import AppLayout from './components/layout/AppLayout';
import { ProtectedRoute } from './components/common/ProtectedRoute';

const LayoutShell: React.FC = () => (
  <AppLayout>
    <Outlet />
  </AppLayout>
);

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        element={
          <ProtectedRoute>
            <LayoutShell />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/annotate" element={<AnnotationHubPage />} />
        <Route path="/annotate/chat" element={<ChatAnnotationPage />} />
        <Route path="/annotate/translation" element={<TranslationAnnotationPage />} />
        <Route path="/annotate/persona" element={<PersonaChatPage />} />
        <Route path="/leaderboard" element={<LeaderboardPage />} />
        <Route path="/profile" element={<ProfilePage />} />
        <Route
          path="/admin"
          element={
            <ProtectedRoute roles={["admin"]}>
              <Outlet />
            </ProtectedRoute>
          }
        >
          <Route index element={<AdminOverviewPage />} />
          <Route path="users" element={<AdminUsersPage />} />
          <Route path="topics" element={<AdminTopicsPage />} />
          <Route path="personas" element={<AdminPersonasPage />} />
          <Route path="stats" element={<AdminStatsPage />} />
        </Route>
        <Route
          path="/review"
          element={
            <ProtectedRoute roles={["admin"]}>
              <Outlet />
            </ProtectedRoute>
          }
        >
          <Route index element={<ReviewQueuePage />} />
          <Route path=":id" element={<ReviewConversationPage />} />
        </Route>
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
};

export default App;
