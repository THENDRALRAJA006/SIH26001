/**
 * App.jsx — LAND-JEPA Route Tree
 * ================================
 * Routes:
 *   /                   → LandingPage (monochrome reference entry + role selection)
 *   /citizen            → CitizenLayout (anonymous, frictionless citizen experience)
 *   /officer/login      → OfficerLoginPage (AI secure login)
 *   /officer/dashboard  → OfficerLayout (command center overview)
 *   /officer/forecast   → OfficerLayout (multi-horizon forecast)
 *   /officer/gis        → OfficerLayout (GIS analytics & layers)
 *   /officer/alerts     → OfficerLayout (alerts center)
 *   /officer/reports    → OfficerLayout (citizen report triage)
 *   /officer/analytics  → OfficerLayout (AI model performance)
 *   /officer/model      → OfficerLayout (model registry & thresholds)
 *   /officer            → redirect → /officer/dashboard
 *   *                   → redirect → /
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { Routes, Route, Navigate } from "react-router-dom";
import LandingPage       from "./pages/LandingPage";
import CitizenLayout     from "./pages/CitizenLayout";
import CitizenReportPage from "./pages/CitizenReportPage";
import OfficerLoginPage  from "./pages/OfficerLoginPage";
import OfficerLayout     from "./pages/OfficerLayout";
import WeatherPage       from "./pages/WeatherPage";
import TerrainPage       from "./pages/TerrainPage";
import DataPage          from "./pages/DataPage";
import AiPage            from "./pages/AiPage";
import EarlyWarningPage  from "./pages/EarlyWarningPage";
import SystemStatusPage  from "./pages/SystemStatusPage";
import NotificationsPage from "./pages/NotificationsPage";

export default function App() {
  return (
    <Routes>
      {/* Landing — premium reference entry page */}
      <Route path="/"                   element={<LandingPage />} />

      {/* Top Nav Subsystem Pages */}
      <Route path="/weather"            element={<WeatherPage />} />
      <Route path="/terrain"            element={<TerrainPage />} />
      <Route path="/data"               element={<DataPage />} />
      <Route path="/ai"                 element={<AiPage />} />
      <Route path="/early-warning"      element={<EarlyWarningPage />} />
      <Route path="/system-status"      element={<SystemStatusPage />} />
      <Route path="/notifications"      element={<NotificationsPage />} />

      {/* Citizen Flow */}
      <Route path="/citizen"            element={<CitizenLayout />} />
      <Route path="/citizen/report"     element={<CitizenReportPage />} />

      {/* Officer AI Secure Login */}
      <Route path="/officer/login"      element={<OfficerLoginPage />} />

      {/* Dedicated Officer Command Center Routes */}
      <Route path="/officer/dashboard"  element={<OfficerLayout />} />
      <Route path="/officer/forecast"   element={<OfficerLayout />} />
      <Route path="/officer/gis"        element={<OfficerLayout />} />
      <Route path="/officer/alerts"     element={<OfficerLayout />} />
      <Route path="/officer/reports"    element={<OfficerLayout />} />
      <Route path="/officer/analytics"  element={<OfficerLayout />} />
      <Route path="/officer/model"      element={<OfficerLayout />} />
      <Route path="/officer/benchmark"  element={<OfficerLayout />} />
      <Route path="/officer/prediction" element={<OfficerLayout />} />
      <Route path="/officer/settings"   element={<OfficerLayout />} />
      <Route path="/officer/notifications" element={<Navigate to="/notifications" replace />} />

      {/* Convenience redirects */}
      <Route path="/officer"            element={<Navigate to="/officer/dashboard" replace />} />
      <Route path="/live-gis"           element={<Navigate to="/officer/gis" replace />} />

      {/* 404 fallback */}
      <Route path="*"                   element={<Navigate to="/" replace />} />
    </Routes>
  );
}
