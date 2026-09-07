/**
 * OfficerAuthContext.jsx
 * ========================
 * JWT-based officer authentication context.
 * Calls real backend /api/v1/auth/login endpoint.
 * Does NOT implement fake frontend-only auth.
 */
import { createContext, useContext, useState, useCallback } from "react";

const OfficerAuthContext = createContext(null);

const TOKEN_KEY = "lj_officer_token";
const ROLE_KEY  = "lj_officer_role";

export function OfficerAuthProvider({ children }) {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY));
  const [role,  setRole]  = useState(() => sessionStorage.getItem(ROLE_KEY));
  const [user,  setUser]  = useState(null);

  const isAuthenticated = !!(token && role === "officer");

  const login = useCallback(async (officerId, password) => {
    const res = await fetch("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: officerId, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || "Authentication failed");
    }
    const data = await res.json();
    const jwt  = data.access_token;
    const userRole = data.role || data.user?.role || "officer";
    sessionStorage.setItem(TOKEN_KEY, jwt);
    sessionStorage.setItem(ROLE_KEY, userRole);
    setToken(jwt);
    setRole(userRole);
    setUser(data.user || { officer_id: officerId });
    return data;
  }, []);

  const logout = useCallback(() => {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(ROLE_KEY);
    setToken(null);
    setRole(null);
    setUser(null);
  }, []);

  return (
    <OfficerAuthContext.Provider value={{ token, role, user, isAuthenticated, login, logout }}>
      {children}
    </OfficerAuthContext.Provider>
  );
}

export function useOfficerAuth() {
  const ctx = useContext(OfficerAuthContext);
  if (!ctx) throw new Error("useOfficerAuth must be used within OfficerAuthProvider");
  return ctx;
}

/** Auth header for officer API calls */
export function getOfficerHeaders() {
  const token = sessionStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}
