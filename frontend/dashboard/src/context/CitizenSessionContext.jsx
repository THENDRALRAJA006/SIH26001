/**
 * CitizenSessionContext.jsx
 * ==========================
 * Anonymous citizen session — no login required.
 * Stores location and a session ID in sessionStorage.
 */
import { createContext, useContext, useState, useCallback } from "react";

const CitizenSessionContext = createContext(null);

const SESSION_KEY = "lj_citizen_session";

function generateSessionId() {
  return "cit_" + Math.random().toString(36).slice(2, 11) + "_" + Date.now();
}

export function CitizenSessionProvider({ children }) {
  const [session, setSession] = useState(() => {
    try {
      const raw = sessionStorage.getItem(SESSION_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });

  const [locationStatus, setLocationStatus] = useState(
    session?.coords ? "granted" : "pending"
  );

  const initSession = useCallback((coords, locationName) => {
    const newSession = {
      session_id: generateSessionId(),
      coords,
      location_name: locationName || "Northeast India",
      started_at: new Date().toISOString(),
      role: "citizen",
    };
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(newSession));
    setSession(newSession);
    setLocationStatus(coords ? "granted" : "denied");
    return newSession;
  }, []);

  const requestLocation = useCallback(() => {
    return new Promise((resolve) => {
      if (!navigator.geolocation) {
        resolve(null);
        return;
      }
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        },
        () => resolve(null),
        { timeout: 8000 }
      );
    });
  }, []);

  const clearSession = useCallback(() => {
    sessionStorage.removeItem(SESSION_KEY);
    setSession(null);
    setLocationStatus("pending");
  }, []);

  return (
    <CitizenSessionContext.Provider
      value={{ session, locationStatus, initSession, requestLocation, clearSession }}
    >
      {children}
    </CitizenSessionContext.Provider>
  );
}

export function useCitizenSession() {
  const ctx = useContext(CitizenSessionContext);
  if (!ctx) throw new Error("useCitizenSession must be used within CitizenSessionProvider");
  return ctx;
}
