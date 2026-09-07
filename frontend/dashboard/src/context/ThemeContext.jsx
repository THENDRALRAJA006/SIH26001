/**
 * ThemeContext.jsx
 * =================
 * LAND-JEPA — Global Light / Dark Theme Provider
 * Applies `data-theme="dark|light"` on <html> so all CSS vars respond.
 * Persists preference to localStorage.
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { createContext, useContext, useState, useEffect, useCallback } from "react";

const ThemeContext = createContext(null);

const STORAGE_KEY = "lj_theme";
const DEFAULT_THEME = "dark"; // Officer platform defaults dark

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) || DEFAULT_THEME;
    } catch {
      return DEFAULT_THEME;
    }
  });

  // Apply data-theme to <html> for CSS var cascade
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem(STORAGE_KEY, theme); } catch {}
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(t => (t === "dark" ? "light" : "dark"));
  }, []);

  const isDark = theme === "dark";

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, isDark }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
  return ctx;
}
