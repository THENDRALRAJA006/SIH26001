import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from "react";
import en from "../i18n/en.json";
import hi from "../i18n/hi.json";
import as from "../i18n/as.json";
import bn from "../i18n/bn.json";
import mni from "../i18n/mni.json";

export const LANGUAGES = [
  { code: "en", label: "English", native: "English", flag: "🇬🇧" },
  { code: "hi", label: "Hindi", native: "हिन्दी", flag: "🇮🇳" },
  { code: "as", label: "Assamese", native: "অসমীয়া", flag: "🇮🇳" },
  { code: "bn", label: "Bengali", native: "বাংলা", flag: "🇮🇳" },
  { code: "mni", label: "Manipuri", native: "মৈতেই", flag: "🇮🇳" },
];

const DICTIONARIES = { en, hi, as, bn, mni };
const STORAGE_KEY = "landjepa_language";

const LanguageContext = createContext({
  language: "en",
  setLanguage: () => {},
  t: (key) => key,
  languages: LANGUAGES,
  currentLanguageObj: LANGUAGES[0],
  formatNumber: (n) => String(n),
  formatDate: (d) => String(d),
  formatDuration: (h) => `${h} h`,
});

function resolveKey(obj, path) {
  if (!obj || !path) return undefined;
  const parts = path.split(".");
  let cur = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = cur[p];
  }
  return typeof cur === "string" ? cur : undefined;
}

export function LanguageProvider({ children }) {
  const [language, setLanguageState] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && DICTIONARIES[saved]) {
        return saved;
      }
    } catch {
      // ignore storage error
    }
    return "en";
  });

  const setLanguage = useCallback((code) => {
    if (!DICTIONARIES[code]) {
      console.warn(`[i18n] Language code not supported: ${code}, falling back to en`);
      code = "en";
    }
    setLanguageState(code);
    try {
      localStorage.setItem(STORAGE_KEY, code);
      document.documentElement.lang = code;
    } catch {
      // ignore storage error
    }
  }, []);

  useEffect(() => {
    try {
      document.documentElement.lang = language;
    } catch {
      // ignore
    }
  }, [language]);

  const currentLanguageObj = useMemo(() => {
    return LANGUAGES.find((l) => l.code === language) || LANGUAGES[0];
  }, [language]);

  /**
   * Translate key with fallback to English and param interpolation
   * e.g. t("risk.high"), t("common.hoursUnit")
   */
  const t = useCallback((key, params) => {
    if (!key) return "";
    const activeDict = DICTIONARIES[language] || en;
    let str = resolveKey(activeDict, key);

    // Fallback to English if missing
    if (str === undefined) {
      str = resolveKey(en, key);
      if (str === undefined) {
        console.warn(`[MISSING_TRANSLATION]: key "${key}" not found in language "${language}" or fallback "en"`);
        // Return readable segment rather than blank or undefined
        const parts = key.split(".");
        return parts[parts.length - 1];
      }
    }

    // Param interpolation: {name} -> params.name
    if (params && typeof params === "object") {
      Object.keys(params).forEach((k) => {
        str = str.replace(new RegExp(`\\{${k}\\}`, "g"), params[k]);
      });
    }

    return str;
  }, [language]);

  /**
   * Locale-aware number formatting
   */
  const formatNumber = useCallback((num, options) => {
    try {
      const locale = language === "mni" ? "bn-IN" : `${language}-IN`;
      return new Intl.NumberFormat(locale, options).format(num);
    } catch {
      return String(num);
    }
  }, [language]);

  /**
   * Locale-aware date formatting
   */
  const formatDate = useCallback((date, options) => {
    try {
      const d = date instanceof Date ? date : new Date(date);
      const locale = language === "mni" ? "bn-IN" : `${language}-IN`;
      return new Intl.DateTimeFormat(locale, options || {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(d);
    } catch {
      return String(date);
    }
  }, [language]);

  /**
   * Format lead time duration localized
   */
  const formatDuration = useCallback((hours) => {
    const unit = t("common.hoursUnit");
    return `${hours} ${unit}`;
  }, [t]);

  const value = useMemo(() => ({
    language,
    setLanguage,
    t,
    languages: LANGUAGES,
    currentLanguageObj,
    formatNumber,
    formatDate,
    formatDuration,
  }), [language, setLanguage, t, currentLanguageObj, formatNumber, formatDate, formatDuration]);

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return ctx;
}
