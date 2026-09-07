import React, { useState, useRef, useEffect } from "react";
import { useLanguage, LANGUAGES } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";

export default function LanguageSelector({ compact = false, style = {} }) {
  const { language, setLanguage, currentLanguageObj, t } = useLanguage();
  const { isDark } = useTheme();
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  // Close when clicking outside
  useEffect(() => {
    function handleClickOutside(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("touchstart", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("touchstart", handleClickOutside);
    };
  }, [open]);

  // Keyboard accessibility
  function handleKeyDown(e) {
    if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div
      ref={containerRef}
      onKeyDown={handleKeyDown}
      style={{
        position: "relative",
        display: "inline-block",
        ...style,
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-label={t("a11y.langSelectorLabel") || "Select interface language"}
        aria-haspopup="listbox"
        aria-expanded={open}
        style={{
          display: "flex",
          alignItems: "center",
          gap: compact ? 4 : 6,
          padding: compact ? "4px 8px" : "6px 12px",
          borderRadius: 20,
          background: isDark ? "rgba(255, 255, 255, 0.06)" : "#FFFFFF",
          border: isDark ? "1px solid rgba(255, 255, 255, 0.12)" : "1px solid rgba(0, 0, 0, 0.09)",
          cursor: "pointer",
          fontFamily: "inherit",
          fontSize: compact ? 11 : 12,
          fontWeight: 600,
          color: isDark ? "#E2E8F0" : "#0F172A",
          boxShadow: isDark ? "none" : "0 1px 3px rgba(0, 0, 0, 0.04)",
          transition: "background 0.2s, border-color 0.2s, transform 0.1s",
          userSelect: "none",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = isDark ? "rgba(255, 255, 255, 0.10)" : "#F8FAFC";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = isDark ? "rgba(255, 255, 255, 0.06)" : "#FFFFFF";
        }}
      >
        <span aria-hidden="true" style={{ fontSize: 13, lineHeight: 1 }}>🌐</span>
        <span>{currentLanguageObj.native}</span>
        <span
          style={{
            fontSize: 9,
            color: isDark ? "#64748B" : "#94A3B8",
            transform: open ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
            lineHeight: 1,
          }}
        >
          ▾
        </span>
      </button>

      {open && (
        <div
          role="listbox"
          aria-label={t("a11y.langSelectorLabel") || "Languages"}
          style={{
            position: "absolute",
            top: "calc(100% + 6px)",
            right: 0,
            minWidth: 155,
            background: isDark ? "#15181F" : "#FFFFFF",
            borderRadius: 12,
            boxShadow: isDark
              ? "0 14px 40px rgba(0, 0, 0, 0.55), 0 2px 8px rgba(0, 0, 0, 0.35)"
              : "0 14px 40px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.04)",
            border: isDark ? "1px solid rgba(255, 255, 255, 0.12)" : "1px solid rgba(0, 0, 0, 0.08)",
            padding: 6,
            zIndex: 9999,
            animation: "fadeIn 0.15s ease-out",
          }}
        >
          <div
            style={{
              padding: "4px 8px 6px 8px",
              fontSize: 9.5,
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: isDark ? "#64748B" : "#94A3B8",
              borderBottom: isDark ? "1px solid rgba(255, 255, 255, 0.06)" : "1px solid #F1F5F9",
              marginBottom: 4,
            }}
          >
            {t("a11y.langSelectorLabel") || "Language"}
          </div>

          {LANGUAGES.map((langItem) => {
            const isSelected = language === langItem.code;
            return (
              <div
                key={langItem.code}
                role="option"
                aria-selected={isSelected}
                tabIndex={0}
                onClick={() => {
                  setLanguage(langItem.code);
                  setOpen(false);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setLanguage(langItem.code);
                    setOpen(false);
                  }
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "7px 10px",
                  fontSize: 12,
                  fontWeight: isSelected ? 700 : 500,
                  color: isSelected
                    ? isDark ? "#FFFFFF" : "#0F172A"
                    : isDark ? "#94A3B8" : "#475569",
                  background: isSelected
                    ? isDark ? "rgba(255, 255, 255, 0.10)" : "#F1F5F9"
                    : "transparent",
                  borderRadius: 7,
                  cursor: "pointer",
                  transition: "background 0.15s",
                  outline: "none",
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.background = isDark
                      ? "rgba(255, 255, 255, 0.05)"
                      : "#F8FAFC";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.background = "transparent";
                  }
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 13 }}>{langItem.native}</span>
                  <span
                    style={{
                      fontSize: 10,
                      color: isDark ? "#64748B" : "#94A3B8",
                      fontWeight: 400,
                    }}
                  >
                    {langItem.label}
                  </span>
                </div>
                {isSelected && (
                  <span
                    style={{
                      fontSize: 11,
                      color: isDark ? "#38BDF8" : "#0284C7",
                      fontWeight: 800,
                    }}
                  >
                    ✓
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
