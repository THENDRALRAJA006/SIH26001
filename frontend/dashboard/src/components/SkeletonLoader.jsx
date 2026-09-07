/**
 * SkeletonLoader.jsx
 * ===================
 * Premium skeleton loading states for LAND-JEPA dashboards.
 */

export function SkeletonLine({ width = "100%", height = 14, dark = true }) {
  return (
    <div
      className={dark ? "skeleton-dark" : "skeleton"}
      style={{ width, height, borderRadius: 4 }}
      aria-hidden="true"
    />
  );
}

export function SkeletonCard({ dark = true, rows = 3, style = {} }) {
  return (
    <div
      style={{
        padding: "20px 24px",
        borderRadius: 16,
        border: dark ? "1px solid rgba(255,255,255,0.07)" : "1px solid rgba(0,0,0,0.06)",
        background: dark ? "rgba(21,24,31,0.7)" : "#FFFFFF",
        display: "flex",
        flexDirection: "column",
        gap: 10,
        ...style,
      }}
      aria-busy="true"
      aria-label="Loading..."
    >
      <SkeletonLine width="45%" height={12} dark={dark} />
      <SkeletonLine width="90%" height={20} dark={dark} />
      {Array.from({ length: rows - 1 }).map((_, i) => (
        <SkeletonLine key={i} width={i % 2 === 0 ? "80%" : "65%"} dark={dark} />
      ))}
    </div>
  );
}

export function SkeletonMap({ dark = true }) {
  return (
    <div
      className={dark ? "skeleton-dark" : "skeleton"}
      style={{ width: "100%", height: "100%", minHeight: 280, borderRadius: 16 }}
      aria-busy="true"
      aria-label="Loading map..."
    />
  );
}

export function SkeletonTable({ rows = 5, cols = 4, dark = true }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {/* Header */}
      <div style={{ display: "flex", gap: 16, paddingBottom: 8, borderBottom: `1px solid ${dark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)"}` }}>
        {Array.from({ length: cols }).map((_, i) => (
          <SkeletonLine key={i} width={`${100 / cols}%`} height={10} dark={dark} />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} style={{ display: "flex", gap: 16, alignItems: "center" }}>
          {Array.from({ length: cols }).map((_, c) => (
            <SkeletonLine key={c} width={c === 0 ? "40%" : `${100 / cols}%`} height={12} dark={dark} />
          ))}
        </div>
      ))}
    </div>
  );
}
