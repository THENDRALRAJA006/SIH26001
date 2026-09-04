/**
 * RiskGauge — SVG semicircle gauge showing risk probability [0–1]
 *
 * Renders a smooth arc coloured by risk level with an animated needle.
 */
import { useEffect, useRef } from "react";

const levelColour = (score) => {
  if (score >= 0.60) return "var(--risk-high)";
  if (score >= 0.30) return "var(--risk-medium)";
  return "var(--risk-low)";
};

export default function RiskGauge({ score = 0, size = 200, label = "" }) {
  const arcRef = useRef(null);
  const needleRef = useRef(null);

  const cx = size / 2;
  const cy = size / 2 + 10;
  const R  = (size / 2) * 0.8;
  const startAngle = -180; // degrees, left
  const endAngle   = 0;    // degrees, right

  // Arc path helper
  const polarToCart = (angle, r = R) => {
    const rad = (angle * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };

  const describeArc = (from, to) => {
    const s = polarToCart(from);
    const e = polarToCart(to);
    const large = to - from > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${R} ${R} 0 ${large} 1 ${e.x} ${e.y}`;
  };

  const scoreAngle = startAngle + score * 180;
  const colour = levelColour(score);
  const pct = Math.round(score * 100);

  return (
    <div id="risk-gauge" style={{ textAlign: "center", userSelect: "none" }}>
      <svg width={size} height={size * 0.65} viewBox={`0 0 ${size} ${size * 0.65}`}>
        {/* Track arc */}
        <path
          d={describeArc(-180, 0)}
          fill="none"
          stroke="var(--bg-3)"
          strokeWidth={14}
          strokeLinecap="round"
        />
        {/* Coloured progress arc */}
        <path
          ref={arcRef}
          d={describeArc(-180, scoreAngle)}
          fill="none"
          stroke={colour}
          strokeWidth={14}
          strokeLinecap="round"
          style={{
            filter: `drop-shadow(0 0 6px ${colour})`,
            transition: "all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
          }}
        />
        {/* Risk level tick marks */}
        {[0, 0.3, 0.6, 1].map((tick) => {
          const a = -180 + tick * 180;
          const inner = polarToCart(a, R - 10);
          const outer = polarToCart(a, R + 4);
          return (
            <line
              key={tick}
              x1={inner.x} y1={inner.y} x2={outer.x} y2={outer.y}
              stroke="var(--text-muted)" strokeWidth={1.5}
            />
          );
        })}
        {/* Needle */}
        <line
          ref={needleRef}
          x1={cx} y1={cy}
          x2={polarToCart(scoreAngle, R - 18).x}
          y2={polarToCart(scoreAngle, R - 18).y}
          stroke={colour}
          strokeWidth={2.5}
          strokeLinecap="round"
          style={{ transition: "all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)" }}
        />
        {/* Centre dot */}
        <circle cx={cx} cy={cy} r={5} fill={colour}
          style={{ filter: `drop-shadow(0 0 4px ${colour})` }} />
        {/* Percentage label */}
        <text
          x={cx} y={cy - 30}
          textAnchor="middle"
          fontSize={28}
          fontWeight={700}
          fill={colour}
          fontFamily="Inter, sans-serif"
          style={{ transition: "fill 0.5s" }}
        >
          {pct}%
        </text>
        {/* Sub-label */}
        {label && (
          <text x={cx} y={cy - 10} textAnchor="middle" fontSize={11}
            fill="var(--text-secondary)" fontFamily="Inter, sans-serif">
            {label}
          </text>
        )}
        {/* Scale labels */}
        <text x={polarToCart(-180, R + 14).x - 2} y={polarToCart(-180, R + 14).y + 4}
          textAnchor="middle" fontSize={9} fill="var(--text-muted)" fontFamily="Inter">0</text>
        <text x={polarToCart(0, R + 14).x + 4} y={polarToCart(0, R + 14).y + 4}
          textAnchor="middle" fontSize={9} fill="var(--text-muted)" fontFamily="Inter">1.0</text>
      </svg>
    </div>
  );
}
