/**
 * LAND-JEPA Mobile — Risk Gauge (SVG semicircle)
 * Uses react-native-svg for cross-platform rendering.
 */
import React from "react";
import { View, Text, StyleSheet } from "react-native";
import Svg, { Path, Circle, Line, Text as SvgText } from "react-native-svg";
import { colours, font, riskColour } from "../theme";

function polarToCart(cx, cy, R, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + R * Math.cos(rad), y: cy + R * Math.sin(rad) };
}

function describeArc(cx, cy, R, startDeg, endDeg) {
  const s = polarToCart(cx, cy, R, startDeg);
  const e = polarToCart(cx, cy, R, endDeg);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${R} ${R} 0 ${large} 1 ${e.x} ${e.y}`;
}

export default function RiskGauge({ score = 0, size = 220 }) {
  const cx = size / 2;
  const cy = size / 2 + 8;
  const R  = size * 0.38;

  const startAngle = -180;
  const scoreAngle = startAngle + score * 180;
  const colour     = riskColour(score >= 0.6 ? "HIGH" : score >= 0.3 ? "MEDIUM" : "LOW");
  const pct        = Math.round(score * 100);

  const needleTip = polarToCart(cx, cy, R - 18, scoreAngle);

  return (
    <View style={{ alignItems: "center" }}>
      <Svg width={size} height={size * 0.62}>
        {/* Track */}
        <Path
          d={describeArc(cx, cy, R, -180, 0)}
          fill="none"
          stroke={colours.bg3}
          strokeWidth={13}
          strokeLinecap="round"
        />
        {/* Coloured arc */}
        <Path
          d={describeArc(cx, cy, R, -180, scoreAngle)}
          fill="none"
          stroke={colour}
          strokeWidth={13}
          strokeLinecap="round"
        />
        {/* Needle */}
        <Line
          x1={cx} y1={cy}
          x2={needleTip.x} y2={needleTip.y}
          stroke={colour}
          strokeWidth={3}
          strokeLinecap="round"
        />
        {/* Centre dot */}
        <Circle cx={cx} cy={cy} r={6} fill={colour} />
        {/* Percentage */}
        <SvgText
          x={cx} y={cy - 32}
          textAnchor="middle"
          fontSize={30}
          fontWeight="700"
          fill={colour}
        >
          {pct}%
        </SvgText>
        <SvgText
          x={cx} y={cy - 12}
          textAnchor="middle"
          fontSize={10}
          fill={colours.textSecondary}
        >
          Risk Probability
        </SvgText>
        {/* Scale labels */}
        <SvgText
          x={polarToCart(cx, cy, R + 14, -180).x}
          y={polarToCart(cx, cy, R + 14, -180).y + 4}
          textAnchor="middle" fontSize={9} fill={colours.textMuted}
        >0</SvgText>
        <SvgText
          x={polarToCart(cx, cy, R + 14, 0).x}
          y={polarToCart(cx, cy, R + 14, 0).y + 4}
          textAnchor="middle" fontSize={9} fill={colours.textMuted}
        >1</SvgText>
      </Svg>
    </View>
  );
}
