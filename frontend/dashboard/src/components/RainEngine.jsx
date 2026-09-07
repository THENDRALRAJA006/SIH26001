/**
 * RainEngine.jsx
 * ================
 * LAND-JEPA — Cinematic Physics Rain with 3D Glass Collision
 *
 * Core physics:
 *  - 3 Depth Layers: bg (alpha-fade), mid (collision physics), fg (full physics)
 *  - UI Card Glass Collision:
 *      · Drops detect card edges frame-by-frame (line-segment vs rect-edge test)
 *      · TOP edge hit: drop deflects upward, spawns edge impact ring + glass drip
 *      · SIDE edge hit: drop deflects sideways, spawns smaller splash
 *      · Glass Drips: teardrops slide DOWN the card face with moisture trail + light gleam
 *      · Impact Gleam: brief white highlight at collision point (3D glass illusion)
 *      · Condensation dots: faint micro-droplets on card surface
 *  - Natural wind gust oscillation (sin composite)
 *  - Motion blur (shadowBlur) on foreground drops
 *  - Atmospheric mist + ground splashes
 *  - prefers-reduced-motion support
 *  - Device-adaptive particle count (mobile/tablet/desktop)
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useEffect, useRef, memo } from "react";

/* ── Intensity configs ─────────────────────────────────── */
const INTENSITY_CONFIG = {
  none:     { bg: 0,   mid: 0,   fg: 0,   spd: 0,    wind: 0.3, mist: 0,     splash: 0    },
  low:      { bg: 75,  mid: 30,  fg: 8,   spd: 0.75, wind: 0.7, mist: 0.010, splash: 0.15 },
  moderate: { bg: 170, mid: 75,  fg: 20,  spd: 1.0,  wind: 1.3, mist: 0.025, splash: 0.38 },
  heavy:    { bg: 260, mid: 120, fg: 35,  spd: 1.45, wind: 2.0, mist: 0.052, splash: 0.72 },
  critical: { bg: 360, mid: 175, fg: 52,  spd: 1.85, wind: 2.8, mist: 0.080, splash: 1.0  },
};

const rand  = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* ── Line-segment to rect-edge collision ─────────────── */
// Returns {type:'top'|'left'|'right', ix, iy, rectIdx} or null
function detectEdgeCollision(x0, y0, x1, y1, rects) {
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i];
    const MARGIN = 1.5;

    // TOP edge: drop was above card, now crosses top
    if (y0 < r.y + MARGIN && y1 >= r.y - MARGIN) {
      // Interpolate x at y = r.y
      const t  = y0 === y1 ? 0.5 : (r.y - y0) / (y1 - y0);
      const ix = x0 + t * (x1 - x0);
      if (ix >= r.x - 4 && ix <= r.x + r.w + 4) {
        return { type: "top", ix, iy: r.y, rectIdx: i, rect: r };
      }
    }

    // LEFT edge
    if (x0 < r.x + MARGIN && x1 >= r.x - MARGIN) {
      const t  = x0 === x1 ? 0.5 : (r.x - x0) / (x1 - x0);
      const iy = y0 + t * (y1 - y0);
      if (iy >= r.y + 4 && iy <= r.y + r.h - 4) {
        return { type: "left", ix: r.x, iy, rectIdx: i, rect: r };
      }
    }

    // RIGHT edge
    if (x0 > r.x + r.w - MARGIN && x1 <= r.x + r.w + MARGIN) {
      const t  = x0 === x1 ? 0.5 : (r.x + r.w - x0) / (x1 - x0);
      const iy = y0 + t * (y1 - y0);
      if (iy >= r.y + 4 && iy <= r.y + r.h - 4) {
        return { type: "right", ix: r.x + r.w, iy, rectIdx: i, rect: r };
      }
    }
  }
  return null;
}

/* ── Inside a protected rect? ────────────────────────── */
function insideRect(x, y, rects) {
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i];
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return true;
  }
  return false;
}

/* ── Alpha fade near (not inside) protected zones ────── */
function zoneAlpha(x, y, rects, margin = 30) {
  let minDist = Infinity;
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i];
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return 0;
    const dx = Math.max(r.x - x, 0, x - (r.x + r.w));
    const dy = Math.max(r.y - y, 0, y - (r.y + r.h));
    minDist = Math.min(minDist, Math.sqrt(dx * dx + dy * dy));
  }
  return minDist >= margin ? 1 : minDist / margin;
}

/* ── RainEngine ─────────────────────────────────────── */
function RainEngine({ intensity = "moderate", isDarkTheme = false, protectedRects = [], className = "", style = {} }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true, willReadFrequently: false });
    if (!ctx) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let animId;
    let W = (canvas.width  = window.innerWidth);
    let H = (canvas.height = window.innerHeight);

    const isMobile  = W < 600;
    const isTablet  = W < 1100;
    const SCALE     = isMobile ? 0.30 : isTablet ? 0.58 : 1.0;

    const cfg = INTENSITY_CONFIG[intensity] || INTENSITY_CONFIG.moderate;
    const bgN  = Math.floor(cfg.bg  * SCALE);
    const midN = Math.floor(cfg.mid * SCALE);
    const fgN  = Math.floor(cfg.fg  * SCALE);

    const BASE_RGB = isDarkTheme ? "210,220,235" : "16,22,38";

    /* ── Drop factories ────────────────────────────── */
    function mkBg() {
      return {
        x: rand(-100, W + 100), y: rand(-H * 0.5, H * 1.1),
        spd: rand(9, 16) * cfg.spd, len: rand(10, 16),
        w: rand(0.45, 0.85), a: rand(0.08, 0.18),
        wo: rand(-0.2, 0.2), phase: rand(0, Math.PI * 2),
      };
    }

    function mkMid() {
      return {
        x: rand(-160, W + 160), y: rand(-H * 0.5, H * 1.05),
        vx: 0, vy: rand(17, 27) * cfg.spd,  // velocity
        spd: rand(17, 27) * cfg.spd,
        len: rand(20, 38), w: rand(1.0, 1.5),
        a: rand(0.22, 0.40), wo: rand(-0.35, 0.35),
        phase: rand(0, Math.PI * 2),
        deflected: false, deflectLife: 0,
      };
    }

    function mkFg() {
      return {
        x: rand(-220, W + 220), y: rand(-H * 0.4, H * 1.0),
        vx: 0, vy: rand(30, 50) * cfg.spd,
        spd: rand(30, 50) * cfg.spd,
        len: rand(46, 88), w: rand(1.6, 2.6),
        a: rand(0.50, 0.82), hr: rand(1.2, 2.2),
        wo: rand(-0.45, 0.45), phase: rand(0, Math.PI * 2),
        blur: rand(0.5, 1.8),
        deflected: false, deflectLife: 0,
      };
    }

    const bgDrops  = Array.from({ length: bgN  }, mkBg);
    const midDrops = Array.from({ length: midN }, mkMid);
    const fgDrops  = Array.from({ length: fgN  }, mkFg);

    /* ── Pool caps ────────────────────────────────── */
    const groundSplashes = [];  // elliptical ground impacts
    const edgeSplashes   = [];  // circular card-edge impacts
    const glassDrips     = [];  // teardrops sliding down card faces
    const impactGleams   = [];  // brief white flashes at impact point

    function addGroundSplash(x, y, sc = 1) {
      if (groundSplashes.length > 55) return;
      groundSplashes.push({ x, y, r: 1, maxR: rand(4, 12) * sc, a: 0.38 * sc, gr: rand(0.5, 1.0) });
    }

    function addEdgeSplash(ix, iy, isMajor = false) {
      if (edgeSplashes.length > 70) return;
      const count = isMajor ? 3 : 1;
      for (let k = 0; k < count; k++) {
        edgeSplashes.push({
          x: ix + rand(-3, 3), y: iy + rand(-2, 2),
          r: 0.5, maxR: rand(3, isMajor ? 9 : 5),
          a: isMajor ? 0.55 : 0.38, gr: rand(0.35, 0.8),
          // scatter particles
          scatter: isMajor ? [
            { x: ix, y: iy, vx: rand(-2, 2), vy: rand(-3, -0.5), a: 0.60, r: rand(0.8, 1.8) },
            { x: ix, y: iy, vx: rand(-2, 2), vy: rand(-2, -0.3), a: 0.45, r: rand(0.5, 1.2) },
          ] : [],
        });
      }
    }

    function addGlassDrip(ix, iy, rect) {
      if (glassDrips.length > 45) return;
      glassDrips.push({
        x: ix + rand(-4, 4),
        y: iy,
        vy: rand(0.3, 1.0),              // slow slide speed
        size: rand(1.8, 3.4),
        a: rand(0.28, 0.50),
        rect,                             // constrained to card face
        trail: [],
        wander: rand(-0.06, 0.06),        // slow horizontal drift on glass
        maxTrail: 28,
      });
    }

    function addImpactGleam(ix, iy) {
      if (impactGleams.length > 30) return;
      impactGleams.push({ x: ix, y: iy, a: 0.75, r: rand(2.5, 5), life: 6 });
    }

    /* ── Handle collision → deflect + spawn effects ── */
    function handleCollision(drop, col, wind) {
      const { type, ix, iy, rect } = col;
      const isMajor = drop.spd > 20; // fg drops are major

      addEdgeSplash(ix, iy, isMajor);
      addImpactGleam(ix, iy);
      if (isMajor || Math.random() < 0.6) addGlassDrip(ix, iy, rect);

      // Reflect velocity
      if (type === "top") {
        // Normal is (0, -1) → reflect Y
        drop.vy = -Math.abs(drop.vy) * rand(0.18, 0.32);
        drop.vx = (drop.vx || wind) + rand(-1.5, 1.5);
        drop.y  = iy - 2;
      } else if (type === "left") {
        drop.vx = -Math.abs(drop.vx || wind) * rand(0.25, 0.45) + rand(-0.5, 0.5);
        drop.x  = ix - 2;
      } else {
        drop.vx = Math.abs(drop.vx || wind) * rand(0.25, 0.45) + rand(-0.5, 0.5);
        drop.x  = ix + 2;
      }

      drop.deflected    = true;
      drop.deflectLife  = Math.floor(rand(5, 12));
      drop.a           *= rand(0.45, 0.70);
    }

    /* ── Resize ─────────────────────────────────── */
    const onResize = () => { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; };
    window.addEventListener("resize", onResize);

    let time = 0;

    /* ── Main render loop ────────────────────────── */
    const render = () => {
      time += 0.016;
      ctx.clearRect(0, 0, W, H);

      // Composite wind
      const wind = cfg.wind + Math.sin(time * 0.47) * 0.26 + Math.sin(time * 1.28) * 0.11;

      /* ═══ Ground Mist ════════════════════════════ */
      if (cfg.mist > 0) {
        const mg = ctx.createLinearGradient(0, H * 0.62, 0, H);
        mg.addColorStop(0, `rgba(${BASE_RGB}, 0)`);
        mg.addColorStop(0.45, `rgba(${BASE_RGB}, ${cfg.mist * 0.4})`);
        mg.addColorStop(1,   `rgba(${BASE_RGB}, ${cfg.mist})`);
        ctx.fillStyle = mg;
        ctx.fillRect(0, H * 0.60, W, H * 0.40);
      }

      /* ═══ LAYER 1: Background (alpha-fade, no physics) ═ */
      ctx.beginPath();
      for (let i = 0; i < bgDrops.length; i++) {
        const d = bgDrops[i];
        const wo = wind + d.wo + Math.sin(time * 0.85 + d.phase) * 0.07;
        d.y += d.spd;
        d.x += wo * 0.85;
        if (d.y > H + d.len) { Object.assign(d, mkBg()); d.y = -d.len; }

        const am = zoneAlpha(d.x, d.y, protectedRects, 28);
        if (am <= 0.01) continue;
        ctx.moveTo(d.x - wo * 1.1, d.y - d.len);
        ctx.lineTo(d.x, d.y);
      }
      ctx.strokeStyle = `rgba(${BASE_RGB}, 0.13)`;
      ctx.lineWidth = 0.65;
      ctx.stroke();

      /* ═══ LAYER 2: Midground (collision physics) ════════ */
      for (let i = 0; i < midDrops.length; i++) {
        const d = midDrops[i];

        // Update velocity
        if (!d.deflected) {
          d.vx = wind * 1.12 + d.wo + Math.sin(time * 1.05 + d.phase) * 0.10;
          d.vy = d.spd;
        } else {
          d.vy += 1.8; // gravity pulls back down
          d.vx *= 0.88;
          d.deflectLife--;
          if (d.deflectLife <= 0) { d.deflected = false; d.a = Math.min(d.a + 0.08, 0.40); }
        }

        const nx = d.x + d.vx;
        const ny = d.y + d.vy;

        // Collision detection (only when falling down and not already deflected)
        if (!d.deflected && d.vy > 0 && protectedRects.length > 0) {
          const col = detectEdgeCollision(d.x, d.y, nx, ny, protectedRects);
          if (col) { handleCollision(d, col, wind); continue; }
        }

        d.x = nx; d.y = ny;

        if (d.y > H + d.len) {
          if (!insideRect(d.x, H - 2, protectedRects) && Math.random() < cfg.splash * 0.4) {
            addGroundSplash(d.x, H - 2, 0.60);
          }
          Object.assign(d, mkMid());
        }

        // Skip if inside card
        if (insideRect(d.x, d.y, protectedRects)) continue;

        // Draw
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = `rgba(${BASE_RGB}, ${d.a})`;
        ctx.lineWidth = d.w;
        ctx.lineCap = "round";
        const ex = d.deflected ? d.x - d.vx * 1.5 : d.x - d.vx * 1.8;
        const ey = d.deflected ? d.y - d.vy * 1.0 : d.y - d.len;
        ctx.moveTo(ex, ey);
        ctx.lineTo(d.x, d.y);
        ctx.stroke();
        ctx.restore();
      }

      /* ═══ LAYER 3: Foreground (full physics + blur) ═════ */
      for (let i = 0; i < fgDrops.length; i++) {
        const d = fgDrops[i];

        if (!d.deflected) {
          d.vx = wind * 1.48 + d.wo + Math.sin(time * 1.38 + d.phase) * 0.16;
          d.vy = d.spd;
        } else {
          d.vy += 2.8;
          d.vx *= 0.82;
          d.deflectLife--;
          if (d.deflectLife <= 0) { d.deflected = false; d.a = Math.min(d.a + 0.12, 0.80); }
        }

        const nx = d.x + d.vx;
        const ny = d.y + d.vy;

        if (!d.deflected && d.vy > 0 && protectedRects.length > 0) {
          const col = detectEdgeCollision(d.x, d.y, nx, ny, protectedRects);
          if (col) { handleCollision(d, col, wind); continue; }
        }

        d.x = nx; d.y = ny;

        if (d.y > H + d.len + 10) {
          if (!insideRect(d.x, H - 4, protectedRects) && Math.random() < cfg.splash) {
            addGroundSplash(d.x, H - 4, 1.1);
          }
          Object.assign(d, mkFg());
        }

        if (insideRect(d.x, d.y, protectedRects)) continue;

        // Draw fg drop with gradient streak + head
        ctx.save();
        ctx.shadowBlur  = d.blur;
        ctx.shadowColor = `rgba(${BASE_RGB}, ${d.a * 0.38})`;

        const ex  = d.deflected ? d.x - d.vx * 2.0 : d.x - d.vx * 2.5;
        const ey  = d.deflected ? d.y - d.vy * 0.8 : d.y - d.len;
        const sg  = ctx.createLinearGradient(ex, ey, d.x, d.y);
        sg.addColorStop(0,    `rgba(${BASE_RGB}, 0)`);
        sg.addColorStop(0.55, `rgba(${BASE_RGB}, ${d.a * 0.50})`);
        sg.addColorStop(1,    `rgba(${BASE_RGB}, ${d.a})`);

        ctx.beginPath();
        ctx.strokeStyle = sg;
        ctx.lineWidth = d.w;
        ctx.lineCap = "round";
        ctx.moveTo(ex, ey);
        ctx.lineTo(d.x, d.y);
        ctx.stroke();

        // Teardrop head
        ctx.beginPath();
        ctx.fillStyle = `rgba(${BASE_RGB}, ${d.a * 0.90})`;
        ctx.arc(d.x, d.y, d.hr, 0, Math.PI * 2);
        ctx.fill();

        // Light gleam on fg drop (3D glass feel)
        ctx.beginPath();
        ctx.fillStyle = `rgba(255,255,255,${d.a * 0.28})`;
        ctx.ellipse(d.x - d.hr * 0.35, d.y - d.hr * 0.38, d.hr * 0.32, d.hr * 0.22, -0.4, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
      }

      /* ═══ CARD-SURFACE GLASS DRIPS ══════════════════════ */
      for (let i = glassDrips.length - 1; i >= 0; i--) {
        const dr = glassDrips[i];
        const r  = dr.rect;

        // Gravity + surface tension wander
        dr.vy   = Math.min(dr.vy + 0.018, 1.8);
        dr.x   += dr.wander;
        dr.y   += dr.vy;

        // Keep within card width with slight bounce-off-edge
        if (dr.x < r.x + 4)      { dr.wander = Math.abs(dr.wander);  dr.x = r.x + 4; }
        if (dr.x > r.x + r.w - 4){ dr.wander = -Math.abs(dr.wander); dr.x = r.x + r.w - 4; }

        // Trail
        dr.trail.push({ x: dr.x, y: dr.y });
        if (dr.trail.length > dr.maxTrail) dr.trail.shift();

        // Remove if it drips past bottom edge
        if (dr.y > r.y + r.h + 8) {
          // Small drip-off splash at bottom
          addEdgeSplash(dr.x, r.y + r.h, false);
          glassDrips.splice(i, 1);
          continue;
        }

        // Draw moisture trail
        if (dr.trail.length > 2) {
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(dr.trail[0].x, dr.trail[0].y);
          for (let t = 1; t < dr.trail.length; t++) {
            ctx.lineTo(dr.trail[t].x, dr.trail[t].y);
          }
          ctx.strokeStyle = `rgba(${BASE_RGB}, ${dr.a * 0.14})`;
          ctx.lineWidth   = dr.size * 0.55;
          ctx.lineCap = "round";
          ctx.stroke();
          ctx.restore();
        }

        // Teardrop body with radial gradient (3D water bead on glass)
        ctx.save();
        ctx.shadowBlur  = 3;
        ctx.shadowColor = `rgba(${BASE_RGB}, ${dr.a * 0.25})`;

        const tg = ctx.createRadialGradient(
          dr.x - dr.size * 0.22, dr.y - dr.size * 0.25, 0,
          dr.x, dr.y, dr.size * 1.6
        );
        tg.addColorStop(0,    `rgba(${BASE_RGB}, ${dr.a * 0.60})`);
        tg.addColorStop(0.55, `rgba(${BASE_RGB}, ${dr.a * 0.35})`);
        tg.addColorStop(1,    `rgba(${BASE_RGB}, 0)`);

        ctx.beginPath();
        ctx.fillStyle = tg;
        // Elongate teardrop downward
        ctx.ellipse(dr.x, dr.y + dr.size * 0.25, dr.size * 0.85, dr.size * 1.35, 0, 0, Math.PI * 2);
        ctx.fill();

        // ── 3D Light Gleam (this is the key to the glass illusion) ──
        // Main specular highlight (top-left)
        ctx.beginPath();
        ctx.fillStyle = `rgba(255,255,255,${dr.a * 0.62})`;
        ctx.ellipse(dr.x - dr.size * 0.28, dr.y - dr.size * 0.32, dr.size * 0.30, dr.size * 0.20, -0.5, 0, Math.PI * 2);
        ctx.fill();

        // Secondary smaller gleam
        ctx.beginPath();
        ctx.fillStyle = `rgba(255,255,255,${dr.a * 0.28})`;
        ctx.ellipse(dr.x - dr.size * 0.10, dr.y - dr.size * 0.50, dr.size * 0.14, dr.size * 0.10, -0.3, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
      }

      /* ═══ CARD-EDGE IMPACT SPLASHES ═════════════════════ */
      for (let i = edgeSplashes.length - 1; i >= 0; i--) {
        const s = edgeSplashes[i];
        s.r += s.gr;
        s.a -= 0.030;
        if (s.a <= 0 || s.r >= s.maxR) { edgeSplashes.splice(i, 1); continue; }

        // Impact ring
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = `rgba(${BASE_RGB}, ${s.a})`;
        ctx.lineWidth = 0.9;
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.stroke();

        // Scatter micro-particles
        if (s.scatter) {
          for (let k = 0; k < s.scatter.length; k++) {
            const sc = s.scatter[k];
            sc.x  += sc.vx;
            sc.y  += sc.vy;
            sc.vy += 0.4; // gravity
            sc.a  -= 0.035;
            if (sc.a > 0) {
              ctx.beginPath();
              ctx.fillStyle = `rgba(${BASE_RGB}, ${sc.a})`;
              ctx.arc(sc.x, sc.y, sc.r, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
        ctx.restore();
      }

      /* ═══ IMPACT GLEAMS (brief flash at collision point) ═ */
      for (let i = impactGleams.length - 1; i >= 0; i--) {
        const g = impactGleams[i];
        g.a -= 0.12;
        g.r += 0.5;
        g.life--;
        if (g.a <= 0 || g.life <= 0) { impactGleams.splice(i, 1); continue; }

        ctx.save();
        const gg = ctx.createRadialGradient(g.x, g.y, 0, g.x, g.y, g.r);
        gg.addColorStop(0, `rgba(255,255,255,${g.a})`);
        gg.addColorStop(0.5, `rgba(255,255,255,${g.a * 0.4})`);
        gg.addColorStop(1, `rgba(255,255,255,0)`);
        ctx.fillStyle = gg;
        ctx.beginPath();
        ctx.arc(g.x, g.y, g.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      /* ═══ GROUND SPLASHES ════════════════════════════════ */
      for (let i = groundSplashes.length - 1; i >= 0; i--) {
        const s = groundSplashes[i];
        s.r += s.gr;
        s.a -= 0.022;
        if (s.a <= 0 || s.r >= s.maxR) { groundSplashes.splice(i, 1); continue; }
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = `rgba(${BASE_RGB}, ${s.a})`;
        ctx.lineWidth = 0.8;
        ctx.ellipse(s.x, s.y, s.r * 2.0, s.r * 0.50, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      /* ═══ WET-GLASS EDGE VIGNETTE ════════════════════════ */
      if (!isDarkTheme) {
        const vg = ctx.createRadialGradient(W / 2, H / 2, H * 0.34, W / 2, H / 2, H * 0.84);
        vg.addColorStop(0, "rgba(16,22,38,0)");
        vg.addColorStop(1, "rgba(16,22,38,0.055)");
        ctx.fillStyle = vg;
        ctx.fillRect(0, 0, W, H);
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => { cancelAnimationFrame(animId); window.removeEventListener("resize", onResize); };
  }, [intensity, isDarkTheme, protectedRects]);

  return (
    <canvas
      ref={canvasRef}
      className={`rain-canvas ${className}`}
      aria-hidden="true"
      style={{
        position: "fixed", top: 0, left: 0,
        width: "100%", height: "100%",
        pointerEvents: "none", zIndex: 1, ...style,
      }}
    />
  );
}

export default memo(RainEngine);
