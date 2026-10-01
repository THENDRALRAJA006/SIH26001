/**
 * MountainScene3D.jsx
 * ====================
 * LAND-JEPA — Cinematic 3D Geospatial Mountain & Landslide Experience
 *
 * Renders a dramatic dark stormy Himalayan/Shillong-plateau mountain environment
 * using Three.js WebGL with:
 *  - Multi-ridge procedural terrain (dark green/charcoal rocky slopes)
 *  - Winding mountain highway visible on slope (dark wet asphalt + white dashes)
 *  - Dense pine forest silhouettes
 *  - 3D angled rain streaks with wind
 *  - Drifting valley mist strata
 *  - Slow cinematic drone camera with mouse parallax
 *  - CITIZEN click: camera dive → landslide cascade → monochrome → /citizen
 *  - OFFICER click: tactical recon → AI grid → InSAR vectors → /officer/login
 *  - Instanced rock boulders + dust plume (physics landslide)
 *  - Accessibility: prefers-reduced-motion bypass
 *  - Clean Three.js resource disposal
 *
 * SIH26001 · Team ZAIX · Northeast India
 */
import { useEffect, useRef, memo } from "react";
import * as THREE from "three";
import { cinematicAudio } from "../services/CinematicAudioEngine";

export const SCENE_STATES = {
  LANDING_IDLE: "LANDING_IDLE",
  CITIZEN_TRANSITION: "CITIZEN_TRANSITION",
  OFFICER_TRANSITION: "OFFICER_TRANSITION",
  LANDSLIDE_ACTIVE: "LANDSLIDE_ACTIVE",
  MONOCHROME_TRANSITION: "MONOCHROME_TRANSITION",
  ROUTE_TRANSITION: "ROUTE_TRANSITION",
  PAGE_ENTERED: "PAGE_ENTERED",
};

/* ─── Simplex-like multi-octave terrain noise ─────────── */
function terrainH(x, z) {
  // Large-scale ridges
  const r1 = Math.sin(x * 0.006 + 0.4)  * Math.cos(z * 0.008 - 0.2) * 80;
  const r2 = Math.cos(x * 0.011 - 1.2)  * Math.sin(z * 0.012 + 0.9) * 55;
  const r3 = Math.sin(x * 0.022 + z * 0.018 + 2.1) * 28;
  const r4 = Math.cos(x * 0.042 - z * 0.038 - 1.0) * 14;
  // Detail
  const d1 = Math.sin(x * 0.09  + z * 0.07)  * 6;
  const d2 = Math.cos(x * 0.14  - z * 0.12)  * 3;

  let h = r1 + r2 + r3 + r4 + d1 + d2;

  // Valley depression (river valley at z ≈ 180–300)
  const valley = Math.max(0, 1 - Math.abs(z - 230) / 80);
  h -= valley * 42;

  // Raise the back-wall mountain dramatically
  if (z < -20) h += (-z - 20) * 0.72;

  return h;
}

/* ─── Highway centreline as a function of x ──────────── */
function roadZ(x) {
  return -30 + Math.sin(x * 0.010) * 38 + Math.cos(x * 0.006) * 18;
}
function roadY(x) {
  // Follows slope + small grade
  const rz = roadZ(x);
  return terrainH(x, rz) + 0.55;
}

const MountainScene3D = memo(function MountainScene3D({
  state = SCENE_STATES.LANDING_IDLE,
  rainIntensity = "moderate",
  isDark = true,
  onTransitionComplete,
  style = {},
}) {
  const mountRef   = useRef(null);
  const stateRef   = useRef(state);

  useEffect(() => { stateRef.current = state; }, [state]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const prefersReducedMotion =
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let W = container.clientWidth  || window.innerWidth;
    let H = container.clientHeight || window.innerHeight;
    const isMobile = W < 768;

    /* ═══════════════════════════════════════════════════
       1. RENDERER + SCENE
    ═══════════════════════════════════════════════════ */
    const renderer = new THREE.WebGLRenderer({
      antialias: !isMobile,
      powerPreference: "high-performance",
      stencil: false,
    });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
    renderer.toneMapping       = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    renderer.shadowMap.enabled = false;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();

    // ALWAYS dark dramatic storm — the 3D mountain scene is always cinematic regardless of UI theme
    const SKY_COLOR = new THREE.Color(0x050c18);
    scene.background = SKY_COLOR;
    scene.fog = new THREE.FogExp2(0x08111f, isMobile ? 0.0032 : 0.0025);

    /* ═══════════════════════════════════════════════════
       2. CAMERA
    ═══════════════════════════════════════════════════ */
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.5, 2000);
    // Positioned like a drone hovering over the valley, looking toward the mountain wall
    const BASE_POS  = new THREE.Vector3(0,  88, 360);
    const BASE_LOOK = new THREE.Vector3(0,  28, -80);
    camera.position.copy(BASE_POS);
    camera.lookAt(BASE_LOOK);

    /* ═══════════════════════════════════════════════════
       3. LIGHTING  (dark dramatic storm)
    ═══════════════════════════════════════════════════ */
    scene.add(new THREE.AmbientLight(0x1a2a42, 2.8));

    // Primary storm-diffuse from top-right
    const sun = new THREE.DirectionalLight(0x6a92c0, 2.2);
    sun.position.set(200, 400, 150);
    scene.add(sun);

    // Cold indigo rim for ridge silhouettes
    const rim = new THREE.DirectionalLight(0x162050, 2.0);
    rim.position.set(-350, 180, -250);
    scene.add(rim);

    // Low warm fill from valley floor
    const fill = new THREE.DirectionalLight(0x2a1e14, 0.8);
    fill.position.set(0, -50, 300);
    scene.add(fill);

    /* ═══════════════════════════════════════════════════
       4. TERRAIN MESH
    ═══════════════════════════════════════════════════ */
    const TX = isMobile ? 100 : 160;
    const TZ = isMobile ?  88 : 140;
    const GEO = new THREE.PlaneGeometry(1200, 900, TX, TZ);
    GEO.rotateX(-Math.PI / 2);

    const posAttr = GEO.attributes.position;
    const colorArr = new Float32Array(posAttr.count * 3);

    // Unstable slope zone (road-cut scarp)
    const SCARP_X = 40, SCARP_Z = -38, SCARP_R = 52;

    for (let i = 0; i < posAttr.count; i++) {
      const vx = posAttr.getX(i);
      const vz = posAttr.getZ(i);
      const vy = terrainH(vx, vz);
      posAttr.setY(i, vy);

      const dScarp = Math.hypot(vx - SCARP_X, vz - SCARP_Z);

      // Colour by zone
      if (dScarp < SCARP_R) {
        // Exposed road-cut scarp — warm reddish-ochre mudstone (geologically realistic)
        const blend = dScarp / SCARP_R;
        colorArr[i * 3]     = THREE.MathUtils.lerp(0.52, 0.28, blend);
        colorArr[i * 3 + 1] = THREE.MathUtils.lerp(0.30, 0.22, blend);
        colorArr[i * 3 + 2] = THREE.MathUtils.lerp(0.14, 0.18, blend);
      } else if (vy > 92) {
        // High cliff faces — cold dark blue-slate rock
        const jit = Math.random() * 0.05;
        colorArr[i * 3]     = 0.16 + jit;
        colorArr[i * 3 + 1] = 0.20 + jit;
        colorArr[i * 3 + 2] = 0.28 + jit * 1.2;
      } else if (vy > 55) {
        // Mid-elevation: mixed rock outcrops
        const jit = Math.random() * 0.04;
        colorArr[i * 3]     = 0.14 + jit;
        colorArr[i * 3 + 1] = 0.20 + jit;
        colorArr[i * 3 + 2] = 0.20 + jit;
      } else if (vz > 180 && vz < 310) {
        // Valley riverbed — dark wet stone/gravel
        colorArr[i * 3]     = 0.08;
        colorArr[i * 3 + 1] = 0.10;
        colorArr[i * 3 + 2] = 0.16;
      } else {
        // Dense moist pine forest slope — deep saturated dark green
        const jitter = (Math.random() - 0.5) * 0.035;
        colorArr[i * 3]     = 0.048 + jitter * 0.6;
        colorArr[i * 3 + 1] = 0.118 + jitter;
        colorArr[i * 3 + 2] = 0.062 + jitter * 0.6;
      }
    }

    GEO.setAttribute("color", new THREE.BufferAttribute(colorArr, 3));
    GEO.computeVertexNormals();

    const terrainMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.92,
      metalness: 0.06,
    });
    scene.add(new THREE.Mesh(GEO, terrainMat));

    /* ═══════════════════════════════════════════════════
       5. HIGHWAY RIBBON  (the star of the scene!)
    ═══════════════════════════════════════════════════ */
    // Build CatmullRom road curve from x=-520 to x=520
    const roadPts = [];
    for (let rx = -520; rx <= 520; rx += 12) {
      const rz = roadZ(rx);
      const ry = roadY(rx);
      roadPts.push(new THREE.Vector3(rx, ry + 0.25, rz));
    }
    const roadCurve = new THREE.CatmullRomCurve3(roadPts);

    // Asphalt band – thin flat tube
    const roadGeo = new THREE.TubeGeometry(roadCurve, 200, 6.5, 4, false);
    const roadPos = roadGeo.attributes.position;
    for (let i = 0; i < roadPos.count; i++) {
      // Squash the tube vertically to a flat ribbon
      const y = roadPos.getY(i);
      const cy = roadPts[0].y; // approx
      roadPos.setY(i, roadPos.getY(i) * 0.05 + 0.28);
    }
    roadGeo.computeVertexNormals();

    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x0e1218,   // near-black wet asphalt
      roughness: 0.28,   // highly reflective wet sheen
      metalness: 0.55,
    });
    scene.add(new THREE.Mesh(roadGeo, roadMat));

    // Dashed centre line
    const ctrPts = roadCurve.getPoints(300).map(p =>
      new THREE.Vector3(p.x, p.y + 0.28, p.z)
    );
    const lineGeo = new THREE.BufferGeometry().setFromPoints(ctrPts);
    const lineMat = new THREE.LineDashedMaterial({
      color: 0xffffff, dashSize: 6, gapSize: 5, linewidth: 2,
    });
    const centerLine = new THREE.Line(lineGeo, lineMat);
    centerLine.computeLineDistances();
    scene.add(centerLine);

    // White edge line on the mountain side of road
    const edgePts = roadCurve.getPoints(300).map(p => {
      // Offset towards the mountain (negative z)
      const tan = roadCurve.getTangent(0);
      return new THREE.Vector3(p.x, p.y + 0.30, p.z - 7.0);
    });
    const edgeGeo = new THREE.BufferGeometry().setFromPoints(edgePts);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
    scene.add(new THREE.Line(edgeGeo, edgeMat));

    /* ═══════════════════════════════════════════════════
       6. GUARDRAIL
    ═══════════════════════════════════════════════════ */
    const railPts = roadCurve.getPoints(300).map(p =>
      new THREE.Vector3(p.x, p.y + 0.80, p.z + 7.2)
    );
    const railGeo = new THREE.BufferGeometry().setFromPoints(railPts);
    const railMat = new THREE.LineBasicMaterial({ color: 0x8899aa, linewidth: 1 });
    scene.add(new THREE.Line(railGeo, railMat));

    /* ═══════════════════════════════════════════════════
       7. RIVER
    ═══════════════════════════════════════════════════ */
    const riverGeo = new THREE.PlaneGeometry(1000, 48, 40, 8);
    riverGeo.rotateX(-Math.PI / 2);
    const riverP = riverGeo.attributes.position;
    for (let i = 0; i < riverP.count; i++) {
      const rx = riverP.getX(i);
      const rz = 230 + Math.sin(rx * 0.010) * 28;
      riverP.setZ(i, rz);
      riverP.setY(i, terrainH(rx, rz) + 0.15);
    }
    riverGeo.computeVertexNormals();
    const riverMat = new THREE.MeshStandardMaterial({
      color: 0x101e30,
      roughness: 0.06, metalness: 0.90,
      transparent: true, opacity: 0.85,
    });
    scene.add(new THREE.Mesh(riverGeo, riverMat));

    /* ═══════════════════════════════════════════════════
       8. PINE FOREST  (instanced cones)
    ═══════════════════════════════════════════════════ */
    const TREE_N = isMobile ? 120 : 280;
    const treeGeo = new THREE.ConeGeometry(3.8, 15, 5);
    const treeMat = new THREE.MeshStandardMaterial({ color: 0x091408, roughness: 0.97 });
    const trees   = new THREE.InstancedMesh(treeGeo, treeMat, TREE_N);
    const dummy   = new THREE.Object3D();
    let placed = 0;

    for (let att = 0; att < TREE_N * 4 && placed < TREE_N; att++) {
      const tx = (Math.random() - 0.5) * 900;
      const tz = (Math.random() - 0.5) * 700;
      const ty = terrainH(tx, tz);
      const distRoad  = Math.abs(tz - roadZ(tx));
      const distScarp = Math.hypot(tx - SCARP_X, tz - SCARP_Z);

      if (distRoad > 20 && distScarp > 48 && ty > 12 && ty < 110) {
        dummy.position.set(tx, ty + 7, tz);
        const s = 0.7 + Math.random() * 0.7;
        dummy.scale.set(s, s * (0.9 + Math.random() * 0.3), s);
        dummy.rotation.y = Math.random() * Math.PI * 2;
        dummy.updateMatrix();
        trees.setMatrixAt(placed++, dummy.matrix);
      }
    }
    trees.instanceMatrix.needsUpdate = true;
    scene.add(trees);

    /* ═══════════════════════════════════════════════════
       9. VALLEY MIST PLANES
    ═══════════════════════════════════════════════════ */
    const MIST_COL = 0x1a2e4a;
    const mistGroup = new THREE.Group();
    for (let m = 0; m < 7; m++) {
      const mg = new THREE.PlaneGeometry(700, 150);
      mg.rotateX(-Math.PI / 2.5);
      const mm = new THREE.MeshBasicMaterial({
        color: MIST_COL,
        transparent: true, opacity: 0.13 + m * 0.018, depthWrite: false,
      });
      const mMesh = new THREE.Mesh(mg, mm);
      mMesh.position.set((m - 3.5) * 80, 28 + m * 5, 80 + m * 28);
      mistGroup.add(mMesh);
    }
    scene.add(mistGroup);

    /* ═══════════════════════════════════════════════════
       10. 3D RAIN
    ═══════════════════════════════════════════════════ */
    const RAIN_N = isMobile ? 2000 : 5500;
    const rainGeo = new THREE.BufferGeometry();
    const rainPos = new Float32Array(RAIN_N * 6);   // 2 pts per streak
    const rainVel = new Float32Array(RAIN_N);
    const BOX = { W: 900, H: 320, D: 700 };

    for (let i = 0; i < RAIN_N; i++) {
      const rx = (Math.random() - 0.5) * BOX.W;
      const ry = Math.random() * BOX.H;
      const rz = (Math.random() - 0.5) * BOX.D;
      const len = 4.5 + Math.random() * 6;
      const wx = -0.28, wz = -0.14;

      rainPos[i * 6]     = rx;
      rainPos[i * 6 + 1] = ry;
      rainPos[i * 6 + 2] = rz;
      rainPos[i * 6 + 3] = rx + wx * len;
      rainPos[i * 6 + 4] = ry - len;
      rainPos[i * 6 + 5] = rz + wz * len;
      rainVel[i] = 120 + Math.random() * 100;
    }
    rainGeo.setAttribute("position", new THREE.BufferAttribute(rainPos, 3));

    const rainMat = new THREE.LineBasicMaterial({
      color: 0x90b4d0,
      transparent: true,
      opacity: 0.42,
      linewidth: 1,
    });
    const rainMesh = new THREE.LineSegments(rainGeo, rainMat);
    scene.add(rainMesh);

    /* ═══════════════════════════════════════════════════
       11. AI OVERLAY GROUP  (hidden until transition)
    ═══════════════════════════════════════════════════ */
    const aiGroup = new THREE.Group();
    aiGroup.visible = false;

    // Wireframe terrain grid on the unstable scarp
    const wireGeo = new THREE.PlaneGeometry(95, 80, 20, 18);
    wireGeo.rotateX(-Math.PI / 2);
    const wireP = wireGeo.attributes.position;
    for (let i = 0; i < wireP.count; i++) {
      const wx = wireP.getX(i) + SCARP_X;
      const wz = wireP.getZ(i) + SCARP_Z;
      wireP.setX(i, wx); wireP.setZ(i, wz);
      wireP.setY(i, terrainH(wx, wz) + 1.4);
    }
    wireGeo.computeVertexNormals();
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x00e8ff, wireframe: true, transparent: true, opacity: 0.0,
    });
    const wireMesh = new THREE.Mesh(wireGeo, wireMat);
    aiGroup.add(wireMesh);

    // InSAR displacement arrows
    for (let v = 0; v < 18; v++) {
      const ax = SCARP_X + (Math.random() - 0.5) * 60;
      const az = SCARP_Z + (Math.random() - 0.5) * 44;
      const ay = terrainH(ax, az) + 2.0;
      const dir = new THREE.Vector3(0.25, -0.60, 0.75).normalize();
      const len = 6 + Math.random() * 6;
      const col = v % 3 === 0 ? 0xff2233 : v % 2 === 0 ? 0xffaa00 : 0x00e8ff;
      aiGroup.add(new THREE.ArrowHelper(dir, new THREE.Vector3(ax, ay, az), len, col, 2.2, 1.2));
    }

    // Hazard boundary ring
    const bCurve = new THREE.EllipseCurve(SCARP_X, SCARP_Z, 40, 30, 0, 2 * Math.PI);
    const bPts   = bCurve.getPoints(60).map(p =>
      new THREE.Vector3(p.x, terrainH(p.x, p.y) + 1.8, p.y)
    );
    const bGeo = new THREE.BufferGeometry().setFromPoints(bPts);
    const bMat = new THREE.LineBasicMaterial({ color: 0xff2233, linewidth: 3, transparent: true, opacity: 0.0 });
    const boundLine = new THREE.LineLoop(bGeo, bMat);
    aiGroup.add(boundLine);

    scene.add(aiGroup);

    /* ═══════════════════════════════════════════════════
       12. LANDSLIDE DEBRIS  (instanced rocks + dust)
    ═══════════════════════════════════════════════════ */
    const ROCK_N = isMobile ? 55 : 130;
    const rockGeo = new THREE.DodecahedronGeometry(1.8, 1);
    const rockMat = new THREE.MeshStandardMaterial({ color: 0x3a3228, roughness: 0.93, metalness: 0.07 });
    const rockMesh = new THREE.InstancedMesh(rockGeo, rockMat, ROCK_N);

    const rocks = [];
    for (let r = 0; r < ROCK_N; r++) {
      const ix = SCARP_X + (Math.random() - 0.5) * 38;
      const iz = SCARP_Z - 5 - Math.random() * 28;
      const iy = terrainH(ix, iz) + 1.0;
      const sc = 0.45 + Math.random() * 1.8;

      dummy.position.set(ix, iy, iz);
      dummy.scale.set(sc, sc, sc);
      dummy.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
      dummy.updateMatrix();
      rockMesh.setMatrixAt(r, dummy.matrix);

      rocks.push({
        pos: new THREE.Vector3(ix, iy, iz),
        vel: new THREE.Vector3(0, 0, 0),
        rot: new THREE.Euler(Math.random() * Math.PI, Math.random() * Math.PI, 0),
        rotV: new THREE.Vector3((Math.random() - 0.5) * 9, (Math.random() - 0.5) * 9, 0),
        sc, active: false, settled: false,
        releaseDelay: 0.55 + Math.random() * 1.0,
      });
    }
    rockMesh.instanceMatrix.needsUpdate = true;
    scene.add(rockMesh);

    // Dust billboards
    const DUST_N = isMobile ? 22 : 48;
    const dustGeo = new THREE.PlaneGeometry(14, 14);
    const dustGroup = new THREE.Group();
    const dustParticles = [];

    for (let d = 0; d < DUST_N; d++) {
      const dm = new THREE.MeshBasicMaterial({
        color: 0x6a5040, transparent: true, opacity: 0, depthWrite: false,
      });
      const dMesh = new THREE.Mesh(dustGeo, dm);
      dMesh.rotation.x = -Math.PI / 3.5;
      dMesh.position.set(
        SCARP_X + (Math.random() - 0.5) * 28,
        terrainH(SCARP_X, SCARP_Z) + 2,
        SCARP_Z + Math.random() * 18
      );
      dustGroup.add(dMesh);
      dustParticles.push({ mesh: dMesh, life: 0, active: false, delay: 1.1 + Math.random() * 0.9, maxS: 3.5 + Math.random() * 4 });
    }
    scene.add(dustGroup);

    /* ═══════════════════════════════════════════════════
       13. HAZARD SPOTLIGHT  (Officer transition)
    ═══════════════════════════════════════════════════ */
    const hSpot = new THREE.SpotLight(0xff4422, 0, 400, Math.PI / 5, 0.6, 1.0);
    hSpot.position.set(SCARP_X + 5, 190, SCARP_Z + 60);
    hSpot.target.position.set(SCARP_X, 55, SCARP_Z);
    scene.add(hSpot); scene.add(hSpot.target);

    /* ═══════════════════════════════════════════════════
       14. MOUSE PARALLAX
    ═══════════════════════════════════════════════════ */
    let mouseX = 0, mouseY = 0;
    const onMouseMove = e => {
      mouseX = (e.clientX / window.innerWidth)  * 2 - 1;
      mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    };
    window.addEventListener("mousemove", onMouseMove, { passive: true });

    /* ═══════════════════════════════════════════════════
       15. ANIMATION LOOP
    ═══════════════════════════════════════════════════ */
    const clock = new THREE.Clock();
    let rafId = null;
    let transStart = null;
    const TRANS_DUR = 3.0;
    let completeFired = false;

    const rainMults = { none: 0, low: 0.6, moderate: 1.0, heavy: 1.5, critical: 2.2 };

    function animate() {
      rafId = requestAnimationFrame(animate);
      const dt  = Math.min(clock.getDelta(), 0.05);
      const t   = clock.getElapsedTime();
      const cur = stateRef.current;

      /* ─── Rain update ──────────────────────────────── */
      const rMult = (cur !== SCENE_STATES.LANDING_IDLE) ? 2.2 : (rainMults[rainIntensity] || 1.0);
      const rp = rainMesh.geometry.attributes.position.array;
      for (let i = 0; i < RAIN_N; i++) {
        const fall = rainVel[i] * dt * rMult;
        rp[i * 6 + 1] -= fall;
        rp[i * 6 + 4] -= fall;
        if (rp[i * 6 + 1] < -12) {
          const nx = (Math.random() - 0.5) * BOX.W;
          const nz = (Math.random() - 0.5) * BOX.D;
          const ny = BOX.H * 0.88 + Math.random() * 40;
          const ln = 4.5 + Math.random() * 6;
          rp[i * 6] = nx; rp[i * 6 + 1] = ny; rp[i * 6 + 2] = nz;
          rp[i * 6 + 3] = nx - 0.28 * ln; rp[i * 6 + 4] = ny - ln; rp[i * 6 + 5] = nz - 0.14 * ln;
        }
      }
      rainMesh.geometry.attributes.position.needsUpdate = true;

      /* ─── Mist drift ────────────────────────────────── */
      for (const child of mistGroup.children) {
        child.position.x += dt * 5.5;
        if (child.position.x > 420) child.position.x = -420;
      }

      /* ─── River oscillation ─────────────────────────── */
      riverMat.opacity = 0.80 + Math.sin(t * 1.2) * 0.05;

      /* ─── IDLE: slow drone drift + parallax ─────────── */
      if (cur === SCENE_STATES.LANDING_IDLE) {
        aiGroup.visible = false;
        hSpot.intensity = 0;
        completeFired   = false;
        transStart      = null;

        const tgtX = BASE_POS.x + Math.sin(t * 0.18) * 26 + mouseX * 18;
        const tgtY = BASE_POS.y + Math.sin(t * 0.14) *  7 - mouseY *  9;
        const tgtZ = BASE_POS.z + Math.cos(t * 0.12) * 12;

        camera.position.lerp(new THREE.Vector3(tgtX, tgtY, tgtZ), dt * 1.2);
        camera.lookAt(BASE_LOOK.x + mouseX * 10, BASE_LOOK.y - mouseY * 6, BASE_LOOK.z);
      } else {
        /* ─── TRANSITION: cinematic sequence ─────────── */
        if (!transStart) { transStart = t; cinematicAudio.playLandslideSequence(); }
        const prog = Math.min((t - transStart) / TRANS_DUR, 1.0);

        if (prefersReducedMotion) {
          if (prog >= 0.6 && !completeFired) { completeFired = true; onTransitionComplete?.(); }
        } else {
          const isOfficer = cur === SCENE_STATES.OFFICER_TRANSITION;

          // Camera dive targets
          const [tX, tY, tZ, lX, lY, lZ] = isOfficer
            ? [SCARP_X + 2, 75, SCARP_Z + 130, SCARP_X, 52, SCARP_Z]
            : [20,          82, 165,             SCARP_X - 5, 48, SCARP_Z - 8];

          const spd = isOfficer ? 3.5 : 2.8;
          camera.position.x = THREE.MathUtils.lerp(camera.position.x, tX, dt * spd);
          camera.position.y = THREE.MathUtils.lerp(camera.position.y, tY, dt * spd);
          camera.position.z = THREE.MathUtils.lerp(camera.position.z, tZ, dt * spd);
          camera.lookAt(lX, lY, lZ);

          // AI overlays
          if (prog > 0.18) {
            aiGroup.visible = true;
            const fadeIn = Math.min(1, (prog - 0.18) / 0.25);
            wireMat.opacity  = fadeIn * (isOfficer ? 0.75 : 0.38);
            bMat.opacity     = fadeIn * 0.85 * (0.7 + Math.sin(t * 14) * 0.3);
            hSpot.intensity  = isOfficer ? fadeIn * 4.5 : 0;
          }

          // Landslide rocks
          if (prog > 0.22) {
            const slideT = t - transStart - 0.65;
            for (let r = 0; r < ROCK_N; r++) {
              const rock = rocks[r];
              if (!rock.active && slideT >= rock.releaseDelay) {
                rock.active = true;
                rock.vel.set((Math.random() - 0.5) * 14, -8 - Math.random() * 14, 20 + Math.random() * 26);
              }
              if (rock.active && !rock.settled) {
                rock.vel.y -= 40 * dt;
                rock.pos.addScaledVector(rock.vel, dt);
                rock.rot.x += rock.rotV.x * dt;
                rock.rot.y += rock.rotV.y * dt;
                const gy = terrainH(rock.pos.x, rock.pos.z);
                if (rock.pos.y <= gy + rock.sc) {
                  rock.pos.y = gy + rock.sc;
                  rock.vel.y = -rock.vel.y * 0.25;
                  rock.vel.x *= 0.60; rock.vel.z *= 0.60;
                  if (rock.vel.length() < 3.0) rock.settled = true;
                }
                dummy.position.copy(rock.pos);
                dummy.scale.setScalar(rock.sc);
                dummy.rotation.copy(rock.rot);
                dummy.updateMatrix();
                rockMesh.setMatrixAt(r, dummy.matrix);
              }
            }
            rockMesh.instanceMatrix.needsUpdate = true;

            // Dust plume
            for (const dp of dustParticles) {
              if (slideT >= dp.delay) {
                dp.active = true; dp.life += dt * 1.6;
                const p = Math.min(1, dp.life);
                const s = THREE.MathUtils.lerp(1, dp.maxS, p);
                dp.mesh.scale.setScalar(s);
                dp.mesh.position.y += dt * 5.5;
                dp.mesh.position.z += dt * 5.0;
                dp.mesh.material.opacity = Math.sin(p * Math.PI) * 0.50;
              }
            }
          }

          if (prog >= 0.95 && !completeFired) { completeFired = true; onTransitionComplete?.(); }
        }
      }

      renderer.render(scene, camera);
    }

    animate();

    /* ─── Resize ────────────────────────────────────── */
    const onResize = () => {
      W = container.clientWidth || window.innerWidth;
      H = container.clientHeight || window.innerHeight;
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      renderer.setSize(W, H);
    };
    window.addEventListener("resize", onResize);

    /* ─── Cleanup ─────────────────────────────────── */
    // Keep reference for river opacity animation
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("resize", onResize);
      try {
        GEO.dispose(); terrainMat.dispose();
        roadGeo.dispose(); roadMat.dispose();
        lineGeo.dispose(); lineMat.dispose();
        edgeGeo.dispose(); edgeMat.dispose();
        railGeo.dispose(); railMat.dispose();
        riverGeo.dispose(); riverMat.dispose();
        treeGeo.dispose(); treeMat.dispose();
        rainGeo.dispose(); rainMat.dispose();
        rockGeo.dispose(); rockMat.dispose();
        dustGeo.dispose();
        renderer.dispose();
        if (container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      } catch (_) {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDark, onTransitionComplete]);

  return (
    <div
      ref={mountRef}
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        pointerEvents: "none",
        zIndex: 0,
        overflow: "hidden",
        ...style,
      }}
    />
  );
});

export default MountainScene3D;
