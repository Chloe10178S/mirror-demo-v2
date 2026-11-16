// Interactive mirror (v4: Ray-Ban Wayfarer + Aviator)
//
// Method (MediaPipe 3D Face Transform): both accessories are placed once on MediaPipe's *canonical face model*
// (a reference face in centimetres). Every frame the face pose (rotation from the facial transformation matrix,
// scale + position fitted to your live face landmarks) carries them onto your face. Your live face mesh plus a
// skull volume are drawn into the depth buffer only, so the glasses' arms go behind your head.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { FilesetResolver, PoseLandmarker, FaceLandmarker } from './lib/mediapipe/vision_bundle.mjs';

// ---------------------------------------------------------------- config
const MODELS = {
  pose: ['./lib/mediapipe/pose_landmarker_full.task',
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'],
  face: ['./lib/mediapipe/face_landmarker.task',
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'],
};
const LOOKS = [
  { name: 'Ray-Ban Wayfarer', glasses: 'wayfarer' },
  { name: 'Aviator Sunglasses', glasses: 'aviator' },
];
const WAVE = { swings: 3, windowMs: 1600, cooldownMs: 1300 };

// ---- fitting on the canonical face (cm; x = your left, y up, z out of the face)
// Canonical refs: pupils ≈ (±3.18, 2.62, 3.8), nose bridge 168 = (0, 3.27, 5.24), brows y ≈ 5.1,
// temples 127/356 = (±7.74, 2.36, -2.0), forehead 151 = (0, 6.55, 5.03).
// How real frames sit (applied to both models at load):
//  • pantoscopic tilt — the front leans ~8° so the lower rims sit closer to the cheeks than the top bar;
const PANTO_DEG = 8;
const FITS = {
  wayfarer: {
    cmPerUnit: 1.30,                                 // Ray-Ban units -> cm: 10.96 u across the hinges ≈ 14.3 cm (50 mm Wayfarer)
    modelEye: new THREE.Vector3(0.14, 1.95, 4.25),    // between the lenses at pupil height, on the back of the lenses
    faceEye: new THREE.Vector3(0, 2.62, 5.75),        // bridge rests on the nose (168), lenses ~2 cm in front of the eyes
    hinge: { y: 2.8, z: 4.0 },                        // where the temples meet the front (model units)
    isTemple: (n) => /Splints/.test(n),
  },
  aviator: {
    cmPerUnit: 8.3,                                  // Aviator units -> cm: 0.70 u lens width ≈ 58 mm; 1.70 u ≈ 14.1 cm overall
    modelEye: new THREE.Vector3(0, -0.12, 0.25),      // lens centres are at y -0.15; pupils sit just above, as with real aviators
    faceEye: new THREE.Vector3(0, 2.62, 5.45),        // metal frames ride ~1 mm further out on their nose pads
    hinge: { y: -0.02, z: 0.26 },
    isTemple: (n) => /Temple/.test(n),
  },
};
// Your head behind the face (skull + a little hair), for occlusion: ellipsoid in canonical cm
const SKULL = { c: new THREE.Vector3(0, 5.2, -4.6), r: new THREE.Vector3(7.7, 11.2, 9.7) };
// face landmarks used to fit scale/position (stable, rigid-ish points)
const FIT_PAIRS = [[127, 356], [234, 454], [10, 152], [33, 263], [168, 152]];
const FIT_POINTS = [168, 6, 9, 33, 263, 133, 362, 127, 356, 10, 152, 234, 454];
// pose landmark indices (for the wave)
const P_ = { lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16 };

// ---------------------------------------------------------------- dom / ui
const $ = (id) => document.getElementById(id);
const video = $('cam'), stage = $('stage');
const ui = { item: $('item'), name: $('item-name'), dots: $('dots'), hint: $('hint'), meter: $('wave-meter'),
  fill: $('wave-fill'), status: $('status'), start: $('start') };
LOOKS.forEach(() => ui.dots.appendChild(document.createElement('span')));
const status = (m) => { ui.status.textContent = m || ''; ui.status.classList.toggle('hidden', !m); };
const TWEAK_DEFAULT = { glasses: 1, glassesY: 0 };
const tweak = Object.assign({ ...TWEAK_DEFAULT }, JSON.parse(localStorage.getItem('mirror3-tweak') || '{}'));
const saveTweak = () => localStorage.setItem('mirror3-tweak', JSON.stringify(tweak));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

// One Euro filter (Casiez et al.): steady when still, responsive when moving
class OneEuro {
  constructor(minCutoff, beta, dCutoff = 1) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0, t: 0 }); }
  static a(cut, dt) { const r = 2 * Math.PI * cut * dt; return r / (r + 1); }
  filter(v, t) {
    if (this.x === null) { this.x = v; this.t = t; return v; }
    const dt = Math.max(1e-3, t - this.t); this.t = t;
    this.dx += OneEuro.a(this.dCutoff, dt) * ((v - this.x) / dt - this.dx);
    this.x += OneEuro.a(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (v - this.x);
    return this.x;
  }
}
const filters = new Map();
function smooth(key, lms, t, minCut = 1.6, beta = 12) {
  if (!lms) { for (const k of [...filters.keys()]) if (k.startsWith(key + ':')) filters.delete(k); return null; }
  return lms.map((p, i) => {
    const k = `${key}:${i}`;
    let f = filters.get(k);
    if (!f) filters.set(k, f = [new OneEuro(minCut, beta), new OneEuro(minCut, beta), new OneEuro(minCut * 0.6, beta * 0.5)]);
    return { x: f[0].filter(p.x, t), y: f[1].filter(p.y, t), z: f[2].filter(p.z, t), visibility: p.visibility ?? 1 };
  });
}

// ---------------------------------------------------------------- renderer / scene
const renderer = new THREE.WebGLRenderer({ canvas: stage, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const hemi = new THREE.HemisphereLight(0xffffff, 0x3a3836, 1.0); scene.add(hemi);
const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(0.3, 1, 1.2); scene.add(key);
const camera = new THREE.OrthographicCamera(0, 1, 0, -1, -50000, 50000);   // screen pixels
// Everything face-related lives in un-mirrored camera space (MediaPipe's frame) and is flipped once, like the video.
const mirror = new THREE.Group(); mirror.scale.x = -1; scene.add(mirror);
const head = new THREE.Group(); head.matrixAutoUpdate = false; mirror.add(head);   // canonical face space -> px
let W = 1, H = 1;
function resize() {
  W = innerWidth; H = innerHeight; renderer.setSize(W, H, false);
  Object.assign(camera, { left: 0, right: W, top: 0, bottom: -H }); camera.updateProjectionMatrix();
  mirror.position.x = W;
}
addEventListener('resize', resize); resize();
function cover() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720, s = Math.max(W / vw, H / vh);
  return { dw: vw * s, dh: vh * s, ox: (W - vw * s) / 2, oy: (H - vh * s) / 2 };
}
// landmark -> un-mirrored px space (x right, y up, z toward you; MediaPipe z is scaled like x)
const camPx = (p, c, out = V()) => out.set(c.ox + p.x * c.dw, -(c.oy + p.y * c.dh), -p.z * c.dw);
const b64 = (s, T) => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new T(u.buffer); };

// ---------------------------------------------------------------- assets
let canon = null, faceOcc = null, skullOcc = null;
const frames = {};   // wayfarer / aviator groups (canonical face space)
const occMat = new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide });
const debugMat = new THREE.MeshBasicMaterial({ color: 0xd9a58a, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });

async function loadAssets() {
  canon = await (await fetch('./assets/face_model.json')).json();
  canon.vec = canon.v.map((p) => V(...p));
  canonBasis();

  // ---- Ray-Ban Wayfarer
  const G = await (await fetch('./assets/glasses.json')).json();
  const gm = {
    frame: new THREE.MeshPhysicalMaterial({ color: 0x0a0a0b, roughness: 0.16, clearcoat: 1, clearcoatRoughness: 0.08 }),
    lens: new THREE.MeshPhysicalMaterial({ color: 0x1c2a21, roughness: 0.03, metalness: 0.35, transparent: true, opacity: 0.9, envMapIntensity: 1.6 }),
    metal: new THREE.MeshStandardMaterial({ color: 0xd8d8d8, roughness: 0.15, metalness: 1 }),
  };
  const wi = new THREE.Group();
  for (const p of G) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(b64(p.pos, Float32Array), 3));
    geo.setIndex(new THREE.BufferAttribute(b64(p.idx, Uint32Array), 1)); geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, /Glass/.test(p.name) ? gm.lens : /Frame|Splints/.test(p.name) ? gm.frame : gm.metal);
    mesh.name = p.name; wi.add(mesh);
  }
  frames.wayfarer = wrap(wi, FITS.wayfarer);

  // ---- Aviator: keep the designer's look (black metal frame, dark smoked lenses). Its glass uses a
  // transmission material, which can't see the live video behind it, so the lenses become tinted transparent glass.
  const av = (await new GLTFLoader().loadAsync('./assets/aviator.glb')).scene;
  const avFrame = new THREE.MeshPhysicalMaterial({ color: 0x18181a, metalness: 1, roughness: 0.22, clearcoat: 0.4, envMapIntensity: 1.2, side: THREE.DoubleSide });
  const avLens = new THREE.MeshPhysicalMaterial({ color: 0x0e0f11, metalness: 0.45, roughness: 0.02, transparent: true, opacity: 0.84,
    envMapIntensity: 1.7, side: THREE.DoubleSide });
  av.traverse((c) => { if (c.isMesh) { const lens = /Lens/i.test(c.material.name); c.material = lens ? avLens : avFrame; c.frustumCulled = false; c.renderOrder = lens ? 2 : 0; } });
  frames.aviator = wrap(av, FITS.aviator);

  // ---- occluders: your live face mesh (vertices = landmarks each frame) + skull volume behind it
  const fg = new THREE.BufferGeometry();
  fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(468 * 3), 3));
  fg.setIndex(canon.f.flat());
  faceOcc = new THREE.Mesh(fg, occMat); faceOcc.renderOrder = -1; faceOcc.frustumCulled = false;
  mirror.add(faceOcc);
  skullOcc = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), occMat);
  skullOcc.position.copy(SKULL.c); skullOcc.scale.copy(SKULL.r); skullOcc.renderOrder = -1;
  head.add(skullOcc);
  applyFits();
}
// Bake the two adjustments above into the model's geometry (model units).
function shapeFrames(root, fit) {
  root.updateMatrixWorld(true);
  const meshes = [];
  root.traverse((c) => { if (c.isMesh) meshes.push(c); });
  const flat = new THREE.Group();
  for (const m of meshes) {          // flatten node transforms so everything is in one model space
    m.geometry = m.geometry.clone().applyMatrix4(m.matrixWorld);
    m.position.set(0, 0, 0); m.quaternion.identity(); m.scale.set(1, 1, 1);
    flat.add(m);
  }
  const u = fit.cmPerUnit, h = fit.hinge, tilt = THREE.MathUtils.degToRad(PANTO_DEG);
  const toModelZ = (zc) => (zc - fit.faceEye.z) / u + fit.modelEye.z;
  for (const m of meshes) {
    const p = m.geometry.attributes.position, temple = fit.isTemple(m.name);
    if (temple) {
      // arms left straight, as in the original model (outward splay removed)
    } else {
      // pantoscopic tilt of the front about the hinge line (bottom of the rims toward the cheeks)
      for (let i = 0; i < p.count; i++) {
        const dy = p.getY(i) - h.y, dz = p.getZ(i) - h.z;
        p.setY(i, h.y + dy * Math.cos(tilt) - dz * Math.sin(tilt));
        p.setZ(i, h.z + dy * Math.sin(tilt) + dz * Math.cos(tilt));
      }
    }
    p.needsUpdate = true; m.geometry.computeVertexNormals(); m.geometry.computeBoundingSphere();
  }
  return flat;
}
function wrap(obj, fit) {
  obj = shapeFrames(obj, fit);
  const outer = new THREE.Group(), inner = new THREE.Group();
  inner.add(obj); inner.position.copy(fit.modelEye).multiplyScalar(-1);
  outer.add(inner); outer.visible = false; head.add(outer);
  return outer;
}
// static placement of each accessory on the canonical face (re-run when you nudge sizes/heights)
function applyFits() {
  for (const [name, g] of Object.entries(frames)) {
    const fit = FITS[name];
    g.scale.setScalar(fit.cmPerUnit * tweak.glasses);
    g.position.copy(fit.faceEye).add(V(0, tweak.glassesY, 0));
  }
}

// ---------------------------------------------------------------- face pose
const _R = new THREE.Matrix4(), _q = new THREE.Quaternion(), poseQ = new THREE.Quaternion(), _x = V(), _y = V(), _z = V(), _a = V(), _b = V();
const pose = { k: 0, p: V(), has: false };
const _qL = new THREE.Quaternion(), _qM = new THREE.Quaternion();
function rotationFrom(F, M) {
  // axes from the face mesh itself (always available)
  const c = cover();
  camPx(F[356], c, _x).sub(camPx(F[127], c, _a));   // toward your left
  camPx(F[10], c, _y).sub(camPx(F[152], c, _a));    // up the face
  _x.normalize(); _z.crossVectors(_x, _y).normalize(); _y.crossVectors(_z, _x);
  _qL.setFromRotationMatrix(_R.makeBasis(_x, _y, _z));
  // the canonical face's own up axis is tilted vs. forehead->chin; express the mesh basis in canonical terms
  _qL.multiply(CANON_BASIS_INV);
  if (!M) return _q.copy(_qL);
  // MediaPipe facial transformation matrix (column-major, canonical -> camera space): steadier, keep its rotation —
  // but only if it agrees with the mesh (guards against any axis-convention surprise)
  _R.fromArray(M.data || M);
  const e = _R.elements;
  _x.set(e[0], e[1], e[2]).normalize(); _y.set(e[4], e[5], e[6]).normalize(); _z.crossVectors(_x, _y).normalize(); _y.crossVectors(_z, _x);
  _qM.setFromRotationMatrix(_R.makeBasis(_x, _y, _z));
  matrixTrusted = _qM.angleTo(_qL) < THREE.MathUtils.degToRad(25);
  return _q.copy(matrixTrusted ? _qM : _qL);
}
let matrixTrusted = false, CANON_BASIS_INV = new THREE.Quaternion();
function canonBasis() {      // same construction as above, on the canonical model itself
  const v = canon.vec, x = v[356].clone().sub(v[127]), y = v[10].clone().sub(v[152]);
  x.normalize(); const z = V().crossVectors(x, y).normalize(); y.crossVectors(z, x);
  CANON_BASIS_INV = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z)).invert();
}
function solveFace(F, M) {
  const c = cover();
  const q = rotationFrom(F, M);
  if (!pose.has) poseQ.copy(q); else poseQ.slerp(q, 0.6);
  // scale: your live landmark distances vs the canonical face seen at the same rotation (weak perspective)
  let num = 0, den = 0;
  for (const [i, j] of FIT_PAIRS) {
    camPx(F[i], c, _a).sub(camPx(F[j], c, _b));
    num += Math.hypot(_a.x, _a.y);
    _b.copy(canon.vec[i]).sub(canon.vec[j]).applyQuaternion(poseQ);
    den += Math.hypot(_b.x, _b.y);
  }
  const k = num / den;                                     // px per cm
  // position: least-squares translation over stable points (x, y and depth)
  const p = V();
  for (const i of FIT_POINTS) p.add(camPx(F[i], c, _a).sub(_b.copy(canon.vec[i]).applyQuaternion(poseQ).multiplyScalar(k)));
  p.multiplyScalar(1 / FIT_POINTS.length);
  if (!pose.has) { pose.k = k; pose.p.copy(p); } else { pose.k += (k - pose.k) * 0.6; pose.p.lerp(p, 0.7); }
  pose.has = true;
  head.matrix.compose(pose.p, poseQ, V(pose.k, pose.k, pose.k));
  head.matrixWorldNeedsUpdate = true;
  // live face mesh for occlusion
  const pa = faceOcc.geometry.attributes.position;
  for (let i = 0; i < 468; i++) { camPx(F[i], c, _a); pa.setXYZ(i, _a.x, _a.y, _a.z); }
  pa.needsUpdate = true; faceOcc.geometry.computeBoundingSphere();
}

// ---------------------------------------------------------------- light matching
const probe = Object.assign(document.createElement('canvas'), { width: 32, height: 18 });
const pctx = probe.getContext('2d', { willReadFrequently: true });
const light = { exp: 1.0, tint: new THREE.Color(1, 1, 1), kx: 0.3, ky: 1 };
function sampleLight() {
  if (video.readyState < 2) return;
  pctx.drawImage(video, 0, 0, 32, 18);
  const d = pctx.getImageData(0, 0, 32, 18).data;
  let r = 0, g = 0, bl = 0, left = 0, right = 0, top = 0, bot = 0;
  for (let y = 0; y < 18; y++) for (let x = 0; x < 32; x++) {
    const i = (y * 32 + x) * 4, l = (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255;
    r += d[i]; g += d[i + 1]; bl += d[i + 2];
    if (x < 16) left += l; else right += l; if (y < 9) top += l; else bot += l;
  }
  const n = 576, m = Math.max(r, g, bl) || 1, k = 0.25;
  light.exp += (clamp(0.45 + 1.1 * (left + right) / n, 0.6, 1.4) - light.exp) * k;
  light.tint.lerp(new THREE.Color(r / m, g / m, bl / m).lerp(new THREE.Color(1, 1, 1), 0.65), k);
  light.kx += (clamp(((left - right) / (n / 2)) * 6, -1.5, 1.5) - light.kx) * k;
  light.ky += (clamp(1 + ((top - bot) / (n / 2)) * 3, 0.2, 2) - light.ky) * k;
  renderer.toneMappingExposure = light.exp;
  hemi.color.copy(light.tint); key.color.copy(light.tint); key.position.set(light.kx, light.ky, 1.2);
}

// ---------------------------------------------------------------- looks + wave
let look = 0, pop = 1;
function setLook(i, animate = true) {
  look = (i + LOOKS.length) % LOOKS.length;
  ui.name.textContent = LOOKS[look].name;
  ui.name.classList.remove('pop'); void ui.name.offsetWidth; ui.name.classList.add('pop');
  [...ui.dots.children].forEach((d, k) => d.classList.toggle('on', k === look));
  if (animate) { pop = 0; const f = document.createElement('div'); f.className = 'flash'; document.body.appendChild(f); setTimeout(() => f.remove(), 500); }
}
setLook(0, false);
const hands = { l: { a: false }, r: { a: false } };
let lastWave = 0;
function waveHand(h, wr, el, th, now) {
  if (!((wr.visibility ?? 1) > 0.5 && (el.visibility ?? 1) > 0.3 && wr.y < el.y - th * 0.4)) { Object.assign(h, { a: false, sw: [] }); return 0; }
  const x = wr.x;
  if (!h.a) Object.assign(h, { a: true, dir: 0, ext: x, sw: [] });
  const swing = (d) => { h.dir = d; h.ext = x; h.sw.push(now); };
  if (h.dir === 0) { if (x - h.ext > th) swing(1); else if (h.ext - x > th) swing(-1); }
  else if (h.dir === 1) { if (x > h.ext) h.ext = x; else if (h.ext - x > th) swing(-1); }
  else { if (x < h.ext) h.ext = x; else if (x - h.ext > th) swing(1); }
  h.sw = h.sw.filter((t) => now - t < WAVE.windowMs);
  return h.sw.length;
}
function detectWave(P, now) {
  const c = cover(), px = (p) => ({ x: p.x * c.dw, y: p.y * c.dh, visibility: p.visibility });
  const S = P.map(px);
  const th = Math.max(22, Math.hypot(S[P_.lSh].x - S[P_.rSh].x, S[P_.lSh].y - S[P_.rSh].y) * 0.12);
  const best = Math.max(waveHand(hands.l, S[P_.lWr], S[P_.lEl], th, now), waveHand(hands.r, S[P_.rWr], S[P_.rEl], th, now));
  ui.meter.classList.toggle('hidden', !(hands.l.a || hands.r.a) || now - lastWave < WAVE.cooldownMs);
  ui.fill.style.width = `${Math.min(1, best / WAVE.swings) * 100}%`;
  if (best >= WAVE.swings && now - lastWave > WAVE.cooldownMs) { lastWave = now; hands.l = { a: false }; hands.r = { a: false }; setLook(look + 1); }
}

// ---------------------------------------------------------------- per frame
let lastFace = 0, showHead = false;
function frame(F, M, P, now) {
  const t = now / 1000;
  F = smooth('f', F, t); P = smooth('p', P, t, 1.2, 6);
  const L_ = LOOKS[look];
  if (F) {
    lastFace = now;
    solveFace(F, M);
    const popS = 0.9 + 0.1 * (1 - Math.pow(1 - pop, 3));
    for (const [name, g] of Object.entries(frames)) {
      g.visible = L_.glasses === name;
      g.scale.setScalar(FITS[name].cmPerUnit * tweak.glasses * popS);
    }
    faceOcc.visible = skullOcc.visible = true;
    ui.item.classList.remove('hidden'); ui.hint.classList.remove('hidden'); status('');
  } else if (now - lastFace > 500) {
    for (const g of Object.values(frames)) g.visible = false;
    faceOcc.visible = skullOcc.visible = false; pose.has = false;
    ui.hint.classList.add('hidden'); ui.item.classList.add('hidden'); status('Looking for you… step in front of the mirror');
  }
  faceOcc.material = skullOcc.material = showHead ? debugMat : occMat;
  if (P) detectWave(P, now);
}

let faceLm = null, poseLm = null, lastVideoTime = -1, injected = null, tick = 0;
function loop() {
  const now = performance.now();
  pop = Math.min(1, pop + 0.06);
  if (canon && frames.aviator) {
    if (injected) frame(injected.F, injected.M, injected.P, now);
    else if (faceLm && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const fr = faceLm.detectForVideo(video, now);
      const P = poseLm && tick % 2 === 0 ? poseLm.detectForVideo(video, now).landmarks?.[0] || null : null;
      frame(fr.faceLandmarks?.[0] || null, fr.facialTransformationMatrixes?.[0] || null, P, now);
    }
  }
  if (++tick % 8 === 0) sampleLight();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

async function pick([local, remote]) {
  try { const r = await fetch(local, { method: 'HEAD' }); if (r.ok) return local; } catch {}
  return remote;
}
async function make(Kind, fileset, path, extra) {
  for (const delegate of ['GPU', 'CPU']) {
    try { return await Kind.createFromOptions(fileset, { baseOptions: { modelAssetPath: path, delegate }, runningMode: 'VIDEO', ...extra }); }
    catch (e) { console.warn(`${delegate} delegate failed`, e); }
  }
  throw new Error('Could not start tracking');
}
async function start() {
  ui.start.classList.add('hidden');
  try {
    status('Turning on camera…');
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream; await video.play();
    status('Loading…'); await assetsReady;
    const fileset = await FilesetResolver.forVisionTasks('./lib/mediapipe/wasm');
    faceLm = await make(FaceLandmarker, fileset, await pick(MODELS.face), { numFaces: 1, outputFacialTransformationMatrixes: true });
    try { poseLm = await make(PoseLandmarker, fileset, await pick(MODELS.pose), { numPoses: 1 }); }
    catch (e) { console.warn('pose (wave) unavailable — use the arrow keys', e); }
    status('Looking for you… step in front of the mirror');
  } catch (e) {
    console.error(e);
    status(e.name === 'NotAllowedError' ? 'Camera permission was blocked — allow it in the address bar and reload.' : 'Could not start: ' + (e.message || e));
  }
}
$('start-btn').addEventListener('click', start);

addEventListener('keydown', (e) => {
  const k = e.key;
  if (k === 'ArrowRight' || k === ' ') setLook(look + 1);
  else if (k === 'ArrowLeft') setLook(look - 1);
  else if (k === 'f' || k === 'F') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  else if (k === 'h' || k === 'H') showHead = !showHead;
  else if (k === '[' || k === ']') tweak.glasses = clamp(tweak.glasses * (k === ']' ? 1.02 : 1 / 1.02), 0.7, 1.4);
  else if (k === '-' || k === '=') tweak.glassesY += k === '=' ? 0.15 : -0.15;
  else if (k === '0') Object.assign(tweak, TWEAK_DEFAULT);
  else return;
  saveTweak(); if (frames.aviator) applyFits();
});

// automated-render hook (no effect in normal use)
window.__mirror = { trusted: () => matrixTrusted, inject: (F, M, P) => { injected = F || P ? { F, M, P } : null; }, setLook, ready: () => !!(canon && frames.aviator), showHead: (v) => { showHead = v; } };

const assetsReady = loadAssets().catch((e) => { console.error(e); status('Could not load the 3D models: ' + e.message); });
requestAnimationFrame(loop);
