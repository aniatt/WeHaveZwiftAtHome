import * as THREE from "three";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const container = document.getElementById("canvas-container");
const instructions = document.getElementById("instructions");
const crosshair = document.getElementById("crosshair");
const loadBtn = document.getElementById("load-btn");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const hud = document.getElementById("hud");

// ---------------------------------------------------------------------------
// Renderer + Camera
// ---------------------------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 1.6, 0);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------------------
// GaussianSplats3D viewer (non-self-driven, no built-in controls)
// ---------------------------------------------------------------------------
let splatViewer = null;
let sceneLoaded = false;

function createSplatViewer() {
  return new GaussianSplats3D.Viewer({
    selfDrivenMode: false,
    renderer,
    camera,
    useBuiltInControls: false,
  });
}

// ---------------------------------------------------------------------------
// File loading (shared by button and drag-and-drop)
// ---------------------------------------------------------------------------
function detectFormat(filename) {
  const ext = filename.toLowerCase().split(".").pop();
  switch (ext) {
    case "ply":    return GaussianSplats3D.SceneFormat.Ply;
    case "splat":  return GaussianSplats3D.SceneFormat.Splat;
    case "ksplat": return GaussianSplats3D.SceneFormat.KSplat;
    default:       throw new Error(`Unsupported splat format: .${ext}`);
  }
}

async function loadFile(file) {
  const url = URL.createObjectURL(file);
  loadBtn.textContent = "Loading…";
  loadBtn.disabled = true;

  try {
    if (splatViewer) {
      splatViewer.dispose();
    }
    splatViewer = createSplatViewer();

    await splatViewer.addSplatScene(url, {
      showLoadingUI: false,
      splatAlphaRemovalThreshold: 5,
      format: detectFormat(file.name),
    });

    sceneLoaded = true;
    loadBtn.textContent = file.name;

    // Expose for debugging from the browser console: window.splatViewer / camera
    window.splatViewer = splatViewer;
    window.camera = camera;
    window.THREE = THREE;

    // (a) Convert COLMAP world frame (Y-down, Z-forward) to Three.js (Y-up,
    //     Z-back) by rotating the splat mesh 180 degrees around X.
    // (b) Compute robust scene extent from splat centers (mean +/- 2*stddev)
    //     so a few outlier splats don't blow up the bounding box.
    // (c) Position camera back from the centroid along the longest principal
    //     axis (typically the trajectory direction for a cycling clip).
    try {
      const sm = splatViewer.splatMesh;
      sm.rotation.x = Math.PI;
      sm.updateMatrix();
      sm.updateMatrixWorld(true);

      const count = sm.getSplatCount();
      const tmp = new THREE.Vector3();
      const min = new THREE.Vector3( Infinity,  Infinity,  Infinity);
      const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
      let mx = 0, my = 0, mz = 0;

      // Pass 1: raw bbox + centroid (in world space, after the X-rotation).
      for (let i = 0; i < count; i++) {
        sm.getSplatCenter(i, tmp);
        tmp.applyMatrix4(sm.matrixWorld);
        min.min(tmp); max.max(tmp);
        mx += tmp.x; my += tmp.y; mz += tmp.z;
      }
      mx /= count; my /= count; mz /= count;

      // Pass 2: per-axis variance (need centroid first).
      let vx = 0, vy = 0, vz = 0;
      for (let i = 0; i < count; i++) {
        sm.getSplatCenter(i, tmp);
        tmp.applyMatrix4(sm.matrixWorld);
        vx += (tmp.x - mx) ** 2;
        vy += (tmp.y - my) ** 2;
        vz += (tmp.z - mz) ** 2;
      }
      const sx = Math.sqrt(vx / count);
      const sy = Math.sqrt(vy / count);
      const sz = Math.sqrt(vz / count);

      const center  = new THREE.Vector3(mx, my, mz);
      const effSize = new THREE.Vector3(4 * sx, 4 * sy, 4 * sz);

      console.log("Splat scene loaded:", {
        count,
        raw_bbox_min: min.toArray().map((v) => v.toFixed(2)),
        raw_bbox_max: max.toArray().map((v) => v.toFixed(2)),
        centroid:     center.toArray().map((v) => v.toFixed(2)),
        stddev_xyz:   [sx, sy, sz].map((v) => v.toFixed(2)),
        effective_size: effSize.toArray().map((v) => v.toFixed(2)),
      });

      const axes = ["x", "y", "z"];
      const longest = axes.reduce(
        (best, a) => (effSize[a] > effSize[best] ? a : best),
        "x",
      );
      const camPos = center.clone();
      camPos[longest] -= effSize[longest] * 0.6;
      camera.position.copy(camPos);
      camera.lookAt(center);

      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
      yaw = Math.atan2(-fwd.x, -fwd.z);
      pitch = Math.asin(fwd.y);

      // Match the original ride pace: traverse the trajectory in roughly
      // the same time the cameras did. Without per-scene metadata we assume
      // ~30 sec of input footage; a 60-sec clip would result in 2x ride speed.
      // For arbitrary clip lengths the right fix is to bake duration into
      // the .ply filename or load it from a sidecar config.
      const ASSUMED_CLIP_SECONDS = 30;
      MOVE_SPEED = Math.max(0.5, effSize[longest] / ASSUMED_CLIP_SECONDS);
      console.log(`MOVE_SPEED set to ${MOVE_SPEED.toFixed(2)} units/s`);
    } catch (e) {
      console.warn("Could not auto-position camera:", e);
    }
  } catch (err) {
    console.error("Failed to load splat scene:", err);
    loadBtn.textContent = "Load .ply";
    loadBtn.disabled = false;
  } finally {
    URL.revokeObjectURL(url);
  }
}

loadBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) loadFile(file);
});

// Drag-and-drop
let dragCounter = 0;

document.addEventListener("dragenter", (e) => {
  e.preventDefault();
  dragCounter++;
  dropZone.classList.add("active");
});

document.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dragCounter--;
  if (dragCounter === 0) dropZone.classList.remove("active");
});

document.addEventListener("dragover", (e) => {
  e.preventDefault();
});

document.addEventListener("drop", (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropZone.classList.remove("active");
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

// ---------------------------------------------------------------------------
// First-person controls (PointerLock + WASD)
// ---------------------------------------------------------------------------
// Calibrated to scene size after a .ply loads — see auto-position block.
// COLMAP units are arbitrary, so a hardcoded "m/s" is meaningless across scenes.
let MOVE_SPEED = 3.0;
const SPRINT_MULTIPLIER = 2.5;
const MOUSE_SENSITIVITY = 0.002;
const PITCH_LIMIT = Math.PI / 2 - 0.05; // ~85 deg

let yaw = 0;
let pitch = 0;

const keys = {};

document.addEventListener("keydown", (e) => {
  keys[e.code] = true;
});
document.addEventListener("keyup", (e) => {
  keys[e.code] = false;
});

// Pointer lock — only allow after a scene is loaded
const canvas = renderer.domElement;

function requestLock() {
  if (!sceneLoaded) return;
  canvas.requestPointerLock();
}

instructions.addEventListener("click", requestLock);
canvas.addEventListener("click", requestLock);

document.addEventListener("pointerlockchange", () => {
  const locked = document.pointerLockElement === canvas;
  instructions.classList.toggle("hidden", locked);
  crosshair.classList.toggle("visible", locked);
});

document.addEventListener("mousemove", (e) => {
  if (document.pointerLockElement !== canvas) return;
  yaw -= e.movementX * MOUSE_SENSITIVITY;
  pitch -= e.movementY * MOUSE_SENSITIVITY;
  pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pitch));
});

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
const velocity = new THREE.Vector3();
const forward = new THREE.Vector3();
const right = new THREE.Vector3();
let prevTime = performance.now();

function update() {
  requestAnimationFrame(update);

  const now = performance.now();
  const dt = (now - prevTime) / 1000;
  prevTime = now;

  // Camera orientation from yaw/pitch
  const euler = new THREE.Euler(pitch, yaw, 0, "YXZ");
  camera.quaternion.setFromEuler(euler);

  // Movement relative to camera heading
  const speed = MOVE_SPEED * (keys["ShiftLeft"] || keys["ShiftRight"] ? SPRINT_MULTIPLIER : 1);

  forward.set(0, 0, -1).applyQuaternion(camera.quaternion);
  forward.y = 0;
  forward.normalize();

  right.set(1, 0, 0).applyQuaternion(camera.quaternion);
  right.y = 0;
  right.normalize();

  velocity.set(0, 0, 0);
  if (keys["KeyW"]) velocity.add(forward);
  if (keys["KeyS"]) velocity.sub(forward);
  if (keys["KeyD"]) velocity.add(right);
  if (keys["KeyA"]) velocity.sub(right);

  if (velocity.lengthSq() > 0) {
    velocity.normalize().multiplyScalar(speed * dt);
    camera.position.add(velocity);
  }

  // HUD
  const currentSpeed = velocity.length() / dt;
  if (currentSpeed > 0.01) {
    hud.textContent = `${currentSpeed.toFixed(1)} m/s`;
  } else {
    hud.textContent = "";
  }

  // Render splat scene
  if (splatViewer && sceneLoaded) {
    splatViewer.update();
    splatViewer.render();
  }
}

update();
