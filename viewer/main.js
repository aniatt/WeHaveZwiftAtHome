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
    });

    sceneLoaded = true;
    loadBtn.textContent = file.name;
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
const MOVE_SPEED = 3.0;
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
