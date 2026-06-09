import { FilesetResolver, FaceLandmarker, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const confidenceLabel = document.getElementById("confidence");
const cameraStatus = document.getElementById("cameraStatus");
const memePanel = document.querySelector(".meme-panel");
const reactionLabel = document.getElementById("reactionLabel");
const confettiLayer = document.getElementById("confettiLayer");
const monkeyImage = document.getElementById("monkeyImage");
const speedFaceImage = document.getElementById("speedFaceImage");
const moggingImage = document.getElementById("moggingImage");

const overlayContext = overlay.getContext("2d");

const FINGER_MOUTH_THRESHOLD = 0.70;
const SPEED_FACE_THRESHOLD = 0.70;
const CHIN_FINGER_THRESHOLD = 0.72;
const CONFETTI_THRESHOLD = 0.42;

let faceLandmarker = null;
let handLandmarker = null;
let latestResult = null;
let streamReady = false;
let confettiHoldTimer = null;
let previousHandCenters = null;
let lastVideoTime = -1;

let monkeyImageAvailable = false;
let speedFaceImageAvailable = false;
let moggingImageAvailable = false;

async function start() {
  createConfetti();
  await refreshAssetStatus();
  await startCamera();
  await initMediaPipe();
  startDetectionLoop();
  window.addEventListener("resize", resizeOverlay);
}

async function refreshAssetStatus() {
  const checkAsset = async (url) => {
    try {
      const response = await fetch(url, { method: "HEAD" });
      return response.ok;
    } catch {
      return false;
    }
  };

  try {
    monkeyImageAvailable = await checkAsset("assets/thinking_monkey.jpeg");
    speedFaceImageAvailable = await checkAsset("assets/speed_face.png");
    moggingImageAvailable = await checkAsset("assets/mogger.jpeg");

    monkeyImage.classList.toggle("available", monkeyImageAvailable);
    speedFaceImage.classList.toggle("available", speedFaceImageAvailable);
    moggingImage.classList.toggle("available", moggingImageAvailable);
  } catch (error) {
    console.error("Error checking assets client-side:", error);
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 960, height: 720, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    streamReady = true;
    cameraStatus.textContent = "";
    resizeOverlay();
  } catch (error) {
    cameraStatus.textContent = `Camera unavailable: ${error.message}`;
    console.error(error);
  }
}

async function initMediaPipe() {
  cameraStatus.textContent = "Loading tracking models...";
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "assets/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false
    });

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "assets/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2
    });

    cameraStatus.textContent = "";
  } catch (error) {
    cameraStatus.textContent = `Model loading failed: ${error.message}`;
    console.error("Error loading MediaPipe:", error);
  }
}

function startDetectionLoop() {
  function loop() {
    if (streamReady && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      runDetection();
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

function runDetection() {
  if (!faceLandmarker || !handLandmarker) {
    return;
  }

  const timestamp = performance.now();
  const faceResults = faceLandmarker.detectForVideo(video, timestamp);
  const handResults = handLandmarker.detectForVideo(video, timestamp);

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    return;
  }

  const face = extractFace(faceResults, width, height);
  const hands = extractHands(handResults, width, height);

  const fingerMouthConfidence = computeFingerMouthConfidence(face, hands, width, height);
  const chinFingerConfidence = computeChinFingerConfidence(face, hands, width, height);
  const confettiConfidence = computeConfettiConfidence(hands, height);
  const speedFaceConfidence = computeSpeedFaceConfidence(face);

  const fingerMouthActive = fingerMouthConfidence >= FINGER_MOUTH_THRESHOLD;
  const speedFaceActive = speedFaceConfidence >= SPEED_FACE_THRESHOLD;
  const chinFingerActive = chinFingerConfidence >= CHIN_FINGER_THRESHOLD;
  const confettiActive = confettiConfidence >= CONFETTI_THRESHOLD;

  const activeImage = strongestActiveImage([
    ["monkey", fingerMouthConfidence, FINGER_MOUTH_THRESHOLD],
    ["speedFace", speedFaceConfidence, SPEED_FACE_THRESHOLD],
    ["mogging", chinFingerConfidence, CHIN_FINGER_THRESHOLD]
  ]);

  const maxConfidence = Math.max(
    fingerMouthConfidence,
    speedFaceConfidence,
    chinFingerConfidence,
    confettiConfidence
  );

  latestResult = {
    faceBox: face ? face.box : null,
    handBox: hands.length > 0 ? hands[0].box : null,
    handBoxes: hands.map(h => h.box),
    confidence: maxConfidence,
    fingerMouthConfidence,
    speedFaceConfidence,
    chinFingerConfidence,
    confettiConfidence,
    gestureActive: fingerMouthActive || speedFaceActive || chinFingerActive || confettiActive,
    activeImage,
    confettiActive,
    confettiIntensity: confettiConfidence,
    monkeyImageAvailable,
    speedFaceImageAvailable,
    moggingImageAvailable
  };

  updateUi(latestResult);
}

function extractFace(results, width, height) {
  if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
    return null;
  }

  const landmarks = results.faceLandmarks[0];
  const xs = landmarks.map(lm => lm.x * width);
  const ys = landmarks.map(lm => lm.y * height);

  const mouthIndexes = [13, 14, 61, 291];
  const mouthPoints = mouthIndexes.map(i => [landmarks[i].x * width, landmarks[i].y * height]);
  const mouthCenter = averagePoint(mouthPoints);

  const chinIndexes = [152, 148, 176, 149, 150, 377, 400, 378, 379];
  const chinPoints = chinIndexes.map(i => [landmarks[i].x * width, landmarks[i].y * height]);

  const metrics = extractFaceMetrics(landmarks, width, height);

  return {
    box: squareBox(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys), width, height),
    mouth: mouthCenter,
    chinPoints: chinPoints,
    metrics: metrics
  };
}

function extractFaceMetrics(landmarks, width, height) {
  let faceWidth = distancePx(landmarks[234], landmarks[454], width, height);
  faceWidth = Math.max(faceWidth, 1.0);

  const leftEyeOpen = distancePx(landmarks[159], landmarks[145], width, height);
  const rightEyeOpen = distancePx(landmarks[386], landmarks[374], width, height);
  const mouthWidth = distancePx(landmarks[61], landmarks[291], width, height);
  const mouthOpen = distancePx(landmarks[13], landmarks[14], width, height);
  const mouthOuterHeight = distancePx(landmarks[0], landmarks[17], width, height);
  const mouthOuterRoundness = mouthOuterHeight / Math.max(mouthWidth, 1.0);

  return {
    eye_open: ((leftEyeOpen + rightEyeOpen) / 2) / faceWidth,
    mouth_open: mouthOpen / faceWidth,
    mouth_outer_roundness: mouthOuterRoundness
  };
}

function extractHands(results, width, height) {
  if (!results.landmarks || results.landmarks.length === 0) {
    return [];
  }

  const hands = [];
  for (const landmarks of results.landmarks) {
    const xs = landmarks.map(lm => lm.x * width);
    const ys = landmarks.map(lm => lm.y * height);

    const indexTip = landmarks[8];
    const fingertip = [indexTip.x * width, indexTip.y * height];
    const center = averagePoint(landmarks.map(lm => [lm.x * width, lm.y * height]));

    hands.push({
      box: squareBox(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys), width, height),
      fingertip: fingertip,
      center: center
    });
  }
  return hands;
}

function computeFingerMouthConfidence(face, hands, width, height) {
  if (!face || hands.length === 0) {
    return 0.0;
  }

  const [mouthX, mouthY] = face.mouth;
  const frameScale = Math.hypot(width, height);
  const closeDistance = frameScale * 0.035;
  const farDistance = frameScale * 0.16;
  let confidence = 0.0;

  for (const hand of hands) {
    const [fingerX, fingerY] = hand.fingertip;
    const distance = Math.hypot(fingerX - mouthX, fingerY - mouthY);
    const handConfidence = 1.0 - ((distance - closeDistance) / (farDistance - closeDistance));
    confidence = Math.max(confidence, handConfidence);
  }

  return Number(Math.max(0.0, Math.min(1.0, confidence)).toFixed(2));
}

function computeChinFingerConfidence(face, hands, width, height) {
  if (!face || hands.length === 0) {
    return 0.0;
  }

  const [mouthX, mouthY] = face.mouth;
  const frameScale = Math.hypot(width, height);
  const closeDistance = frameScale * 0.03;
  const farDistance = frameScale * 0.10;
  let confidence = 0.0;

  for (const hand of hands) {
    const finger = hand.fingertip;
    const mouthDistance = Math.hypot(finger[0] - mouthX, finger[1] - mouthY);
    if (mouthDistance < frameScale * 0.10) {
      continue;
    }

    const distance = Math.min(...face.chinPoints.map(point => Math.hypot(finger[0] - point[0], finger[1] - point[1])));
    const handConfidence = 1.0 - ((distance - closeDistance) / (farDistance - closeDistance));
    confidence = Math.max(confidence, handConfidence);
  }

  return Number(Math.max(0.0, Math.min(1.0, confidence)).toFixed(2));
}

function computeSpeedFaceConfidence(face) {
  if (!face) {
    return 0.0;
  }

  const metrics = face.metrics;
  const eyeScore = normalizedInverse(metrics.eye_open, 0.075, 0.16);
  const mouthClosedScore = normalizedInverse(metrics.mouth_open, 0.012, 0.045);
  const mouthCircleScore = centeredScore(metrics.mouth_outer_roundness, 0.72, 0.22);
  const confidence = Math.min(eyeScore, mouthClosedScore, mouthCircleScore);

  return Number(Math.max(0.0, Math.min(1.0, confidence)).toFixed(2));
}

function computeConfettiConfidence(hands, height) {
  const centers = hands.map(hand => hand.center).sort((a, b) => a[0] - b[0]);
  if (centers.length < 2) {
    previousHandCenters = centers;
    return 0.0;
  }

  if (!previousHandCenters || previousHandCenters.length < 2) {
    previousHandCenters = centers;
    return 0.0;
  }

  const leftDelta = centers[0][1] - previousHandCenters[0][1];
  const rightDelta = centers[1][1] - previousHandCenters[1][1];
  previousHandCenters = centers;

  const movingOpposite = leftDelta * rightDelta < 0;
  const normalizedSpeed = (Math.abs(leftDelta) + Math.abs(rightDelta)) / Math.max(height, 1);
  if (!movingOpposite) {
    return 0.0;
  }

  const confidence = normalizedScore(normalizedSpeed, 0.025, 0.14);
  return Number(Math.max(0.0, Math.min(1.0, confidence)).toFixed(2));
}

function normalizedInverse(value, low, high) {
  if (high <= low) return 0.0;
  return Math.max(0.0, Math.min(1.0, 1.0 - ((value - low) / (high - low))));
}

function normalizedScore(value, low, high) {
  if (high <= low) return 0.0;
  return Math.max(0.0, Math.min(1.0, (value - low) / (high - low)));
}

function centeredScore(value, target, tolerance) {
  if (tolerance <= 0) return 0.0;
  return Math.max(0.0, Math.min(1.0, 1.0 - (Math.abs(value - target) / tolerance)));
}

function strongestActiveImage(candidates) {
  const activeCandidates = candidates.filter(
    ([name, confidence, threshold]) => confidence >= threshold
  );
  if (activeCandidates.length === 0) {
    return null;
  }
  let strongest = activeCandidates[0];
  for (let i = 1; i < activeCandidates.length; i++) {
    if (activeCandidates[i][1] > strongest[1]) {
      strongest = activeCandidates[i];
    }
  }
  return strongest[0];
}

function averagePoint(points) {
  const sumX = points.reduce((sum, p) => sum + p[0], 0);
  const sumY = points.reduce((sum, p) => sum + p[1], 0);
  return [sumX / points.length, sumY / points.length];
}

function squareBox(minX, minY, maxX, maxY, frameWidth, frameHeight) {
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  let side = Math.max(maxX - minX, maxY - minY) * 1.15;

  let x = Math.max(0, centerX - side / 2);
  let y = Math.max(0, centerY - side / 2);
  side = Math.min(side, frameWidth - x, frameHeight - y);

  return {
    x: Math.round(x),
    y: Math.round(y),
    size: Math.round(Math.max(0, side))
  };
}

function distancePx(pointA, pointB, width, height) {
  return Math.hypot((pointA.x - pointB.x) * width, (pointA.y - pointB.y) * height);
}

function updateUi(result) {
  confidenceLabel.textContent = Number(result.confidence || 0).toFixed(2);
  memePanel.dataset.activeImage = result.activeImage || "";
  confettiLayer.style.setProperty("--confetti-intensity", String(Number(result.confettiIntensity || 0).toFixed(2)));
  updateConfetti(result.confettiActive);
  reactionLabel.textContent = reactionWord(result.activeImage, result.confettiActive);
  monkeyImage.classList.toggle("available", Boolean(result.monkeyImageAvailable));
  speedFaceImage.classList.toggle("available", Boolean(result.speedFaceImageAvailable));
  moggingImage.classList.toggle("available", Boolean(result.moggingImageAvailable));
  drawOverlay(result);
}

function updateConfetti(active) {
  if (active) {
    document.body.dataset.confetti = "active";
    window.clearTimeout(confettiHoldTimer);
    confettiHoldTimer = window.setTimeout(() => {
      document.body.dataset.confetti = "";
    }, 2200);
  }
}

function createConfetti() {
  const colors = ["#ff4f81", "#ffd166", "#06d6a0", "#4cc9f0", "#f72585", "#f8f9fa"];
  for (let index = 0; index < 67; index += 1) {
    const piece = document.createElement("span");
    piece.className = "confetti-piece";
    piece.textContent = index % 2 === 0 ? "6" : "7";
    piece.style.setProperty("--x", `${(index * 37) % 100}%`);
    piece.style.setProperty("--delay", `${-((index * 0.137) % 2.4)}s`);
    piece.style.setProperty("--drift", `${((index % 9) - 4) * 9}px`);
    piece.style.setProperty("--spin", `${180 + (index % 7) * 45}deg`);
    piece.style.color = colors[index % colors.length];
    confettiLayer.appendChild(piece);
  }
}

function reactionWord(activeImage, confettiActive) {
  if (activeImage === "speedFace") {
    return "speed";
  }
  if (activeImage === "monkey") {
    return "monkey";
  }
  if (activeImage === "mogging") {
    return "mogging";
  }
  if (confettiActive) {
    return "confetti";
  }
  return "none";
}

function resizeOverlay() {
  const bounds = video.getBoundingClientRect();
  const deviceScale = window.devicePixelRatio || 1;
  overlay.width = Math.round(bounds.width * deviceScale);
  overlay.height = Math.round(bounds.height * deviceScale);
  overlayContext.setTransform(deviceScale, 0, 0, deviceScale, 0, 0);
  if (latestResult) {
    drawOverlay(latestResult);
  }
}

function drawOverlay(result) {
  const bounds = video.getBoundingClientRect();
  overlayContext.clearRect(0, 0, bounds.width, bounds.height);

  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (!sourceWidth || !sourceHeight) {
    return;
  }

  drawBox(result.faceBox, "#43d67d", "Head", sourceWidth, sourceHeight, bounds);
  const handBoxes = result.handBoxes || (result.handBox ? [result.handBox] : []);
  handBoxes.forEach((box, index) => {
    drawBox(box, "#ffcc33", index === 0 ? "Hand" : "Hand", sourceWidth, sourceHeight, bounds);
  });
}

function drawBox(box, color, label, sourceWidth, sourceHeight, bounds) {
  if (!box) {
    return;
  }

  const scale = Math.max(bounds.width / sourceWidth, bounds.height / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const offsetX = (bounds.width - renderedWidth) / 2;
  const offsetY = (bounds.height - renderedHeight) / 2;

  const x = offsetX + box.x * scale;
  const y = offsetY + box.y * scale;
  const size = box.size * scale;

  overlayContext.strokeStyle = color;
  overlayContext.lineWidth = 3;
  overlayContext.strokeRect(x, y, size, size);
  overlayContext.fillStyle = color;
  overlayContext.font = "700 14px system-ui";
  overlayContext.fillText(label, x + 6, Math.max(18, y - 8));
}

start();
