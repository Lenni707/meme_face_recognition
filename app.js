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
const silencedImage = document.getElementById("silencedImage");

const overlayContext = overlay.getContext("2d");

const FINGER_MOUTH_THRESHOLD = 0.70;
const SPEED_FACE_THRESHOLD = 0.70;
const CHIN_FINGER_THRESHOLD = 0.72;
const CONFETTI_THRESHOLD = 0.42;
const SILENCED_THRESHOLD = 0.70;

let faceLandmarker = null;
let handLandmarker = null;
let latestResult = null;
let streamReady = false;
let previousHandCenters = null;
let lastVideoTime = -1;

let monkeyImageAvailable = false;
let speedFaceImageAvailable = false;
let moggingImageAvailable = false;
let silencedImageAvailable = false;

// Confetti state variables
let lastConfettiTriggerTime = 0;
let maxConfettiIntensityInSession = 0;
const CONFETTI_MIN_DURATION = 2000; // 2 seconds minimum play window

// Cooldown variables to prevent image flickering
const GESTURE_COOLDOWN = 300; // 300ms buffer window
let lastActiveTimes = {
  monkey: 0,
  speedFace: 0,
  mogging: 0,
  silenced: 0
};
let activeSessionConfidences = {
  monkey: 0,
  speedFace: 0,
  mogging: 0,
  silenced: 0
};

async function start() {
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
    silencedImageAvailable = await checkAsset("assets/psst.png");

    monkeyImage.classList.toggle("available", monkeyImageAvailable);
    speedFaceImage.classList.toggle("available", speedFaceImageAvailable);
    moggingImage.classList.toggle("available", moggingImageAvailable);
    silencedImage.classList.toggle("available", silencedImageAvailable);
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
  const silencedConfidence = computeSilencedConfidence(face, hands, width, height);
  const confettiScores = computeConfettiConfidence(hands, height);
  const speedFaceConfidence = computeSpeedFaceConfidence(face);

  const now = performance.now();

  // Helper to check if a gesture is active, incorporating the 300ms cooldown
  const checkGestureActiveState = (name, rawConfidence, threshold) => {
    if (rawConfidence >= threshold) {
      lastActiveTimes[name] = now;
      activeSessionConfidences[name] = Math.max(activeSessionConfidences[name], rawConfidence);
      return { active: true, confidence: activeSessionConfidences[name] };
    } else if (now - lastActiveTimes[name] < GESTURE_COOLDOWN) {
      // Within cooldown: keep it active, but decay the confidence slightly to show it's fading
      const progress = (now - lastActiveTimes[name]) / GESTURE_COOLDOWN;
      const confidence = activeSessionConfidences[name] * (1.0 - progress * 0.3); // decay up to 30%
      return { active: true, confidence: confidence };
    } else {
      activeSessionConfidences[name] = 0.0;
      return { active: false, confidence: 0.0 };
    }
  };

  const monkeyState = checkGestureActiveState("monkey", fingerMouthConfidence, FINGER_MOUTH_THRESHOLD);
  const speedFaceState = checkGestureActiveState("speedFace", speedFaceConfidence, SPEED_FACE_THRESHOLD);
  const chinFingerState = checkGestureActiveState("mogging", chinFingerConfidence, CHIN_FINGER_THRESHOLD);
  const silencedState = checkGestureActiveState("silenced", silencedConfidence, SILENCED_THRESHOLD);

  // Apply stateful lock to keep confetti active for a minimum of 2 seconds once triggered
  let confettiConfidence = 0.0;
  let confettiActive = false;
  const timeSinceTrigger = now - lastConfettiTriggerTime;

  if (timeSinceTrigger < CONFETTI_MIN_DURATION) {
    // If already playing, keep it alive if user maintains the lighter persistence threshold
    if (confettiScores.persist >= CONFETTI_THRESHOLD) {
      lastConfettiTriggerTime = now;
      maxConfettiIntensityInSession = Math.max(maxConfettiIntensityInSession, confettiScores.persist, 0.85);
    }
    confettiActive = true;
    const progress = timeSinceTrigger / CONFETTI_MIN_DURATION;
    confettiConfidence = maxConfettiIntensityInSession * (1.0 - progress * 0.7);
  } else {
    // If not playing, require the higher trigger threshold to start
    if (confettiScores.trigger >= CONFETTI_THRESHOLD) {
      lastConfettiTriggerTime = now;
      maxConfettiIntensityInSession = Math.max(confettiScores.trigger, 0.85);
      confettiActive = true;
      confettiConfidence = maxConfettiIntensityInSession;
    } else {
      maxConfettiIntensityInSession = 0.0;
    }
  }

  const activeCandidates = [
    { name: "monkey", state: monkeyState },
    { name: "speedFace", state: speedFaceState },
    { name: "mogging", state: chinFingerState },
    { name: "silenced", state: silencedState }
  ].filter(c => c.state.active);

  let activeImage = null;
  if (activeCandidates.length > 0) {
    // Select candidate with the highest smoothed confidence
    let strongest = activeCandidates[0];
    for (let i = 1; i < activeCandidates.length; i++) {
      if (activeCandidates[i].state.confidence > strongest.state.confidence) {
        strongest = activeCandidates[i];
      }
    }
    activeImage = strongest.name;
  }

  const maxConfidence = Math.max(
    monkeyState.confidence,
    speedFaceState.confidence,
    chinFingerState.confidence,
    silencedState.confidence,
    confettiConfidence
  );

  latestResult = {
    faceBox: face ? face.box : null,
    handBox: hands.length > 0 ? hands[0].box : null,
    handBoxes: hands.map(h => h.box),
    confidence: maxConfidence,
    fingerMouthConfidence: monkeyState.confidence,
    speedFaceConfidence: speedFaceState.confidence,
    chinFingerConfidence: chinFingerState.confidence,
    silencedConfidence: silencedState.confidence,
    confettiConfidence,
    gestureActive: monkeyState.active || speedFaceState.active || chinFingerState.active || silencedState.active || confettiActive,
    activeImage,
    confettiActive,
    confettiIntensity: confettiConfidence,
    monkeyImageAvailable,
    speedFaceImageAvailable,
    moggingImageAvailable,
    silencedImageAvailable
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

  const chinIndexes = [152, 148, 176, 149, 150, 136, 172, 377, 400, 378, 379, 365, 397];
  const chinPoints = chinIndexes.map(i => [landmarks[i].x * width, landmarks[i].y * height]);

  const metrics = extractFaceMetrics(landmarks, width, height);
  const mouthWidth = distancePx(landmarks[61], landmarks[291], width, height);
  const faceWidth = distancePx(landmarks[234], landmarks[454], width, height);

  return {
    box: squareBox(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys), width, height),
    mouth: mouthCenter,
    chinPoints: chinPoints,
    metrics: metrics,
    mouthWidth: mouthWidth,
    faceWidth: faceWidth
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
    const indexKnuckle = [landmarks[6].x * width, landmarks[6].y * height];

    hands.push({
      box: squareBox(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys), width, height),
      fingertip: fingertip,
      center: center,
      indexKnuckle: indexKnuckle
    });
  }
  return hands;
}

function computeFingerMouthConfidence(face, hands, width, height) {
  if (!face || hands.length === 0) {
    return 0.0;
  }

  const [mouthX, mouthY] = face.mouth;
  const mouthWidth = face.mouthWidth;
  let confidence = 0.0;

  for (const hand of hands) {
    const [fingerX, fingerY] = hand.fingertip;
    const dx = Math.abs(fingerX - mouthX);
    const dy = fingerY - mouthY; // positive if below mouth center

    // Horizontal limit: must be close to mouth center
    const maxDx = mouthWidth * 0.9;
    
    // Vertical limits:
    // - Upwards: up to mouthWidth * 0.45 (slightly over upper lip)
    // - Downwards: up to mouthWidth * 0.20 (prevent triggering downwards of the lower lip)
    const maxDyUp = mouthWidth * 0.45;
    const maxDyDown = mouthWidth * 0.20;

    let isWithinVerticalBounds = false;
    let limitY = 0;
    if (dy < 0) { // Finger is above mouth center
      isWithinVerticalBounds = Math.abs(dy) <= maxDyUp;
      limitY = maxDyUp;
    } else { // Finger is below mouth center
      isWithinVerticalBounds = dy <= maxDyDown;
      limitY = maxDyDown;
    }

    if (dx <= maxDx && isWithinVerticalBounds) {
      const dist = Math.hypot(dx, dy);
      const closeDistance = mouthWidth * 0.15;
      const farDistance = Math.hypot(maxDx, limitY);

      const handConfidence = 1.0 - ((dist - closeDistance) / (farDistance - closeDistance));
      confidence = Math.max(confidence, handConfidence);
    }
  }

  return Number(Math.max(0.0, Math.min(1.0, confidence)).toFixed(2));
}

function computeChinFingerConfidence(face, hands, width, height) {
  if (!face || hands.length === 0) {
    return 0.0;
  }

  const [mouthX, mouthY] = face.mouth;
  const mouthWidth = face.mouthWidth;
  const faceWidth = face.faceWidth;

  const closeDistance = faceWidth * 0.07;
  const farDistance = faceWidth * 0.22;
  let confidence = 0.0;

  for (const hand of hands) {
    const finger = hand.fingertip;
    const mouthDistance = Math.hypot(finger[0] - mouthX, finger[1] - mouthY);
    
    // Ensure the finger is vertically below the mouth and not on the lips
    if (finger[1] < mouthY + mouthWidth * 0.20) {
      continue;
    }

    const distance = Math.min(...face.chinPoints.map(point => Math.hypot(finger[0] - point[0], finger[1] - point[1])));
    const handConfidence = 1.0 - ((distance - closeDistance) / (farDistance - closeDistance));
    confidence = Math.max(confidence, handConfidence);
  }

  return Number(Math.max(0.0, Math.min(1.0, confidence)).toFixed(2));
}

function computeSilencedConfidence(face, hands, width, height) {
  if (!face || hands.length === 0) {
    return 0.0;
  }

  const [mouthX, mouthY] = face.mouth;
  const mouthWidth = face.mouthWidth;
  let confidence = 0.0;

  for (const hand of hands) {
    if (!hand.indexKnuckle) continue;
    const [fingerX, fingerY] = hand.fingertip;
    const [knuckleX, knuckleY] = hand.indexKnuckle;

    const dxTip = Math.abs(fingerX - mouthX);

    // Is the index finger held vertically? (X of tip and joint are close)
    const isVertical = Math.abs(fingerX - knuckleX) < mouthWidth * 0.40;
    
    // Is the fingertip pointing up? (Y of tip is less than Y of joint)
    const isPointingUp = fingerY < knuckleY;
    
    // Is the finger horizontally aligned with the mouth center?
    const isCentered = dxTip < mouthWidth * 0.55;
    
    // Does the finger overlap the mouth center vertically?
    // (Fingertip is above mouth center, knuckle joint is below mouth center)
    const overlapsMouthVertically = fingerY < mouthY + mouthWidth * 0.1 && knuckleY > mouthY - mouthWidth * 0.1;

    if (isVertical && isPointingUp && isCentered && overlapsMouthVertically) {
      // Confidence is higher the closer the finger is to the exact horizontal center of the mouth
      const handConfidence = 1.0 - (dxTip / (mouthWidth * 0.55));
      confidence = Math.max(confidence, handConfidence);
    }
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
    return { trigger: 0.0, persist: 0.0 };
  }

  if (!previousHandCenters || previousHandCenters.length < 2) {
    previousHandCenters = centers;
    return { trigger: 0.0, persist: 0.0 };
  }

  const leftDelta = centers[0][1] - previousHandCenters[0][1];
  const rightDelta = centers[1][1] - previousHandCenters[1][1];
  previousHandCenters = centers;

  // Must be moving in opposite vertical directions (asynchronous up/down)
  const movingOpposite = leftDelta * rightDelta < 0;
  if (!movingOpposite) {
    return { trigger: 0.0, persist: 0.0 };
  }

  const leftSpeed = Math.abs(leftDelta);
  const rightSpeed = Math.abs(rightDelta);
  const totalSpeed = leftSpeed + rightSpeed;

  // Ensure both hands are actively moving to filter out single-hand noise or jitters
  // Trigger speed requires each hand to move at least 1.8% of height per frame
  // Persist speed is lower to allow continuous light waving
  const minHandSpeedTrigger = height * 0.018;
  const minHandSpeedPersist = height * 0.007;

  let triggerConfidence = 0.0;
  let persistConfidence = 0.0;

  if (leftSpeed >= minHandSpeedTrigger && rightSpeed >= minHandSpeedTrigger) {
    triggerConfidence = normalizedScore(totalSpeed / height, 0.035, 0.11);
  }

  if (leftSpeed >= minHandSpeedPersist && rightSpeed >= minHandSpeedPersist) {
    persistConfidence = normalizedScore(totalSpeed / height, 0.012, 0.07);
  }

  return {
    trigger: Number(Math.max(0.0, Math.min(1.0, triggerConfidence)).toFixed(2)),
    persist: Number(Math.max(0.0, Math.min(1.0, persistConfidence)).toFixed(2))
  };
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
  updateConfetti(result.confettiActive, result.confettiIntensity);
  reactionLabel.textContent = reactionWord(result.activeImage, result.confettiActive);
  monkeyImage.classList.toggle("available", Boolean(result.monkeyImageAvailable));
  speedFaceImage.classList.toggle("available", Boolean(result.speedFaceImageAvailable));
  moggingImage.classList.toggle("available", Boolean(result.moggingImageAvailable));
  silencedImage.classList.toggle("available", Boolean(result.silencedImageAvailable));
  drawOverlay(result);
}

function updateConfetti(active, intensity) {
  if (active) {
    document.body.dataset.confetti = "active";
    // Spawn particles dynamically: faster speed -> more particles (up to 3 particles per frame at peak intensity)
    const spawnChance = intensity * 1.5; 
    const numToSpawn = Math.floor(spawnChance) + (Math.random() < (spawnChance % 1) ? 1 : 0);
    for (let i = 0; i < numToSpawn; i++) {
      spawnSingleConfetti(intensity);
    }
  } else {
    document.body.dataset.confetti = "";
  }
}

function spawnSingleConfetti(intensity) {
  const piece = document.createElement("span");
  piece.className = "confetti-piece-dynamic";
  piece.textContent = Math.random() < 0.5 ? "6" : "7";

  const x = Math.random() * 100;
  const drift = (Math.random() - 0.5) * 120;
  const spin = 180 + Math.random() * 360;
  
  // Scales particle size with current hand motion speed
  const size = 18 + Math.random() * 8 + intensity * 16; 

  piece.style.setProperty("--x", `${x}%`);
  piece.style.setProperty("--drift", `${drift}px`);
  piece.style.setProperty("--spin", `${spin}deg`);
  piece.style.fontSize = `${size}px`;

  const colors = ["#ff4f81", "#ffd166", "#06d6a0", "#4cc9f0", "#f72585", "#f8f9fa"];
  piece.style.color = colors[Math.floor(Math.random() * colors.length)];

  // Faster hand speed makes the particles fall faster
  const baseDuration = 2.5 + Math.random() * 1.5; 
  const duration = baseDuration / (0.4 + intensity * 0.8);
  piece.style.animation = `confetti-fall ${duration}s ease-in forwards`;

  // Clean up particle from DOM when it hits the bottom
  piece.addEventListener("animationend", () => {
    piece.remove();
  });

  confettiLayer.appendChild(piece);
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
  if (activeImage === "silenced") {
    return "silenced";
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
