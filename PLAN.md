# PLAN

## Completed
- Created the project plan for a containerized Python browser app.
- Chose a browser-based webcam flow so Docker does not need direct camera access.
- Added the FastAPI/WebSocket backend.
- Added the browser webcam UI with left camera pane, right meme pane, detection boxes, and confidence display.
- Added the `assets/` folder and switched the meme display to `assets/thinking_monkey.jpeg`.
- Added a second facial-expression trigger for `assets/speed_face.png`.
- Added `assets/speed_face.png`.
- Lowered the speed-face threshold to `0.57`.
- Changed the speed-face criteria to closed eyes plus oval/circle mouth shape.
- Updated the speed-face mouth criteria to closed lips in a small oval shape.
- Added a chin/jaw index-finger trigger for `assets/mogger.jpeg`.
- Tightened the chin trigger so mouth touches keep showing the monkey image.
- Added a one-word reaction label in the right pane.
- Added a 67-piece confetti shower for asynchronous two-hand vertical motion.
- Changed the confetti pieces to `6` and `7` characters.
- Made confetti intensity scale with hand motion speed.
- Moved the confetti shower over the whole screen and lengthened the animation.
- Added a hold window so the full confetti animation keeps playing and loops while motion persists.
- Changed image reaction selection to use the highest active confidence.
- Tightened speed face to require a closed mouth with a near-circle outer lip shape.
- Added Dockerfile and Docker Compose configuration.
- Verified `docker compose build` succeeds after Docker was started.
- Started the app container with `docker compose up -d`.
- Initialized the local git repository.
- Created and pushed the public GitHub repository: https://github.com/Lenni707/facial_monkey_test
- **Converted the app to a 100% static client-side application** by moving MediaPipe vision models to run in the browser using WebAssembly.
- **Removed the Python backend and Docker configuration** to make the app backend-free, lightweight, and deployable on static hosting platforms.
- **Renamed the local folder and GitHub repository** to `meme_face_recognition` (new Pages URL: https://lenni707.github.io/meme_face_recognition/).
- **Downloaded MediaPipe model task files locally** (`face_landmarker.task` and `hand_landmarker.task`) into the `assets` folder to ensure self-contained loading without external dependencies.
- **Tightened the monkey reaction trigger** to only activate when the finger is on or slightly above the lips, preventing activation if the finger moves downwards below the lower lip.
- **Fixed the chin/mogging trigger scale issues** by replacing screen-height based bounding boxes with dynamic face-width and mouth-width scaled limits.
- **Expanded jawline landmark points** (`chinIndexes` from 9 to 13 points) to support triggering the mogging reaction anywhere around the chin/jaw contour.
- **Upgraded confetti up-and-down motion checks** to require strict, high-speed vertical opposite movements of both hands, filtering out single-hand waving noise or vertical jitters.
- **Optimized confetti rendering performance** by switching to a dynamic particle spawner in JS, eliminating CSS-recalculation animation stutters.
- **Added the "silenced" gesture trigger** (`psst.png`) when holding the index finger vertically over the lips.
- **Added a 300ms gesture cooldown buffer** to prevent images and confidence indicators from flickering due to temporary tracking dropouts.
- **Simplified the default placeholder text** to the plain text: `"No motion detected."`.
- **Integrated the "67" gesture as a full meme reaction image** (loading `assets/67.jpeg` when active), renaming the active label from `"confetti"` to `"67"`.
- **Optimized the `speedFace` gesture thresholds** to trigger more reliably when lips are closed/touching but form a puckered oval/circle shape (using `normalizedScore` instead of `centeredScore` for roundness to avoid penalizing very round mouth shapes), and when eyes are moderately squinted.

## Future
- Add or replace custom reaction image files in the `assets/` folder if desired.
- Fine-tune specific gesture triggers' thresholds if needed.
