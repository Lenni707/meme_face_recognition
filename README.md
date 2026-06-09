# Facial Monkey Test

A client-side static webcam app that shows your camera feed on the left and displays a reaction image on the right:

- `assets/thinking_monkey.jpeg` when your finger is detected near your mouth.
- `assets/speed_face.png` when your eyes are closed and your lips are closed in a near-circle shape.
- `assets/mogger.jpeg` when your index finger is detected on your lower chin, away from the mouth.
- A full-screen looping 67-piece `6`/`7` confetti shower when both hands move up and down asynchronously; faster motion increases the intensity.

When multiple image reactions are active, the app shows the one with the highest confidence. The right pane also shows one word for the current reaction: `none`, `monkey`, `speed`, `mogging`, or `confetti`.

This app runs entirely in your browser using MediaPipe's client-side vision tasks, meaning no backend or server-side processing is required.

## Run Locally

Due to browser security protocols, accessing webcams and loading WebAssembly modules requires a local server context (you cannot open the `index.html` file directly as a local file).

You can run a local server in the project directory using:

```bash
# Using Python
python3 -m http.server 8000
```

Then open http://localhost:8000 and allow webcam access.

## Deploying

Since this is a static site, you can deploy it for free using:
- **GitHub Pages** (Enable Pages under your repository settings)
- **Vercel** / **Netlify** / **Cloudflare Pages**

## Meme Asset

Current reaction image paths:

```text
assets/thinking_monkey.jpeg
assets/speed_face.png
assets/mogger.jpeg
```

Until an active reaction image exists, the app shows a placeholder in the right pane.
