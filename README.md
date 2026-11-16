# Tattarrattat — Interactive Mirror (v5: Ray-Ban Wayfarer + Aviator + Browline)

Run: double-click `start.command` → Chrome → **Start mirror** → allow the camera.
Wave a raised hand (3 swings) to switch looks: Ray-Ban Wayfarer → Aviator → Browline.

Keys: → next look · F fullscreen · H show the invisible head (debug) · `[` `]` glasses size · `-` `=` glasses height · 0 reset

## How it fits (MediaPipe 3D Face Transform)
- All three pairs are placed once on MediaPipe's **canonical face model** (a reference face in cm):
  - Ray-Ban: lens centres on the pupils, bridge resting on the nose (landmark 168), lenses ~2 cm in front of the eyes,
    frame ≈ 14.3 cm across the hinges (a 50 mm Wayfarer).
  - Aviator: lens centres just below the pupils (as real aviators sit), frame ≈ 14.1 cm overall (58 mm lenses),
    original black metal frame + dark smoked lenses (the file's glass-transmission material is replaced by tinted
    transparent glass so the live video shows through).
  - Browline (clubmaster style, from eyeglasses.rar → converted to assets/browline.glb): lens centres just under the
    pupils, under the brow bar; nose pads on the sides of the nose; ≈ 14.1 cm overall (51 mm lenses). Dark tortoise brow,
    gold metal rims/bridge/arms, brown tinted lenses.
- All frames get ~8° pantoscopic tilt (bottom rims toward the cheeks); the arms stay as in the original models.
- Each frame the face pose (rotation from FaceLandmarker's facial transformation matrix, cross-checked against the
  face mesh; scale + position least-squares-fitted to your live landmarks) carries them onto your head.
- Occlusion: your live 468-point face mesh + a skull volume are drawn into the depth buffer only, so the glasses' arms go
  behind your head and your face stays in front.
