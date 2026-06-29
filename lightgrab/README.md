# PSX LightGrab

Grab the lighting and environment from PSX game scenes and use it as a
Blender world / HDRI. You navigate the game in **DuckStation** (savestates,
rewind, frame advance = your timeline), take screenshots at the angles you
want, and this tool turns those LDR frames into a pseudo-HDR equirectangular
environment map plus a ready-to-run Blender setup script.

Works on captured frames only — no ROM/BIOS parsing — so it's actually
emulator-agnostic (PCSX2, mGBA, Dolphin screenshots all work the same way).

## Honest caveat

PSX hardware never rendered HDR. The dynamic range in the output is
**fabricated** (inverse tonemapping + a boost on the brightest blob so Blender
gets a key light). It lights scenes convincingly and nails the palette, but
it's not physically measured luminance.

## Install (v3)

```
pip install numpy opencv-python Pillow
python3 psx_lightgrab.py --doctor    # diagnoses missing pieces with exact fixes
python3 psx_lightgrab.py             # GUI
```

Common Linux gotchas the doctor will catch: `sudo apt install python3-tk
python3-pil.imagetk`. The CLI modes (`--cli`, `--boost`) work without Tk.

## What's new in v3

- **Auto-crop borders** toggle: strips the black emulator-padding bars
  per shot automatically (on by default).
- **HUD removal**: click the shot thumbnail on a HUD pixel to sample its
  color (health bars, timers, PRESS START); matching pixels get inpainted
  before projection. Clear the hex box to disable.
- **`--doctor`**: environment diagnosis with copy-paste fixes.
- **`--boost in.exr --out name`**: pseudo-HDR pass for any existing render —
  the post step for the Blender OBJ workflow below.
- **`blender_psx_bake.py`**: the OBJ path (see below).

## The OBJ path (best quality)

For real 1:1 geometry, use scurest's duckstation-3D-Screenshot fork
(prebuilt binaries on its GitHub Releases page — no compiling): enable
PGXP Geometry Correction, use its freecam to zoom out, take a 3D screenshot
(exports OBJ/MTL + textures). Then either:

```
blender --background --python blender_psx_bake.py -- \
    --obj shot.obj --out pano.exr --size 2048
python3 psx_lightgrab.py --boost pano.exr --out final
```

or interactively: import the OBJ in Blender, place the 3D cursor where you
want to stand, run blender_psx_bake.py from the Scripting tab. The script
converts materials to PS1-accurate emission (texture x vertex_color x 2 —
that doubling is the PS1 lighting model), drops an equirectangular camera at
the cursor, and renders a float EXR with zero bounces. The boost step then
inflates fire/lamps into proper key lights. No stitching, no seams, and you
can grab the lighting from anywhere inside the scene, not just where the
game camera stood.

## Workflow

1. **Navigate in DuckStation.** Load your ROM. Use save states
   (`F1`–`F8` style hotkeys), rewind, and frame advance to land on the exact
   scene/mission moment you want. This replaces building a custom
   ROM-timeline GUI — DuckStation already is one.
2. **Capture.** Point the in-game camera around and hit the screenshot
   hotkey (default `F10`). Where the camera goes:
   - **Free/rotatable camera** (e.g. lots of 3D platformers): grab Front,
     Right, Back, Left, plus Up/Down if possible → best panoramas.
   - **Fixed camera** (survival horror, racing): grab one shot and use
     **Gradient mode** — it reads the screenshot's vertical color structure
     (sky → ground) and sweeps it around the sphere. You lose detail but keep
     the palette and light-from-above behavior, which is most of what IBL
     gives you.
   - Some games have **freecam cheats/patches** (check DuckStation's
     built-in patch list or GameHacking-style cheat DBs) — those make full
     coverage easy.
3. **Ingest.** In LightGrab, `Add screenshots…` or `Watch folder…` pointed at
   DuckStation's screenshots directory — new shots appear automatically as
   you take them.
4. **Tag.** Select each shot, click Front/Right/Back/Left/Up/Down (or type
   exact yaw/pitch). Set HFOV to roughly match the game (most PSX games sit
   around 50–70°; if straight lines look bent in the panorama seams, nudge it).
5. **Build.** Live preview shows the equirect. Tune:
   - **De-dither** — melts the PSX checkerboard pattern (recommended on).
   - **HDR strength** — how hard highlights are inflated.
   - **Sun boost** — extra multiplier on the brightest blob (sun, lamp,
     fog glow) so it acts as a key light. Set to 1 to disable.
   - **Exposure** — global gain.
6. **Export.** You get:
   - `name.hdr` (Radiance) and `name.exr` — drop either into Blender
   - `name_preview.png` — tonemapped preview
   - `name_blender_setup.py` — run in Blender's Scripting tab (or
     `blender --python name_blender_setup.py`) to wire the map into the World
     shader with a Mapping node for rotation. It defaults to **Closest**
     interpolation so the map stays crunchy-PSX; flip to Linear in the
     Environment Texture node if you want it smooth.

## Headless / batch

Save a project in the GUI, then:

```
python3 psx_lightgrab.py --cli myproject.json --out level3_sunset
python3 psx_lightgrab.py --cli myproject.json --gradient shot.png --out level3_vibe
```

## Tips

- In Blender, if you only want the *lighting* and not the visible blurry
  backdrop: split the World shader into two Background nodes via a Light Path
  node ("Is Camera Ray") — grabbed map drives lighting, anything you like
  drives the visible background.
- Disable DuckStation's enhancements (internal resolution upscale is fine;
  avoid PGXP texture correction artifacts and any post-processing shaders)
  so captures stay faithful.
- If a game stores its sky as an actual skybox texture, DuckStation's
  texture-dumping feature can hand you the raw asset directly — sometimes
  that's cleaner than stitching. LightGrab can ingest those dumps as shots too.
