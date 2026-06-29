#!/usr/bin/env python3
"""
PSX LightGrab — grab scene lighting from PSX games and bring it into Blender.

Workflow:
  1. Play your game in DuckStation (savestates = your "checkpoints"; use
     rewind / frame advance to navigate to the exact moment you want).
  2. Hit DuckStation's screenshot hotkey (default F10) while pointing the
     in-game camera in different directions. Even 1-4 shots is enough.
  3. Point this tool at DuckStation's screenshots folder (it can watch it
     live, so new shots appear as you take them).
  4. Tag each shot's direction (Front / Right / Back / Left / Up / Down or
     exact yaw/pitch), set the game's approximate FOV, and hit Build.
  5. Export -> you get scene.hdr, scene.exr, a preview PNG, and
     blender_setup.py that wires it into Blender's World shader.

No ROM, BIOS, or emulator internals are touched — this works purely on
captured frames, so it's emulator-agnostic (PCSX2 / mGBA shots work too).

Run GUI:        python3 psx_lightgrab.py
Run headless:   python3 psx_lightgrab.py --cli project.json --out mymap
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import threading

import lightgrab_core as core


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def run_doctor():
    """Diagnose the local environment and print exact fixes."""
    import platform
    print(f"Python {platform.python_version()} on {platform.system()}")
    ok = True
    for mod, pipname in [("numpy", "numpy"), ("cv2", "opencv-python"),
                         ("PIL", "Pillow")]:
        try:
            __import__(mod)
            print(f"  [ok] {mod}")
        except ImportError as e:
            ok = False
            print(f"  [MISSING] {mod} -> pip install {pipname}"
                  f"{' --break-system-packages' if platform.system()=='Linux' else ''}"
                  f"   ({e})")
    try:
        import tkinter  # noqa
        print("  [ok] tkinter (GUI)")
    except ImportError:
        ok = False
        print("  [MISSING] tkinter -> Debian/Ubuntu: sudo apt install python3-tk"
              " | Fedora: sudo dnf install python3-tkinter"
              " | Windows/macOS: reinstall python.org Python with Tcl/Tk checked")
    try:
        from PIL import ImageTk  # noqa
        print("  [ok] PIL.ImageTk (thumbnails)")
    except ImportError:
        ok = False
        print("  [MISSING] PIL.ImageTk -> Debian/Ubuntu: sudo apt install"
              " python3-pil.imagetk (or: pip install --upgrade Pillow)")
    try:
        import cv2, numpy as np, os as _os
        _os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import tempfile
        p = _os.path.join(tempfile.gettempdir(), "_lg_probe.exr")
        exr = cv2.imwrite(p, np.ones((4, 4, 3), np.float32))
        print(f"  [{'ok' if exr else 'warn'}] EXR write "
              f"{'supported' if exr else 'unsupported (will fall back to .hdr only)'}")
        if _os.path.exists(p):
            _os.remove(p)
    except Exception as e:
        print(f"  [warn] EXR probe failed: {e} (.hdr output still works)")
    print("Environment OK — run: python3 psx_lightgrab.py" if ok
          else "Fix the items above, then re-run --doctor.")


def run_boost(argv):
    ap = argparse.ArgumentParser(description="Pseudo-HDR boost an existing render")
    ap.add_argument("--boost", metavar="IMAGE", required=True,
                    help="Input .exr/.hdr/.png (e.g. a Blender equirect bake)")
    ap.add_argument("--out", default="boosted", help="Output basename")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--sun", type=float, default=4.0)
    ap.add_argument("--exposure", type=float, default=1.0)
    args = ap.parse_args(argv)
    st = core.BuildSettings(hdr_strength=args.strength, sun_boost=args.sun,
                            exposure=args.exposure)
    hdr = core.boost_existing(args.boost, st)
    core.save_hdr(hdr, args.out + ".hdr")
    if core.save_exr(hdr, args.out + ".exr"):
        print("wrote", args.out + ".exr")
    core.save_preview(hdr, args.out + "_preview.png")
    core.write_blender_script(args.out + ".hdr", args.out + "_blender_setup.py")
    print("wrote", args.out + ".hdr")


def run_cli(argv):
    ap = argparse.ArgumentParser(description="PSX LightGrab headless build")
    ap.add_argument("--cli", metavar="PROJECT_JSON", required=True,
                    help="Project file created in the GUI (shots + settings)")
    ap.add_argument("--out", default="lightgrab_out", help="Output basename")
    ap.add_argument("--gradient", metavar="SCREENSHOT",
                    help="Skip panorama; build gradient env from one screenshot")
    args = ap.parse_args(argv)

    shots, settings = core.load_project(args.cli)
    if args.gradient:
        hdr = core.build_gradient(args.gradient, settings)
    else:
        hdr = core.build_panorama(shots, settings, progress=print)

    core.save_hdr(hdr, args.out + ".hdr")
    if core.save_exr(hdr, args.out + ".exr"):
        print("wrote", args.out + ".exr")
    core.save_preview(hdr, args.out + "_preview.png")
    core.write_blender_script(args.out + ".hdr", args.out + "_blender_setup.py")
    print("wrote", args.out + ".hdr", "and", args.out + "_blender_setup.py")


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk

    DIRS = {"Front": (0, 0), "Right": (90, 0), "Back": (180, 0),
            "Left": (270, 0), "Up": (0, 80), "Down": (0, -80)}

    class App:
        def __init__(self, root: tk.Tk):
            self.root = root
            root.title("PSX LightGrab")
            root.geometry("1180x720")
            self.shots: list[core.Shot] = []
            self.settings = core.BuildSettings()
            self.last_hdr = None
            self.watch_dir: str | None = None
            self._seen: set[str] = set()
            self._preview_imgs = []  # keep refs so Tk doesn't GC them

            self._build_layout()
            self._poll_watch()

        # ---------------- layout ---------------- #
        def _build_layout(self):
            main = ttk.PanedWindow(self.root, orient="horizontal")
            main.pack(fill="both", expand=True)

            # Left: shot list + per-shot controls
            left = ttk.Frame(main, padding=6)
            main.add(left, weight=1)

            btns = ttk.Frame(left)
            btns.pack(fill="x")
            ttk.Button(btns, text="Add screenshots…",
                       command=self.add_shots).pack(side="left")
            ttk.Button(btns, text="Watch folder…",
                       command=self.pick_watch).pack(side="left", padx=4)
            self.watch_lbl = ttk.Label(left, text="(not watching)",
                                       foreground="#888")
            self.watch_lbl.pack(fill="x")

            self.listbox = tk.Listbox(left, height=14, exportselection=False)
            self.listbox.pack(fill="both", expand=True, pady=4)
            self.listbox.bind("<<ListboxSelect>>", lambda e: self.on_select())

            lb_btns = ttk.Frame(left)
            lb_btns.pack(fill="x")
            ttk.Button(lb_btns, text="Remove",
                       command=self.remove_shot).pack(side="left")
            ttk.Button(lb_btns, text="Toggle on/off",
                       command=self.toggle_shot).pack(side="left", padx=4)

            shot_box = ttk.LabelFrame(left, text="Selected shot", padding=6)
            shot_box.pack(fill="x", pady=6)
            self.thumb_lbl = ttk.Label(shot_box)
            self.thumb_lbl.pack()
            self.thumb_lbl.bind("<Button-1>", self._sample_hud_color)
            self._thumb_src = None   # (PIL image, scale_x, scale_y)

            quick = ttk.Frame(shot_box)
            quick.pack(pady=4)
            for name in DIRS:
                ttk.Button(quick, text=name, width=6,
                           command=lambda n=name: self.set_dir(n)
                           ).pack(side="left", padx=1)

            grid = ttk.Frame(shot_box)
            grid.pack()
            self.var_yaw = tk.DoubleVar(value=0)
            self.var_pitch = tk.DoubleVar(value=0)
            self.var_fov = tk.DoubleVar(value=60)
            for col, (label, var) in enumerate(
                    [("Yaw°", self.var_yaw), ("Pitch°", self.var_pitch),
                     ("HFOV°", self.var_fov)]):
                ttk.Label(grid, text=label).grid(row=0, column=col, padx=6)
                e = ttk.Entry(grid, textvariable=var, width=7)
                e.grid(row=1, column=col, padx=6)
                e.bind("<FocusOut>", lambda ev: self.apply_fields())
                e.bind("<Return>", lambda ev: self.apply_fields())

            # Right: settings + preview + actions
            right = ttk.Frame(main, padding=6)
            main.add(right, weight=2)

            st = ttk.LabelFrame(right, text="Build settings", padding=6)
            st.pack(fill="x")

            self.var_width = tk.IntVar(value=self.settings.out_width)
            self.var_hdr = tk.DoubleVar(value=self.settings.hdr_strength)
            self.var_sun = tk.DoubleVar(value=self.settings.sun_boost)
            self.var_exp = tk.DoubleVar(value=self.settings.exposure)
            self.var_dither = tk.BooleanVar(value=self.settings.denoise_dither)
            self.var_dither_mode = tk.StringVar(value=self.settings.dither_mode)

            row = ttk.Frame(st); row.pack(fill="x")
            ttk.Label(row, text="Width").pack(side="left")
            ttk.Combobox(row, textvariable=self.var_width, width=6,
                         values=[1024, 2048, 4096]).pack(side="left", padx=4)
            ttk.Checkbutton(row, text="De-dither",
                            variable=self.var_dither).pack(side="left", padx=10)
            ttk.Combobox(row, textvariable=self.var_dither_mode, width=7,
                         values=["bayer", "median", "off"],
                         state="readonly").pack(side="left")

            self.var_crop = tk.BooleanVar(value=self.settings.auto_crop)
            self.var_hud = tk.StringVar(value=self.settings.hud_color)
            row2 = ttk.Frame(st); row2.pack(fill="x", pady=2)
            ttk.Checkbutton(row2, text="Auto-crop black borders",
                            variable=self.var_crop).pack(side="left")
            ttk.Label(row2, text="  HUD color (hex):").pack(side="left")
            ttk.Entry(row2, textvariable=self.var_hud,
                      width=9).pack(side="left", padx=2)
            self.hud_swatch = tk.Label(row2, text="  ", relief="sunken")
            self.hud_swatch.pack(side="left", padx=2)
            ttk.Label(row2, text="(click the shot thumbnail to sample;"
                                 " empty = off)",
                      foreground="#888").pack(side="left", padx=4)
            self.var_hud.trace_add("write", lambda *a: self._update_swatch())

            for label, var, lo, hi in [("HDR strength", self.var_hdr, 0.0, 0.95),
                                       ("Sun boost", self.var_sun, 1.0, 12.0),
                                       ("Exposure", self.var_exp, 0.2, 4.0)]:
                r = ttk.Frame(st); r.pack(fill="x", pady=2)
                ttk.Label(r, text=label, width=12).pack(side="left")
                ttk.Scale(r, from_=lo, to=hi, variable=var,
                          orient="horizontal").pack(side="left", fill="x",
                                                    expand=True, padx=4)
                ttk.Label(r, textvariable=var, width=5).pack(side="left")

            act = ttk.Frame(right); act.pack(fill="x", pady=6)
            ttk.Button(act, text="Build panorama",
                       command=self.build).pack(side="left")
            ttk.Button(act, text="Gradient from selected shot",
                       command=self.build_gradient).pack(side="left", padx=4)
            ttk.Button(act, text="Export…",
                       command=self.export).pack(side="left", padx=4)
            ttk.Button(act, text="Save project",
                       command=self.save_project).pack(side="right")
            ttk.Button(act, text="Load project",
                       command=self.load_project).pack(side="right", padx=4)

            self.status = tk.StringVar(value="Add screenshots to begin.")
            ttk.Label(right, textvariable=self.status,
                      foreground="#2a6").pack(fill="x")

            self.preview_lbl = ttk.Label(
                right, text="(panorama preview appears here)",
                anchor="center", relief="sunken")
            self.preview_lbl.pack(fill="both", expand=True, pady=4)

        # ---------------- shot management ---------------- #
        def add_shots(self):
            paths = filedialog.askopenfilenames(
                title="Pick emulator screenshots",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
            self._ingest(paths)

        def pick_watch(self):
            d = filedialog.askdirectory(
                title="DuckStation screenshots folder")
            if d:
                self.watch_dir = d
                self._seen = set(self._all_in_watch())
                self.watch_lbl.config(text=f"Watching: {d}")

        def _all_in_watch(self):
            if not self.watch_dir:
                return []
            out = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                out += glob.glob(os.path.join(self.watch_dir, ext))
            return out

        def _poll_watch(self):
            if self.watch_dir:
                new = [p for p in self._all_in_watch() if p not in self._seen]
                if new:
                    self._seen.update(new)
                    self._ingest(sorted(new))
                    self.status.set(f"Picked up {len(new)} new screenshot(s)")
            self.root.after(1500, self._poll_watch)

        def _ingest(self, paths):
            for p in paths:
                self.shots.append(core.Shot(path=p))
            self.refresh_list()

        def refresh_list(self):
            self.listbox.delete(0, "end")
            for s in self.shots:
                flag = "" if s.enabled else "[off] "
                self.listbox.insert(
                    "end", f"{flag}{os.path.basename(s.path)}  "
                           f"yaw {s.yaw_deg:g}  pitch {s.pitch_deg:g}  "
                           f"fov {s.hfov_deg:g}")

        def current(self) -> core.Shot | None:
            sel = self.listbox.curselection()
            return self.shots[sel[0]] if sel else None

        def on_select(self):
            s = self.current()
            if not s:
                return
            self.var_yaw.set(s.yaw_deg)
            self.var_pitch.set(s.pitch_deg)
            self.var_fov.set(s.hfov_deg)
            try:
                full = Image.open(s.path).convert("RGB")
                im = full.copy()
                im.thumbnail((280, 200))
                self._thumb_src = (full, full.width / im.width,
                                   full.height / im.height)
                tkim = ImageTk.PhotoImage(im)
                self._preview_imgs = [tkim]
                self.thumb_lbl.config(image=tkim, text="")
            except Exception as e:
                self._thumb_src = None
                self.thumb_lbl.config(image="", text=f"(can't preview: {e})")

        def _sample_hud_color(self, event):
            if not self._thumb_src:
                return
            full, sx, sy = self._thumb_src
            x = min(int(event.x * sx), full.width - 1)
            y = min(int(event.y * sy), full.height - 1)
            r, g, b = full.getpixel((x, y))
            self.var_hud.set(f"#{r:02x}{g:02x}{b:02x}")
            self.status.set(f"HUD color sampled: #{r:02x}{g:02x}{b:02x} "
                            f"(clear the box to disable removal)")

        def _update_swatch(self):
            c = self.var_hud.get().strip()
            try:
                if len(c.lstrip("#")) == 6:
                    self.hud_swatch.config(bg=c if c.startswith("#") else "#"+c)
            except tk.TclError:
                pass

        def apply_fields(self):
            s = self.current()
            if not s:
                return
            idx = self.listbox.curselection()[0]
            try:
                s.yaw_deg = float(self.var_yaw.get())
                s.pitch_deg = float(self.var_pitch.get())
                s.hfov_deg = float(self.var_fov.get())
            except (ValueError, tk.TclError):
                return
            self.refresh_list()
            self.listbox.selection_set(idx)

        def set_dir(self, name):
            s = self.current()
            if not s:
                return
            yaw, pitch = DIRS[name]
            self.var_yaw.set(yaw)
            self.var_pitch.set(pitch)
            self.apply_fields()

        def remove_shot(self):
            sel = self.listbox.curselection()
            if sel:
                del self.shots[sel[0]]
                self.refresh_list()

        def toggle_shot(self):
            s = self.current()
            if s:
                idx = self.listbox.curselection()[0]
                s.enabled = not s.enabled
                self.refresh_list()
                self.listbox.selection_set(idx)

        # ---------------- build / export ---------------- #
        def _grab_settings(self) -> core.BuildSettings:
            self.settings.out_width = int(self.var_width.get())
            self.settings.hdr_strength = float(self.var_hdr.get())
            self.settings.sun_boost = float(self.var_sun.get())
            self.settings.exposure = float(self.var_exp.get())
            self.settings.denoise_dither = bool(self.var_dither.get())
            self.settings.dither_mode = self.var_dither_mode.get()
            self.settings.auto_crop = bool(self.var_crop.get())
            hud = self.var_hud.get().strip()
            self.settings.hud_color = hud if len(hud.lstrip("#")) == 6 else ""
            self.settings.watch_dir = self.watch_dir or ""
            return self.settings

        def build(self):
            if not any(s.enabled for s in self.shots):
                messagebox.showinfo("PSX LightGrab", "Add at least one shot.")
                return
            self._run_bg(lambda: core.build_panorama(
                self.shots, self._grab_settings(),
                progress=lambda m: self.status.set(m)))

        def build_gradient(self):
            s = self.current()
            if not s:
                messagebox.showinfo("PSX LightGrab",
                                    "Select a shot for gradient mode.")
                return
            self._run_bg(lambda: core.build_gradient(
                s.path, self._grab_settings()))

        def _run_bg(self, fn):
            def work():
                try:
                    hdr = fn()
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Build failed", str(e)))
                    return
                self.last_hdr = hdr
                self.root.after(0, self._show_preview)
            self.status.set("Building…")
            threading.Thread(target=work, daemon=True).start()

        def _show_preview(self):
            import numpy as np
            hdr = self.last_hdr
            tm = hdr / (1.0 + hdr)
            srgb = (core.linear_to_srgb(tm) * 255).astype("uint8")
            im = Image.fromarray(srgb)
            w = max(self.preview_lbl.winfo_width(), 400)
            im.thumbnail((w, w // 2))
            tkim = ImageTk.PhotoImage(im)
            self._preview_imgs.append(tkim)
            self.preview_lbl.config(image=tkim, text="")
            self.status.set(
                f"Done. Peak value {float(np.max(hdr)):.2f} "
                f"(>1.0 means real HDR range). Now hit Export.")

        def export(self):
            if self.last_hdr is None:
                messagebox.showinfo("PSX LightGrab", "Build something first.")
                return
            base = filedialog.asksaveasfilename(
                title="Export basename", defaultextension=".hdr",
                filetypes=[("Radiance HDR", "*.hdr")])
            if not base:
                return
            base = os.path.splitext(base)[0]
            core.save_hdr(self.last_hdr, base + ".hdr")
            exr_ok = core.save_exr(self.last_hdr, base + ".exr")
            core.save_preview(self.last_hdr, base + "_preview.png")
            core.write_blender_script(base + ".hdr",
                                      base + "_blender_setup.py")
            extra = " + .exr" if exr_ok else ""
            self.status.set(
                f"Exported {os.path.basename(base)}.hdr{extra}, preview, "
                f"and blender_setup.py")

        # ---------------- project save/load ---------------- #
        def save_project(self):
            p = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("LightGrab project", "*.json")])
            if p:
                core.save_project(p, self.shots, self._grab_settings())
                self.status.set(f"Project saved: {os.path.basename(p)}")

        def load_project(self):
            p = filedialog.askopenfilename(
                filetypes=[("LightGrab project", "*.json")])
            if not p:
                return
            self.shots, self.settings = core.load_project(p)
            self.var_width.set(self.settings.out_width)
            self.var_hdr.set(self.settings.hdr_strength)
            self.var_sun.set(self.settings.sun_boost)
            self.var_exp.set(self.settings.exposure)
            self.var_dither.set(self.settings.denoise_dither)
            self.var_dither_mode.set(self.settings.dither_mode)
            self.var_crop.set(self.settings.auto_crop)
            self.var_hud.set(self.settings.hud_color)
            if self.settings.watch_dir and os.path.isdir(self.settings.watch_dir):
                self.watch_dir = self.settings.watch_dir
                self._seen = set(self._all_in_watch())
                self.watch_lbl.config(text=f"Watching: {self.watch_dir}")
            self.refresh_list()
            self.status.set(f"Project loaded: {os.path.basename(p)}")

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    if "--doctor" in sys.argv:
        run_doctor()
    elif "--boost" in sys.argv:
        run_boost(sys.argv[1:])
    elif "--cli" in sys.argv:
        run_cli(sys.argv[1:])
    else:
        try:
            run_gui()
        except ImportError as e:
            print(f"GUI failed to start ({e}).\n"
                  "Run:  python3 psx_lightgrab.py --doctor   for exact fixes.")
            sys.exit(1)
