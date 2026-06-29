"""
blender_psx_bake.py — render an equirectangular EXR from inside a PSX scene
captured with duckstation-3D-Screenshot (scurest's fork).

Usage A (headless):
    blender --background --python blender_psx_bake.py -- \
        --obj /path/to/shot.obj --out /path/to/arena_pano.exr --size 2048

Usage B (interactive):
    1. Import the OBJ yourself (File > Import > Wavefront), position the
       3D cursor where you want the "camera" to stand, open this file in the
       Scripting tab, set OBJ_PATH = None below, and Run Script.

What it does:
    1. (optional) imports the OBJ/MTL from the 3D screenshot
    2. converts every material to PS1-style shading: emission of
       texture * vertex_color * 2.0  (PS1 modulation treats 128 as 1.0, so
       vertex colors can brighten up to 2x — that doubling IS the lighting)
    3. textures get Closest interpolation (crunchy pixels, no smearing)
    4. drops an equirectangular panoramic camera at the 3D cursor
    5. renders a float EXR with Cycles, 0 light bounces (the scene already
       carries its lighting in vertex colors; nothing needs to bounce)

Afterwards, fabricate extra dynamic range for the bright bits (fire, lamps):
    python3 psx_lightgrab.py --boost arena_pano.exr --out arena_final
and load arena_final.hdr in your World shader (or use the generated
arena_final_blender_setup.py).

Tested API paths for Blender 3.x and 4.x/5.x (panorama settings moved from
cycles per-camera props to camera data in 4.0; both are handled).
"""

import argparse
import bpy
import os
import sys
import math

# ---------------------------------------------------------------- defaults --
OBJ_PATH = None          # set to a path, or pass --obj on the command line
OUT_PATH = None          # resolved in parse_args; None means derive from blend file
SIZE = 2048              # equirect width; height = SIZE // 2
EMISSION_GAIN = 1.0      # global multiplier on top of the PS1 2x modulation


def parse_args() -> None:
    """Parse CLI args passed after '--' (Blender convention)."""
    global OBJ_PATH, OUT_PATH, SIZE, EMISSION_GAIN
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(
        prog="blender_psx_bake.py",
        description="Render a PSX scene OBJ as an equirectangular EXR")
    ap.add_argument("--obj", metavar="PATH",
                    help="OBJ file exported by DuckStation 3D screenshot")
    ap.add_argument("--out", metavar="PATH", default=None,
                    help="Output .exr path (default: psx_pano.exr beside the .blend)")
    ap.add_argument("--size", type=int, default=SIZE,
                    help="Equirect width in pixels (height = size // 2)")
    ap.add_argument("--gain", type=float, default=EMISSION_GAIN,
                    help="Emission multiplier on top of the PS1 2x modulation")
    args = ap.parse_args(argv)

    OBJ_PATH = args.obj
    SIZE = args.size
    EMISSION_GAIN = args.gain

    if args.out is not None:
        OUT_PATH = os.path.abspath(args.out)
    else:
        blend = bpy.data.filepath
        if blend:
            OUT_PATH = os.path.join(os.path.dirname(blend), "psx_pano.exr")
        else:
            OUT_PATH = os.path.abspath("psx_pano.exr")
            print(f"[psx_bake] no saved .blend file — writing to {OUT_PATH}")


def import_obj(path):
    """OBJ import across Blender versions (operator renamed in 4.0)."""
    if hasattr(bpy.ops.wm, "obj_import"):          # Blender 3.3+ new importer
        bpy.ops.wm.obj_import(filepath=path)
    else:                                          # legacy
        bpy.ops.import_scene.obj(filepath=path)
    return list(bpy.context.selected_objects)


def find_vertex_color_name(mesh_objects):
    """Name of the vertex color / color attribute layer, if any."""
    for ob in mesh_objects:
        me = ob.data
        attrs = getattr(me, "color_attributes", None)
        if attrs and len(attrs):
            return attrs[0].name
        vcols = getattr(me, "vertex_colors", None)
        if vcols and len(vcols):
            return vcols[0].name
    return None


def ps1ify_material(mat, vcol_name, gain):
    """Rebuild material as: Emission( texture * vertex_color * 2 * gain )."""
    mat.use_nodes = True
    nt = mat.node_tree

    # find an existing image texture before clearing
    image = None
    for n in nt.nodes:
        if n.type == "TEX_IMAGE" and n.image:
            image = n.image
            break

    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    emis = nt.nodes.new("ShaderNodeEmission");      emis.location = (400, 0)
    emis.inputs["Strength"].default_value = 2.0 * gain   # PS1: 128 == 1.0
    nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])

    mul = nt.nodes.new("ShaderNodeMix") if hasattr(bpy.types, "ShaderNodeMix") \
        else nt.nodes.new("ShaderNodeMixRGB")
    # Use a simple multiply via Mix node; handle both new and legacy APIs.
    if mul.bl_idname == "ShaderNodeMix":
        mul.data_type = 'RGBA'
        mul.blend_type = 'MULTIPLY'
        mul.inputs["Factor"].default_value = 1.0
        in_a, in_b, out_col = mul.inputs[6], mul.inputs[7], mul.outputs[2]
    else:
        mul.blend_type = 'MULTIPLY'
        mul.inputs["Fac"].default_value = 1.0
        in_a, in_b, out_col = mul.inputs["Color1"], mul.inputs["Color2"], \
            mul.outputs["Color"]
    mul.location = (180, 0)
    nt.links.new(out_col, emis.inputs["Color"])

    # texture leg
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = "Closest"
        tex.location = (-120, 120)
        nt.links.new(tex.outputs["Color"], in_a)
    else:
        in_a.default_value = (1.0, 1.0, 1.0, 1.0)

    # vertex color leg
    if vcol_name:
        if hasattr(bpy.types, "ShaderNodeVertexColor"):
            vc = nt.nodes.new("ShaderNodeVertexColor")
            vc.layer_name = vcol_name
            sock = vc.outputs["Color"]
        else:
            vc = nt.nodes.new("ShaderNodeAttribute")
            vc.attribute_name = vcol_name
            sock = vc.outputs["Color"]
        vc.location = (-120, -120)
        nt.links.new(sock, in_b)
    else:
        in_b.default_value = (1.0, 1.0, 1.0, 1.0)


def make_pano_camera():
    cam_data = bpy.data.cameras.new("PSX_PanoCam")
    cam_data.type = 'PANO'
    # Blender 4.0+: panorama_type lives on camera data; 3.x: on cam.cycles
    if hasattr(cam_data, "panorama_type"):
        cam_data.panorama_type = 'EQUIRECTANGULAR'
    elif hasattr(cam_data, "cycles"):
        cam_data.cycles.panorama_type = 'EQUIRECTANGULAR'
    cam = bpy.data.objects.new("PSX_PanoCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = bpy.context.scene.cursor.location
    # camera looks down -Z; pitch up 90° so the equator is horizontal
    cam.rotation_euler = (math.radians(90), 0, 0)
    return cam


def main():
    parse_args()
    scene = bpy.context.scene

    objects = []
    if OBJ_PATH:
        objects = import_obj(OBJ_PATH)
    mesh_objects = [o for o in (objects or scene.objects) if o.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh objects found — import the 3D screenshot "
                           "OBJ first or pass --obj.")

    vcol = find_vertex_color_name(mesh_objects)
    mats = {slot.material for ob in mesh_objects for slot in ob.material_slots
            if slot.material}
    for m in mats:
        ps1ify_material(m, vcol, EMISSION_GAIN)
    print(f"PS1-ified {len(mats)} material(s); vertex colors: {vcol or 'none'}")

    scene.camera = make_pano_camera()

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16            # emission-only: cheap
    scene.cycles.max_bounces = 0         # lighting is baked in; nothing bounces
    scene.render.resolution_x = SIZE
    scene.render.resolution_y = SIZE // 2
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.filepath = OUT_PATH
    scene.view_settings.view_transform = 'Standard'   # no Filmic on data

    # black void where geometry is missing (frustum-culled regions)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("PSX_Void")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)

    bpy.ops.render.render(write_still=True)
    print(f"Wrote {OUT_PATH} — now run: "
          f"python3 psx_lightgrab.py --boost {OUT_PATH} --out final")


main()
