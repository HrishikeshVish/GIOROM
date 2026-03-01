import sys
import os
import json
import math
import bpy

# --- ARGV HANDLING ---
# Blender wraps arguments. We only want what's after "--"
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = sys.argv
# ---------------------

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Dynamically fetch toolbox path, defaulting to your local setup
TOOLBOX_PATH = os.environ.get("BLENDER_TOOLBOX_DIR", "/media/csuser/DATA/BlenderToolbox")

if not os.path.exists(TOOLBOX_PATH):
    print(f"\n[CRITICAL ERROR] Toolbox not found at: {TOOLBOX_PATH}")
    print("Please set the BLENDER_TOOLBOX_DIR environment variable.")
    sys.exit(1)

# Add to sys.path so 'import custom_mats' works
if TOOLBOX_PATH not in sys.path:
    sys.path.append(TOOLBOX_PATH)

try:
    import custom_mats
except ImportError as e:
    print(f"\n[IMPORT ERROR] Failed to import custom_mats: {e}")
    sys.exit(1)

# ==============================================================================
# CONFIG
# ==============================================================================
mat_names = ["PLASTICINE", "WATER", "ELASTIC", "SAND", "RIGID", "CHOCOLATE"]

# Argument Indices (after shifting argv)
ARG_BULK_OR_SINGLE = 0
ARG_MATERIAL = 1
ARG_CONFIG_PATH = 2
ARG_PTS_TO_MSH = 3
ARG_SHOW_BOX = 4
ARG_IS_MESH_INVISIBLE = 5
ARG_INPUT_FOLDER_OR_OBJ_FILE_PATH = 6
ARG_OUTPUT_FOLDER_PATH = 7
ARG_PNG_FILENAME = 8

BULK_ARG_COUNT = 8 

def print_help():
    print("\nUsage:")
    print("blender -b -P render_visuals.py -- -b <MAT> <config.json> <pts2mesh> <box> <inv_plane> <IN_DIR> <OUT_DIR>")

if len(argv) < BULK_ARG_COUNT:
    print(f"Wrong arg count! Got {len(argv)}, expected at least {BULK_ARG_COUNT}")
    print_help()
    sys.exit(1)

# Parse Arguments
is_bulk = argv[ARG_BULK_OR_SINGLE].strip() == "-b"
mat_name = argv[ARG_MATERIAL].strip().upper()
jsonConfigPath = os.path.abspath(argv[ARG_CONFIG_PATH].strip())

if mat_name not in mat_names:
    print(f"Invalid material: {mat_name}. Options: {mat_names}")
    sys.exit(1)

if not os.path.exists(jsonConfigPath):
    print(f"Config file not found: {jsonConfigPath}")
    sys.exit(1)

with open(jsonConfigPath, "r") as file:
    try:
        configData = json.load(file)
        obj_loc = configData["object"]["location"]
        obj_rot = configData["object"]["rotation"]
        obj_scl = configData["object"]["scale"]
        
        box_loc = configData["box"]["location"]
        box_rot = configData["box"]["rotation"]
        box_scl = configData["box"]["scale"]

        cam_loc = configData["camera"]["location"]
        cam_rot = configData["camera"]["rotation"]
        
        pt_density = configData["pointsToVolumeDensity"]
        pt_voxel = configData["pointsToVolumeVoxelAmount"]
        pt_radius = configData["pointsToVolumeRadius"]
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        sys.exit(1)

try:
    do_point_to_mesh = int(argv[ARG_PTS_TO_MSH]) != 0
    do_box = int(argv[ARG_SHOW_BOX]) != 0
    is_plane_invisible = int(argv[ARG_IS_MESH_INVISIBLE]) != 0
except ValueError:
    print("Error parsing boolean flags (must be 0 or 1)")
    sys.exit(1)

inputPath = os.path.abspath(argv[ARG_INPUT_FOLDER_OR_OBJ_FILE_PATH])
outputPath = os.path.abspath(argv[ARG_OUTPUT_FOLDER_PATH])

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# ==============================================================================
# MAIN RENDERING LOOP
# ==============================================================================
procPairs = []
if is_bulk:
    if not os.path.isdir(inputPath):
        print(f"Input path is not a directory: {inputPath}")
        sys.exit(1)
    
    # Sort files naturally
    all_files = [f for f in os.listdir(inputPath) if f.endswith('.obj')]
    all_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))) if any(c.isdigit() for c in f) else f)

    for f in all_files:
        procPairs.append((os.path.join(inputPath, f), os.path.join(outputPath, f + ".png")))
else:
    fname = argv[ARG_PNG_FILENAME]
    procPairs.append((inputPath, os.path.join(outputPath, fname)))

print(f"Found {len(procPairs)} frames to render.")

for i, (meshPath, outImgPath) in enumerate(procPairs):
    if os.path.exists(outImgPath):
        pass

    print(f"Rendering Frame {i}: {os.path.basename(meshPath)}")

    # 1. Reset Blender
    bpy.ops.wm.read_factory_settings(use_empty=True)
    custom_mats.blenderInit(1000, 1000, 40, 1.5) 

    # 2. Load Mesh
    location = (obj_loc["x"], obj_loc["y"], obj_loc["z"])
    rotation = (obj_rot["x"], obj_rot["y"], obj_rot["z"])
    scale = (obj_scl["x"], obj_scl["y"], obj_scl["z"])
    
    mesh = custom_mats.readMesh(meshPath, location, rotation, scale)

    # 3. Load Box
    if do_box:
        box_path = "Box.obj"
        if not os.path.exists(box_path): box_path = os.path.join(TOOLBOX_PATH, "Box.obj")
        
        if os.path.exists(box_path):
            b_loc = (box_loc["x"], box_loc["y"], box_loc["z"])
            b_rot = (box_rot["x"], box_rot["y"], box_rot["z"])
            b_scl = (box_scl["x"], box_scl["y"], box_scl["z"])
            box_mesh = custom_mats.readMesh(box_path, b_loc, b_rot, b_scl)
            custom_mats.setMatGlassBox(box_mesh)

    # 4. Material
    smooth = True 
    if mat_name == "PLASTICINE": mat = custom_mats.setMatPlasticine(mesh)
    elif mat_name == "WATER":
        mat = custom_mats.setMatWater(mesh)
        smooth = False
    elif mat_name == "SAND":
        mat = custom_mats.setMatSand(mesh)
        smooth = False
    elif mat_name == "ELASTIC": mat = custom_mats.setMatElastic(mesh)
    elif mat_name == "RIGID":
        mat = custom_mats.setMatPlastic(mesh, (144.0/255, 210.0/255, 236.0/255, 1, 0.5, 1.0, 1.0, 0.0, 2.0))
    elif mat_name == "CHOCOLATE": mat = custom_mats.setMatChocolate(mesh)

    custom_mats.pointCloudToSmoothMesh(
        mesh, mat, smooth,
        pointsToVolDensity=pt_density,
        pointsToVolRadius=pt_radius,
        pointsToVolVoxelAmt=pt_voxel,
        doConvertPointsToMesh=do_point_to_mesh
    )

    # 5. Environment
    bpy.context.scene.world.color = (0.6, 0.6, 0.6)
    custom_mats.generateWhiteGroundPlane((-0.56, -0.78, 0), (0, 0, 0), (0.068, 0.068, 1), is_plane_invisible)

    # 6. Camera
    c_loc = (cam_loc["x"], cam_loc["y"], cam_loc["z"])
    c_rot_deg = (cam_rot["x"], cam_rot["y"], cam_rot["z"])
    c_rot_rad = tuple(math.radians(x) for x in c_rot_deg)
    cam = custom_mats.setCamera(c_loc, None, 45, c_rot_rad)

    # 7. Light
    custom_mats.createLightSun(
        (math.radians(338), math.radians(-2148), math.radians(7004)), 
        4.5, 
        10, 
        0.1 # sun_angle
    )
    custom_mats.setLight_ambient((0.1, 0.1, 0.1, 1))
    custom_mats.shadowThreshold(0.05, 'CARDINAL')

    # 8. Render
    custom_mats.renderImage(outImgPath, cam)

print("Batch rendering complete.")