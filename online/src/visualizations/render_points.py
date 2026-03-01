import sys
import os
import bpy
import math
import numpy as np
import re

# --- ARGV HANDLING ---
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = sys.argv

# --- CONFIG ---
if len(argv) < 3:
    print("Usage: <MATERIAL> <IN_DIR> <OUT_DIR>")
    sys.exit(1)

MAT_NAME = argv[0].upper()
IN_DIR = argv[1]
OUT_DIR = argv[2]

if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

# --- COLORS ---
COLORS = {
    "PLASTICINE": (0.1, 0.8, 0.2, 1), 
    "ELASTIC":    (144.0/255, 210.0/255, 236.0/255, 1), 
    "WATER":      (0.2, 0.4, 0.8, 1), 
    "SAND":       (194.0/255, 178.0/255, 128.0/255, 1), 
    "RIGID":      (0.8, 0.2, 0.2, 1), 
    "CHOCOLATE":  (0.36, 0.2, 0.1, 1)
}
ptColor = COLORS.get(MAT_NAME, (0.5, 0.5, 0.5, 1))

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def blenderInit(res_x, res_y, samples, exposure):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.view_settings.exposure = exposure
    
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for d in prefs.devices: d.use = True
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def readMesh(path, loc, rot, scale):
    # Blender 4.0+ Import
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.location = loc
    obj.rotation_euler = [math.radians(x) for x in rot]
    obj.scale = scale
    return obj

def setMat_pointCloud(mesh, color, ptSize):
    # Create GeoNodes Modifier
    mod = mesh.modifiers.new("GeoNodes", "NODES")
    node_group = bpy.data.node_groups.new("PointsToSpheres", "GeometryNodeTree")
    mod.node_group = node_group
    
    # --- CRITICAL FIX FOR BLENDER 4.0+ ---
    if hasattr(node_group, "interface"):
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    else:
        node_group.inputs.new('NodeSocketGeometry', 'Geometry')
        node_group.outputs.new('NodeSocketGeometry', 'Geometry')

    nodes = node_group.nodes
    links = node_group.links
    
    # Create Nodes
    input_node = nodes.new('NodeGroupInput')
    output_node = nodes.new('NodeGroupOutput')
    
    # Instance on Points
    instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
    
    # Icosphere (The particle)
    icosphere = nodes.new('GeometryNodeMeshIcoSphere')
    icosphere.inputs['Radius'].default_value = ptSize
    icosphere.inputs['Subdivisions'].default_value = 2
    
    # Set Material
    set_mat = nodes.new('GeometryNodeSetMaterial')
    
    # Create Material Data
    mat = bpy.data.materials.new(name="PtMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = color
    set_mat.inputs['Material'].default_value = mat
    
    # Links
    links.new(input_node.outputs[0], instance_on_points.inputs['Points'])
    links.new(icosphere.outputs['Mesh'], instance_on_points.inputs['Instance'])
    links.new(instance_on_points.outputs['Instances'], set_mat.inputs['Geometry'])
    links.new(set_mat.outputs['Geometry'], output_node.inputs[0])

def setCamera(loc, lookAt, focal_len):
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.active_object
    cam.data.lens = focal_len
    
    # Track To Constraint
    tt = cam.constraints.new(type='TRACK_TO')
    bpy.ops.object.empty_add(location=lookAt)
    target = bpy.context.active_object
    tt.target = target
    tt.track_axis = 'TRACK_NEGATIVE_Z'
    tt.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = cam
    return cam

def setLight_sun(angle, strength, softness):
    rot = [math.radians(x) for x in angle]
    bpy.ops.object.light_add(type='SUN', rotation=rot)
    sun = bpy.context.active_object
    sun.data.energy = strength
    sun.data.angle = softness

def setLight_ambient(color):
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = color

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

# 1. Init
blenderInit(1480, 1480, 40, 1.5)

if MAT_NAME == "ELASTIC":
    OBJ_LOC = (0.6, -0.8, 0.6)
    LOOK_AT = (0.6 + 0.5, -0.8 + 0.5, 0.6 + 0.5) 
    cam = setCamera(loc=(3.0, -3.0, 2.5), lookAt=LOOK_AT, focal_len=45)
    setLight_sun((-30, -30, 155), 2, 0.3)
    setLight_ambient((0.1, 0.1, 0.1, 1))

else:
    cam = setCamera(loc=(3, 0, 2), lookAt=(0, 0, 0.5), focal_len=45)
    setLight_sun((-30, -30, 155), 2, 0.3)
    setLight_ambient((0.1, 0.1, 0.1, 1))

# 4. Process
files = sorted([f for f in os.listdir(IN_DIR) if f.endswith('.obj')])
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))) if any(c.isdigit() for c in f) else f)

print(f"Rendering {len(files)} frames for {MAT_NAME}...")

for f in files:
    match = re.search(r"(\d+)", f)
    if match:
        frame_idx = int(match.group(1))
        out_name = f"pred_{frame_idx:04d}.png"
    else:
        out_name = f + ".png"

    out_path = os.path.join(OUT_DIR, out_name)
    in_path = os.path.join(IN_DIR, f)
    
    print(f"Processing: {f}")
    
    # Import
    mesh = readMesh(in_path, (0.6, -0.8, 0.6), (90, 0, 120), (1.3, 1.3, 1.3))
    
    # Material
    setMat_pointCloud(mesh, ptColor, 0.01)
    
    # Render
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    
    # Cleanup
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    bpy.ops.object.delete()