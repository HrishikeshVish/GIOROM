# # if you want to call the toolbox the old way with `blender -b -P demo_XXX.py`, then uncomment these two lines
# import sys, os
# sys.path.append("../../BlenderToolbox/")
import math

import bpy
import os
import sys
import json

import custom_mats

mat_names = ["PLASTICINE", "WATER", "ELASTIC", "SAND", "RIGID", "CHOCOLATE"]


def print_config_help():
    print(f"Config JSON file must be of the following structure:")
    print(
        '{\n'
        '\t"object": {\n'
        '\t\t"location": {"x": float, "y": float, "z": float},\n'
        '\t\t"rotation": {"x": float, "y": float, "z": float},\n'
        '\t\t"scale": {"x": float, "y": float, "z": float}\n'
        '\t},\n'
        '\t"box": {'
        '\t\t"location": {"x": float, "y": float, "z": float},\n'
        '\t\t"rotation": {"x": float, "y": float, "z": float},\n'
        '\t\t"scale": {"x": float, "y": float, "z": float}\n'
        '\t}'
        '\t"camera": {\n'
        '\t\t"location": {"x": float, "y": float, "z": float},\n'
        '\t\t"rotation": {"x": float, "y": float, "z": float}\n'
        '\t},\n'
        '\t"pointsToVolumeDensity": float,\n'
        '\t"pointsToVolumeVoxelAmount": float,\n'
        '\t"pointsToVolumeRadius": float\n'
        '}'
    )


def print_mat_help():
    print(f"material must be one of one of {mat_names}")


def print_convert_help():
    print(f"'convert points to mesh', 'show box', and 'is plane invisible' must be 0 or 1, where 0 means the input is already a mesh and 1 is a point cloud input")


def print_help():
    print("\nBulk processing usage:")
    print("\t-b <material> <config json file path> <convert points to mesh> <show box?> <is plane invisible?> <input folder path> <output folder path>")
    print("\nSingle file processing usage:")
    print("\t-s <material> <config json file path> <convert points to mesh> <show box?> <is plane invisible?> <input obj file path> <output folder path> <output png filename>")
    print()
    print_mat_help()
    print()
    print_convert_help()
    print()
    print_config_help()
    print()


SINGLE_ARG_COUNT = 10
BULK_ARG_COUNT = SINGLE_ARG_COUNT - 1
ARG_BULK_OR_SINGLE = 1
ARG_MATERIAL = 2
ARG_CONFIG_PATH = 3
ARG_PTS_TO_MSH = 4
ARG_SHOW_BOX = 5
ARG_IS_MESH_INVISIBLE = 6
ARG_INPUT_FOLDER_OR_OBJ_FILE_PATH = 7
ARG_OUTPUT_FOLDER_PATH = 8
ARG_PNG_FILENAME = 9

if len(sys.argv) < BULK_ARG_COUNT:
    print("Wrong arg count!")
    print_help()
    exit(0)

arg1 = sys.argv[ARG_BULK_OR_SINGLE].strip()
if arg1 not in ["-b", "-s"]:
    print("Arg1 must be -b or -s")
    print_help()
    exit(0)
is_bulk = arg1 == "-b"

mat_name = sys.argv[ARG_MATERIAL].strip().upper()
if mat_name not in mat_names:
    print(f"{mat_name} is not a valid material.")
    print_mat_help()
    exit(1)

jsonConfigPath = os.path.abspath(sys.argv[ARG_CONFIG_PATH].strip())
if not os.path.isfile(jsonConfigPath):
    print("Provided config path either doesn't exist or is a folder")
    exit(1)
with open(jsonConfigPath, "r") as file:
    try:
        configData = json.load(file)

        #print(f"{do_point_to_mesh}")

        obj_location = (configData["object"]["location"]["x"], configData["object"]["location"]["y"], configData["object"]["location"]["z"])
        obj_rotation = (configData["object"]["rotation"]["x"], configData["object"]["rotation"]["y"], configData["object"]["rotation"]["z"])
        obj_scale = (configData["object"]["scale"]["x"], configData["object"]["scale"]["y"], configData["object"]["scale"]["z"])

        #print(f"{obj_location}, {obj_rotation}, {obj_scale}")

        box_location = (configData["box"]["location"]["x"], configData["box"]["location"]["y"], configData["box"]["location"]["z"])
        box_rotation = (configData["box"]["rotation"]["x"], configData["box"]["rotation"]["y"], configData["box"]["rotation"]["z"])
        box_scale = (configData["box"]["scale"]["x"], configData["box"]["scale"]["y"], configData["box"]["scale"]["z"])

        cam_location = (configData["camera"]["location"]["x"], configData["camera"]["location"]["y"], configData["camera"]["location"]["z"])
        cam_rotation = (configData["camera"]["rotation"]["x"], configData["camera"]["rotation"]["y"], configData["camera"]["rotation"]["z"])

        #print(f"{cam_location}, {cam_rotation}, {cam_scale}")

        points_to_volume_density = configData["pointsToVolumeDensity"]
        points_to_volume_voxel_amount = configData["pointsToVolumeVoxelAmount"]
        points_to_volume_radius = configData["pointsToVolumeRadius"]

        #print(f"{points_to_volume_radius}, {points_to_volume_density}, {points_to_volume_voxel_amount}")

    except Exception as e:
        print("Exception thrown:")
        print(e)
        print()
        print("Provided JSON file is either invalid JSON or doesn't contain the correct fields")
        print()
        print_config_help()
        print()
        exit(1)

try:
    do_point_to_mesh = int(sys.argv[ARG_PTS_TO_MSH].strip()) != 0
    do_box = int(sys.argv[ARG_SHOW_BOX].strip()) != 0
    is_plane_invisible = int(sys.argv[ARG_IS_MESH_INVISIBLE].strip()) != 0
except Exception:
    print()
    print_convert_help()
    print()
    exit(1)

if is_bulk:
    if len(sys.argv) != BULK_ARG_COUNT:
        print_help()
        exit(1)

    inputFolderPath = os.path.abspath(sys.argv[ARG_INPUT_FOLDER_OR_OBJ_FILE_PATH].strip())
    outputFolderPath = os.path.abspath(sys.argv[ARG_OUTPUT_FOLDER_PATH].strip())

    if not all(os.path.isdir(path) for path in (inputFolderPath, outputFolderPath)):
        print("One or more paths either don't exist or aren't a folder!")
        exit(1)

    # Duct tape code to only render some frames in a folder if the frames are in the form i.obj for some integer i
    # procPairs = []
    # for f in os.listdir(inputFolderPath):
    #     if int(f.split(".")[0]) % 100 == 0:
    #         procPairs.append((os.path.join(inputFolderPath, f), os.path.join(outputFolderPath, f + ".png")))

    procPairs = [(os.path.join(inputFolderPath, f),
                  os.path.join(outputFolderPath, f + ".png")) for f in os.listdir(inputFolderPath)]

else:
    if len(sys.argv) != SINGLE_ARG_COUNT:
        print_help()
        exit(1)

    MESHPATH = os.path.abspath(sys.argv[ARG_INPUT_FOLDER_OR_OBJ_FILE_PATH].strip())
    if not os.path.isfile(MESHPATH):
        print("Provided mesh path doesn't exist or doesn't refer to a file")
        exit(1)
    OUTPUTPATH = os.path.abspath(sys.argv[ARG_OUTPUT_FOLDER_PATH].strip())
    # print(os.listdir(OUTPUTPATH))
    if not os.path.isdir(OUTPUTPATH):
        print("Output folder is not a valid path")
        exit(1)
    OUTPUTPATH = os.path.join(OUTPUTPATH, sys.argv[ARG_PNG_FILENAME].strip())
    # print(OUTPUTPATH)
    #
    # exit(0)

    procPairs = [(MESHPATH, OUTPUTPATH)]

cwd = os.getcwd()

for meshPath, outputPath in procPairs:

    ## initialize blender
    imgRes_x = 1000
    imgRes_y = 1000
    numSamples = 200
    exposure = 1.5
    custom_mats.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

    ## read mesh (choose either readPLY or readOBJ)

    location = obj_location
    rotation = obj_rotation
    scale = obj_scale

    if do_box:
        box_mesh = custom_mats.readMesh("./Box.obj", box_location, box_rotation, box_scale)
        custom_mats.setMatGlassBox(box_mesh)

    mesh = custom_mats.readMesh(meshPath, location, rotation, scale)

    pointsToVolDensity = points_to_volume_density
    pointsToVolVoxelAmt = points_to_volume_voxel_amount
    pointsToVolRadius = points_to_volume_radius

    if mat_name == mat_names[0]:
        mat = custom_mats.setMatPlasticine(mesh)
        custom_mats.pointCloudToSmoothMesh(mesh, mat, True,
                                           pointsToVolDensity=pointsToVolDensity,
                                           pointsToVolRadius=pointsToVolRadius,
                                           pointsToVolVoxelAmt=pointsToVolVoxelAmt,
                                           doConvertPointsToMesh=do_point_to_mesh)
    elif mat_name == mat_names[2]:
        mat = custom_mats.setMatElastic(mesh)
        custom_mats.pointCloudToSmoothMesh(mesh, mat, True,
                                           pointsToVolDensity=pointsToVolDensity,
                                           pointsToVolRadius=pointsToVolRadius,
                                           pointsToVolVoxelAmt=pointsToVolVoxelAmt,
                                           doConvertPointsToMesh=do_point_to_mesh)
    elif mat_name == mat_names[3]:
        mat = custom_mats.setMatSand(mesh)
        custom_mats.pointCloudToSmoothMesh(mesh, mat, False,
                                           smooth_modifier_factor=1.218,
                                           smooth_modifier_iters=2,
                                           pointsToVolDensity=pointsToVolDensity,
                                           pointsToVolRadius=pointsToVolRadius,
                                           pointsToVolVoxelAmt=pointsToVolVoxelAmt,
                                           doConvertPointsToMesh=do_point_to_mesh)
    elif mat_name == mat_names[1]:
        mat = custom_mats.setMatWater(mesh)
        custom_mats.pointCloudToSmoothMesh(mesh, mat, False,
                                           pointsToVolDensity=pointsToVolDensity,
                                           pointsToVolRadius=pointsToVolRadius,
                                           pointsToVolVoxelAmt=pointsToVolVoxelAmt,
                                           doConvertPointsToMesh=do_point_to_mesh)

    elif mat_name == mat_names[4]:
        mat = custom_mats.setMatPlastic(mesh, (144.0/255, 210.0/255, 236.0/255, 1, 0.5, 1.0, 1.0, 0.0, 2.0))
        custom_mats.pointCloudToSmoothMesh(mesh, mat, True,
                                           pointsToVolDensity=pointsToVolDensity,
                                           pointsToVolRadius=pointsToVolRadius,
                                           pointsToVolVoxelAmt=pointsToVolVoxelAmt,
                                           doConvertPointsToMesh=do_point_to_mesh)

    elif mat_name == mat_names[5]:
        mat = custom_mats.setMatChocolate(mesh)
        custom_mats.pointCloudToSmoothMesh(mesh, mat, True,
                                           pointsToVolDensity=pointsToVolDensity,
                                           pointsToVolRadius=pointsToVolRadius,
                                           pointsToVolVoxelAmt=pointsToVolVoxelAmt,
                                           doConvertPointsToMesh=do_point_to_mesh)

    bpy.context.scene.world.color = (0.6, 0.6, 0.6)

    ## Generate ground
    custom_mats.generateWhiteGroundPlane(location=(-0.56205, -0.78926, 0),
                                         rotation=(0, 0, 0),
                                         scale=(0.068, 0.068, 1),
                                         is_plane_invisible=is_plane_invisible)

    ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)

    camLocation = cam_location
    lookAtLocation = None
    focalLength = 45
    rotation = ((cam_rotation[0] * 2 * math.pi) / 360, (cam_rotation[1] * 2 * math.pi) / 360, (cam_rotation[2] * 2 * math.pi) / 360)

    cam = custom_mats.setCamera(camLocation, lookAtLocation, focalLength, rotation)

    ## set light
    custom_mats.createLightSun(rotation_euler=(338.54, -2148.8, 7004.7),
                               energy=4.540,
                               strength=10,
                               sun_angle=0.32637656)
    # custom_mats.createLightPoint(angle=(0.99034, -38.498, 126.22),
    #                              loc=(-1.1469, -1.5602, 2.8694),
    #                              strength=1200)

    ## set ambient light
    custom_mats.setLight_ambient(color=(0.1, 0.1, 0.1, 1))

    ## set gray shadow to completely white with a threshold
    custom_mats.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

    ## save blender file so that you can adjust parameters in the UI
    if not is_bulk:
        bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

    ## save rendering
    custom_mats.renderImage(outputPath, cam)

    if is_bulk:
        print("==================================================\n\n\n\n\n\n\n\n\n\n")

# FFMPEG convert images to video:
# C:\Users\Badlek\Downloads\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe -f image2 -i "./outputs/pred/pred%01d.0.obj.png" -vf format=yuv420p -r 30 -vcodec libx264 -b:v 9600k "./PRED.mp4"
