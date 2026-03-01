import bpy
import numpy as np
import mathutils
import os


class colorObj(object):
    def __init__(self, RGBA, \
    H = 0.5, S = 1.0, V = 1.0,\
    B = 0.0, C = 0.0):
        self.H = H # hue
        self.S = S # saturation
        self.V = V # value
        self.RGBA = RGBA
        self.B = B # birghtness
        self.C = C # contrast


def initColorNode(tree, color, xloc = [200,400], yloc = [0,0]):
    HSV = tree.nodes.new('ShaderNodeHueSaturation')
    HSV.inputs['Color'].default_value = color.RGBA
    HSV.inputs['Saturation'].default_value = color.S
    HSV.inputs['Value'].default_value = color.V
    HSV.inputs['Hue'].default_value = color.H
    HSV.location.x -= xloc[0]
    HSV.location.y -= yloc[0]

    BS = tree.nodes.new('ShaderNodeBrightContrast')
    BS.inputs['Bright'].default_value = color.B
    BS.inputs['Contrast'].default_value = color.C
    BS.location.x -= xloc[1]
    BS.location.y -= yloc[1]
    tree.links.new(HSV.outputs['Color'], BS.inputs['Color'])

    return BS


def blenderInit(resolution_x, resolution_y, numSamples = 128, exposure = 1.5, use_GPU = True, resolution_percentage = 100):
    # clear all
    bpy.ops.wm.read_homefile()
    bpy.ops.object.select_all(action = 'SELECT')
    bpy.ops.object.delete()
    # use cycle
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    # bpy.context.scene.cycles.film_transparent = True
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.cycles.samples = numSamples
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.film_exposure = exposure
    bpy.context.scene.render.resolution_percentage = resolution_percentage

    # Denoising
    # Note: currently I stop denoising as it will also denoise the alpha shadow channel. TODO: implement blurring shadow in the composite node
    bpy.data.scenes[0].view_layers[0]['cycles']['use_denoising'] = 0
    # bpy.data.scenes[0].view_layers[0]['cycles']['use_denoising'] = 1

    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    # # set devices
    # cyclePref  = bpy.context.preferences.addons['cycles'].preferences
    # for dev in cyclePref.devices:
    # 	print("using rendering device", dev.name, ":", dev.use)
    # if use_GPU:
    # 	bpy.context.scene.cycles.device = "GPU"
    # else:
    # 	bpy.context.scene.cycles.device = "CPU"
    # print("cycles rendering with:", bpy.context.scene.cycles.device)
    return 0


def readPLY(filePath, location, rotation_euler, scale):
    # example input types:
    # - location = (0.5, -0.5, 0)
    # - rotation_euler = (90, 0, 0)
    # - scale = (1,1,1)
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi
    angle = (x,y,z)
    prev = []
    for ii in range(len(list(bpy.data.objects))):
        prev.append(bpy.data.objects[ii].name)
    bpy.ops.wm.ply_import(filepath=filePath)
    after = []
    for ii in range(len(list(bpy.data.objects))):
        after.append(bpy.data.objects[ii].name)
    name = list(set(after) - set(prev))[0]
    # filePath = filePath.rstrip(os.sep)
    # name = os.path.basename(filePath)
    # name = name.replace('.ply', '')
    mesh = bpy.data.objects[name]
    # print(list(bpy.data.objects))
    # mesh = bpy.data.objects[-1]
    mesh.location = location
    mesh.rotation_euler = angle
    mesh.scale = scale
    bpy.context.view_layer.update()
    return mesh


def readOBJ(filePath, location, rotation_euler, scale):
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi
    angle = (x,y,z)

    prev = []
    for ii in range(len(list(bpy.data.objects))):
        prev.append(bpy.data.objects[ii].name)
    bpy.ops.wm.obj_import(filepath=filePath, use_split_groups=False)
    after = []
    for ii in range(len(list(bpy.data.objects))):
        after.append(bpy.data.objects[ii].name)
    name = list(set(after) - set(prev))[0]
    mesh = bpy.data.objects[name]

    mesh.location = location
    mesh.rotation_euler = angle
    mesh.scale = scale
    bpy.context.view_layer.update()

    return mesh


def readSTL(filePath, location, rotation_euler, scale):
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi
    angle = (x,y,z)

    prev = []
    for ii in range(len(list(bpy.data.objects))):
        prev.append(bpy.data.objects[ii].name)
    bpy.ops.wm.stl_import(filepath=filePath)
    after = []
    for ii in range(len(list(bpy.data.objects))):
        after.append(bpy.data.objects[ii].name)
    name = list(set(after) - set(prev))[0]
    mesh = bpy.data.objects[name]

    mesh.location = location
    mesh.rotation_euler = angle
    mesh.scale = scale
    bpy.context.view_layer.update()

    return mesh


def readMesh(filePath, location, rotation_euler, scale):
    _, extension = os.path.splitext(filePath)
    if extension == '.ply' or extension == '.PLY':
        mesh = readPLY(filePath, location, rotation_euler, scale)
    elif extension == '.obj' or extension == '.OBJ':
        mesh = readOBJ(filePath, location, rotation_euler, scale)
    elif extension == '.stl' or extension == '.STL':
        mesh = readSTL(filePath, location, rotation_euler, scale)
    else:
        raise TypeError("only support .ply, .obj, and .stl for now")
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.shade_flat() # defaiult flat shading
    return mesh


def setLight_ambient(color = (0,0,0,1)):
    bpy.data.scenes[0].world.use_nodes = True
    bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = color


def shadowThreshold(alphaThreshold, interpolationMode = 'CARDINAL'):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    RAMP = tree.nodes.new('CompositorNodeValToRGB')
    RAMP.color_ramp.elements[0].color[3] = 0
    RAMP.color_ramp.elements[0].position = alphaThreshold
    RAMP.color_ramp.interpolation = interpolationMode

    REND = tree.nodes["Render Layers"]
    OUT = tree.nodes["Composite"]
    tree.links.new(REND.outputs[1], RAMP.inputs[0])
    tree.links.new(RAMP.outputs[1], OUT.inputs[1])


def renderImage(outputPath, camera):
    bpy.data.scenes['Scene'].render.filepath = outputPath
    bpy.data.scenes['Scene'].camera = camera
    bpy.ops.render.render(write_still = True)


def lookAt(camera, point):
    direction = point - camera.location
    rotQuat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rotQuat.to_euler()


def setCamera(camLocation, lookAtLocation = (0,0,0), focalLength = 35, rotation=None):
    # initialize camera
    if rotation is not None:
        bpy.ops.object.camera_add(location=camLocation, rotation=rotation) # name 'Camera'
    else:
        bpy.ops.object.camera_add(location=camLocation)
    cam = bpy.context.object
    cam.data.lens = focalLength
    if rotation is None:
        print("this ran")
        loc = mathutils.Vector(lookAtLocation)
        lookAt(cam, loc)
    return cam


def setMatElastic(mesh):
    mat = bpy.data.materials.new('Elastic')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # init color node
    BCNode = initColorNode(tree, colorObj((1.0, 0, 0, 1), 0.5, 1, 0.3, 0.3, 0.4))

    # construct amber node
    fresnel = tree.nodes.new('ShaderNodeFresnel')

    maprange = tree.nodes.new('ShaderNodeMapRange')
    maprange.inputs['From Min'].default_value = 0.0
    maprange.inputs['From Max'].default_value = 1.0
    maprange.inputs['To Min'].default_value = 1.250
    maprange.inputs['To Max'].default_value = 3.0

    lightpath = tree.nodes.new('ShaderNodeLightPath')

    tree.links.new(lightpath.outputs['Is Diffuse Ray'], maprange.inputs['Value'])
    tree.links.new(maprange.outputs['Result'], fresnel.inputs['IOR'])

    glass = tree.nodes.new('ShaderNodeBsdfGlass')
    glass.inputs['Roughness'].default_value = 0
    glass.inputs['IOR'].default_value = 1.2
    tree.links.new(BCNode.outputs['Color'], glass.inputs['Color'])

    transparent = tree.nodes.new('ShaderNodeBsdfTransparent')
    tree.links.new(BCNode.outputs['Color'], transparent.inputs['Color'])

    multiply = tree.nodes.new('ShaderNodeMixRGB')
    multiply.blend_type = 'MULTIPLY'
    multiply.inputs['Fac'].default_value = 1
    multiply.inputs['Color2'].default_value = (1, 1, 1, 1)
    tree.links.new(fresnel.outputs['Fac'], multiply.inputs['Color1'])

    mix1 = tree.nodes.new('ShaderNodeMixShader')
    mix1.inputs['Fac'].default_value = 0.708
    tree.links.new(glass.outputs[0], mix1.inputs[1])
    tree.links.new(transparent.outputs[0], mix1.inputs[2])

    glossy = tree.nodes.new('ShaderNodeBsdfGlossy')
    glossy.inputs['Color'].default_value = (0.613, 0.109, 0.099, 1)
    glossy.inputs['Roughness'].default_value = 1.0
    glossy.inputs['Anisotropy'].default_value = 1.0
    glossy.inputs['Rotation'].default_value = 0.669

    mix2 = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(multiply.outputs[0], mix2.inputs[0])
    tree.links.new(mix1.outputs[0], mix2.inputs[1])
    tree.links.new(glossy.outputs[0], mix2.inputs[2])

    tree.links.new(mix2.outputs[0], tree.nodes['Material Output'].inputs['Surface'])

    return mat


def setMatWater(mesh):
    mat = bpy.data.materials.new('Water')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # Clear all default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create nodes
    glossy_bsdf = tree.nodes.new('ShaderNodeBsdfGlossy')
    glossy_bsdf.location = (-300, 200)
    glossy_bsdf.inputs['Roughness'].default_value = 0.246
    glossy_bsdf.inputs['Anisotropy'].default_value = 1.0
    glossy_bsdf.inputs['Rotation'].default_value = 0.669
    glossy_bsdf.inputs['Color'].default_value = (0.175, 0.511, 0.806, 1)

    fresnel = tree.nodes.new('ShaderNodeFresnel')
    fresnel.location = (-500, 400)
    fresnel.inputs['IOR'].default_value = 1.33

    map_range = tree.nodes.new('ShaderNodeMapRange')
    map_range.location = (-300, 0)
    map_range.inputs['From Min'].default_value = 0.0
    map_range.inputs['From Max'].default_value = 1.0
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 0.96
    map_range.clamp = True

    transparent_bsdf = tree.nodes.new('ShaderNodeBsdfTransparent')
    transparent_bsdf.inputs['Color'].default_value = (0.875, 0.950, 1, 1)
    transparent_bsdf.location = (-300, -200)

    glass_bsdf = tree.nodes.new('ShaderNodeBsdfGlass')
    glass_bsdf.location = (-300, -400)
    glass_bsdf.inputs['Color'].default_value = (0.412, 0.708, 1, 1)
    glass_bsdf.inputs['Roughness'].default_value = 0.0
    glass_bsdf.inputs['IOR'].default_value = 1.33

    mix_shader1 = tree.nodes.new('ShaderNodeMixShader')
    mix_shader1.location = (0, -250)

    mix_shader2 = tree.nodes.new('ShaderNodeMixShader')
    mix_shader2.location = (0, -200)
    mix_shader2.inputs['Fac'].default_value = 0.935

    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = (0, -400)
    emission.inputs['Color'].default_value = (0, 0.276, 1, 1)
    emission.inputs['Strength'].default_value = 0.4

    mix_shader3 = tree.nodes.new('ShaderNodeMixShader')
    mix_shader3.location = (200, 0)
    mix_shader3.inputs['Fac'].default_value = 0.075

    material_output = tree.nodes.new('ShaderNodeOutputMaterial')
    material_output.location = (400, 0)

    # Create links
    links = tree.links
    links.new(fresnel.outputs['Fac'], map_range.inputs['Value'])

    links.new(map_range.outputs['Result'], mix_shader1.inputs[0])
    links.new(transparent_bsdf.outputs['BSDF'], mix_shader1.inputs[1])
    links.new(glass_bsdf.outputs['BSDF'], mix_shader1.inputs[2])

    links.new(glossy_bsdf.outputs['BSDF'], mix_shader2.inputs[1])
    links.new(mix_shader1.outputs['Shader'], mix_shader2.inputs[2])

    links.new(emission.outputs['Emission'], mix_shader3.inputs[2])
    links.new(mix_shader2.outputs['Shader'], mix_shader3.inputs[1])

    links.new(mix_shader3.outputs['Shader'], material_output.inputs['Surface'])

    return mat


def setMatPlasticine(mesh):
    meshColor = colorObj((0.404, 0.801, 0.310, 1.0), 0.5, 1.0, 1.0, 0, 0)

    mat = bpy.data.materials.new('Plasticine')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.3
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
    tree.nodes["Principled BSDF"].inputs['Specular IOR Level'].default_value = 0.5
    tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
    tree.nodes["Principled BSDF"].inputs['Transmission Weight'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Coat Roughness'].default_value = 0

    # add Ambient Occlusion
    tree.nodes.new('ShaderNodeAmbientOcclusion')
    tree.nodes.new('ShaderNodeGamma')
    MIXRGB = tree.nodes.new('ShaderNodeMixRGB')
    MIXRGB.blend_type = 'MULTIPLY'
    tree.nodes["Gamma"].inputs["Gamma"].default_value = 6.0
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
    tree.nodes["Gamma"].location.x -= 600

    # set color using Hue/Saturation node
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    HSVNode.inputs['Color'].default_value = meshColor.RGBA
    HSVNode.inputs['Saturation'].default_value = meshColor.S
    HSVNode.inputs['Value'].default_value = meshColor.V
    HSVNode.inputs['Hue'].default_value = meshColor.H
    HSVNode.location.x -= 200

    # set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = meshColor.B
    BCNode.inputs['Contrast'].default_value = meshColor.C
    BCNode.location.x -= 400

    # link all the nodes
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], MIXRGB.inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], MIXRGB.inputs['Color2'])
    tree.links.new(MIXRGB.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    return mat


def setMatSand(mesh):
    mat = bpy.data.materials.new('Sand')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # Clear all existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create Nodes
    tex_coord = tree.nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)

    mapping = tree.nodes.new('ShaderNodeMapping')
    mapping.location = (-600, 0)
    mapping.inputs['Scale'].default_value = (1.0, 1.0, 1.0)

    wave_texture = tree.nodes.new('ShaderNodeTexWave')
    wave_texture.location = (-400, 200)
    wave_texture.inputs['Scale'].default_value = 4.5
    wave_texture.inputs['Distortion'].default_value = 8.3
    wave_texture.inputs['Detail'].default_value = 0.8
    wave_texture.inputs['Detail Scale'].default_value = -1.0
    wave_texture.inputs['Detail Roughness'].default_value = 0
    wave_texture.inputs['Phase Offset'].default_value = 8.3

    noise_texture = tree.nodes.new('ShaderNodeTexNoise')
    noise_texture.location = (-400, -200)
    noise_texture.inputs['Scale'].default_value = 125.1
    noise_texture.inputs['Detail'].default_value = 14.9
    noise_texture.inputs['Roughness'].default_value = 0.908
    noise_texture.inputs['Lacunarity'].default_value = 2.0
    noise_texture.inputs['Distortion'].default_value = 0.0

    mix_rgb = tree.nodes.new('ShaderNodeMix')
    mix_rgb.location = (-200, 100)
    mix_rgb.blend_type = 'MIX'
    mix_rgb.inputs['Factor'].default_value = 0.65

    bump = tree.nodes.new('ShaderNodeBump')
    bump.location = (0, -100)
    bump.inputs['Strength'].default_value = 0.608
    bump.inputs['Distance'].default_value = 1.0

    principled_bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
    principled_bsdf.location = (200, 0)
    principled_bsdf.inputs['Base Color'].default_value = (.570, .3, .011, 1)
    principled_bsdf.inputs['Roughness'].default_value = 0.0
    principled_bsdf.inputs['IOR'].default_value = 1.5
    principled_bsdf.inputs['Specular IOR Level'].default_value = 0.345  # Specular control

    material_output = tree.nodes.new('ShaderNodeOutputMaterial')
    material_output.location = (400, 0)

    # Create Links
    links = tree.links

    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])

    links.new(mapping.outputs['Vector'], wave_texture.inputs['Vector'])
    links.new(mapping.outputs['Vector'], noise_texture.inputs['Vector'])

    links.new(wave_texture.outputs['Fac'], mix_rgb.inputs[1])
    links.new(noise_texture.outputs['Fac'], mix_rgb.inputs[2])

    links.new(mix_rgb.outputs['Result'], bump.inputs['Height'])

    links.new(bump.outputs['Normal'], principled_bsdf.inputs['Normal'])
    links.new(mix_rgb.outputs['Result'], principled_bsdf.inputs['Roughness'])

    links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    return mat


def setMatChocolate(mesh):
    mat = bpy.data.materials.new('Chocolate')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # Clear all existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create Nodes
    tex_coord = tree.nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)

    layer_weight = tree.nodes.new('ShaderNodeLayerWeight')
    layer_weight.location = (-600, 200)
    layer_weight.inputs['Blend'].default_value = 0.5  # Blend Factor

    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.outputs[0].default_value = (0.013, 0.006, 0.003, 1)
    rgb.location = (-600, -100)

    brightness_contrast = tree.nodes.new('ShaderNodeBrightContrast')
    brightness_contrast.location = (-400, -100)
    brightness_contrast.inputs['Bright'].default_value = 0.0
    brightness_contrast.inputs['Contrast'].default_value = 0.0

    diffuse_bsdf = tree.nodes.new('ShaderNodeBsdfDiffuse')
    diffuse_bsdf.location = (-200, 100)
    diffuse_bsdf.inputs['Roughness'].default_value = 0.0

    glossy_bsdf = tree.nodes.new('ShaderNodeBsdfGlossy')
    glossy_bsdf.location = (-200, -100)
    glossy_bsdf.inputs['Roughness'].default_value = 0.2

    mix_shader = tree.nodes.new('ShaderNodeMixShader')
    mix_shader.location = (200, 0)

    noise_texture = tree.nodes.new('ShaderNodeTexNoise')
    noise_texture.location = (-800, -300)
    noise_texture.inputs['Scale'].default_value = 1.5
    noise_texture.inputs['Detail'].default_value = 2.0
    noise_texture.inputs['Roughness'].default_value = 0.440
    noise_texture.inputs['Lacunarity'].default_value = 2.0
    noise_texture.inputs['Distortion'].default_value = 0.0

    rgb_to_bw = tree.nodes.new('ShaderNodeRGBToBW')
    rgb_to_bw.location = (-600, -300)

    bump = tree.nodes.new('ShaderNodeBump')
    bump.location = (-400, -300)
    bump.inputs['Strength'].default_value = 0.5
    bump.inputs['Distance'].default_value = 0

    material_output = tree.nodes.new('ShaderNodeOutputMaterial')
    material_output.location = (400, 0)

    # Create Links
    links = tree.links

    links.new(tex_coord.outputs['Normal'], layer_weight.inputs['Normal'])

    links.new(layer_weight.outputs['Facing'], mix_shader.inputs['Fac'])

    links.new(rgb.outputs['Color'], brightness_contrast.inputs['Color'])

    links.new(rgb.outputs['Color'], diffuse_bsdf.inputs['Color'])
    links.new(brightness_contrast.outputs['Color'], glossy_bsdf.inputs['Color'])

    links.new(noise_texture.outputs['Color'], rgb_to_bw.inputs['Color'])

    links.new(rgb_to_bw.outputs['Val'], bump.inputs['Height'])

    links.new(bump.outputs['Normal'], glossy_bsdf.inputs['Normal'])
    links.new(bump.outputs['Normal'], diffuse_bsdf.inputs['Normal'])

    links.new(diffuse_bsdf.outputs['BSDF'], mix_shader.inputs[1])
    links.new(glossy_bsdf.outputs['BSDF'], mix_shader.inputs[2])

    links.new(mix_shader.outputs['Shader'], material_output.inputs['Surface'])

    return mat


def setMatPlastic(mesh, meshColor, AOStrength=0.0):
    mat = bpy.data.materials.new('Plastic')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.3
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
    tree.nodes["Principled BSDF"].inputs['Specular IOR Level'].default_value = 0.5
    tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
    tree.nodes["Principled BSDF"].inputs['Transmission Weight'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Coat Roughness'].default_value = 0

    # add Ambient Occlusion
    tree.nodes.new('ShaderNodeAmbientOcclusion')
    tree.nodes.new('ShaderNodeGamma')
    MIXRGB = tree.nodes.new('ShaderNodeMixRGB')
    MIXRGB.blend_type = 'MULTIPLY'
    tree.nodes["Gamma"].inputs["Gamma"].default_value = AOStrength
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
    tree.nodes["Gamma"].location.x -= 600

    # set color using Hue/Saturation node
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    HSVNode.inputs['Color'].default_value = meshColor[:4]
    HSVNode.inputs['Saturation'].default_value = meshColor[4]
    HSVNode.inputs['Value'].default_value = meshColor[5]
    HSVNode.inputs['Hue'].default_value = meshColor[6]
    HSVNode.location.x -= 200

    # set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = meshColor[7]
    BCNode.inputs['Contrast'].default_value = meshColor[8]
    BCNode.location.x -= 400

    # link all the nodes
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], MIXRGB.inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], MIXRGB.inputs['Color2'])
    tree.links.new(MIXRGB.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    return mat


def setMatGlassBox(mesh):
    mat = bpy.data.materials.new("GlassBox")
    mesh.data.materials.clear()
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    bsdf = tree.nodes.new('ShaderNodeBsdfGlass')
    bsdf.inputs['Roughness'].default_value = 0
    bsdf.inputs['IOR'].default_value = 1.2

    material_output = tree.nodes.get('Material Output')
    tree.links.new(bsdf.outputs[0], material_output.inputs['Surface'])

    return mat


def pointCloudToSmoothMesh(mesh, mat, is_smooth_shading, smooth_modifier_factor=2.418, smooth_modifier_iters=7,
                           pointsToVolDensity=3.0, pointsToVolVoxelAmt=170, pointsToVolRadius=0.007, doConvertPointsToMesh=True):
    """
    :param mesh: bpy mesh object on which to apply the geometry node and smooth modifier
    :param mat: bpy reference to material which should be injected into the material node of the geometry nodes
    :param is_smooth_shading: apply smooth shading
    :param smooth_modifier_factor: smooth modifier smoothing modifier
    :param smooth_modifier_iters: iterations of the smooth modifier
    :param pointsToVolDensity: Density of the voxels when converting to mesh
    :param pointsToVolVoxelAmt: Number of voxels when converting to mesh
    :param pointsToVolRadius: Radius of voxels when converting to mesh
    :param doConvertPointsToMesh: If True, removes the step to convert points to mesh
        (assumes provided mesh is already a mesh rather than a point cloud). This means
        the previous 3 params become irrelevant
    """
    # turn a point cloud into a smooth mesh
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

    bpy.ops.object.modifier_add(type='NODES')
    bpy.ops.node.new_geometry_nodes_modifier()
    tree = mesh.modifiers[-1].node_group

    IN = tree.nodes['Group Input']
    OUT = tree.nodes['Group Output']

    SETSMOOTHSHADING = tree.nodes.new('GeometryNodeSetShadeSmooth')

    MATERIAL = tree.nodes.new('GeometryNodeSetMaterial')
    MATERIAL.inputs[2].default_value = mat

    if doConvertPointsToMesh:
        MESH2POINT = tree.nodes.new('GeometryNodeMeshToPoints')
        MESH2POINT.location.x -= 100
        MESH2POINT.inputs['Radius'].default_value = 1

        POINTS2VOLUME = tree.nodes.new('GeometryNodePointsToVolume')
        POINTS2VOLUME.resolution_mode = 'VOXEL_AMOUNT'
        POINTS2VOLUME.inputs['Density'].default_value = pointsToVolDensity
        POINTS2VOLUME.inputs['Voxel Amount'].default_value = pointsToVolVoxelAmt
        POINTS2VOLUME.inputs['Radius'].default_value = pointsToVolRadius

        VOLUME2MESH = tree.nodes.new('GeometryNodeVolumeToMesh')
        VOLUME2MESH.resolution_mode = 'GRID'
        VOLUME2MESH.inputs['Threshold'].default_value = 0.1
        VOLUME2MESH.inputs['Adaptivity'].default_value = 0.0

        tree.links.new(IN.outputs['Geometry'], MESH2POINT.inputs['Mesh'])
        tree.links.new(MESH2POINT.outputs['Points'], POINTS2VOLUME.inputs['Points'])
        tree.links.new(POINTS2VOLUME.outputs['Volume'], VOLUME2MESH.inputs['Volume'])

        if is_smooth_shading:
            tree.links.new(VOLUME2MESH.outputs['Mesh'], SETSMOOTHSHADING.inputs['Geometry'])
            tree.links.new(SETSMOOTHSHADING.outputs['Geometry'], MATERIAL.inputs['Geometry'])

        else:
            tree.links.new(VOLUME2MESH.outputs['Mesh'], MATERIAL.inputs['Geometry'])

    else:
        if is_smooth_shading:
            tree.links.new(IN.outputs["Geometry"], SETSMOOTHSHADING.inputs['Geometry'])
            tree.links.new(SETSMOOTHSHADING.outputs['Geometry'], MATERIAL.inputs['Geometry'])
        else:
            tree.links.new(IN.outputs["Geometry"], MATERIAL.inputs['Geometry'])

    tree.links.new(MATERIAL.outputs['Geometry'], OUT.inputs['Geometry'])

    if doConvertPointsToMesh:
        bpy.ops.object.modifier_add(type="SMOOTH")
        smooth = mesh.modifiers[-1]
        smooth.factor = smooth_modifier_factor
        smooth.iterations = smooth_modifier_iters


def createLightPoint(angle, loc, strength):
    bpy.ops.object.light_add(type='POINT', rotation=angle, location=loc)
    lamp = bpy.data.lights['Point']
    lamp.use_nodes = True

    lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength
    return lamp


def createLightSun(rotation_euler, energy, strength, sun_angle):
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi
    angle = (x, y, z)
    bpy.ops.object.light_add(type='SUN', rotation=angle)
    lamp = bpy.data.lights['Sun']
    lamp.use_nodes = True
    lamp.energy = energy
    lamp.angle = sun_angle

    lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength
    return lamp


def generateWhiteGroundPlane(location, rotation, scale, is_plane_invisible=False):
    #bpy.context.scene.cycles.film_transparent = True

    bpy.ops.mesh.primitive_plane_add(location=location, rotation=rotation, scale=scale, size=100)
    plane = bpy.context.object
    if is_plane_invisible:
        plane.is_shadow_catcher = True

    mat = bpy.data.materials.new('GroundMaterial')
    plane.data.materials.append(mat)
    plane.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    PRINC_REFLECT = tree.nodes.new("ShaderNodeBsdfPrincipled")
    PRINC_SHAD = tree.nodes.new("ShaderNodeBsdfPrincipled")

    PRINC_SHAD.inputs["Base Color"].default_value = (0.352, 0.352, 0.352, 1)
    PRINC_SHAD.inputs["Metallic"].default_value = 0.0
    PRINC_SHAD.inputs["Roughness"].default_value = 1.0
    PRINC_SHAD.inputs["IOR"].default_value = 3.2
    PRINC_SHAD.inputs["Alpha"].default_value = 1.0

    PRINC_SHAD.inputs["Subsurface Weight"].default_value = 1.0
    PRINC_SHAD.inputs["Subsurface Radius"].default_value = (13.6, 9.9, 5.4)
    PRINC_SHAD.inputs["Subsurface Scale"].default_value = 3.85
    PRINC_SHAD.inputs["Subsurface Anisotropy"].default_value = 0.432

    PRINC_SHAD.inputs["Specular IOR Level"].default_value = 0
    PRINC_SHAD.inputs["Specular Tint"].default_value = (0.986, 1.0, 0.880, 1.0)
    PRINC_SHAD.inputs["Anisotropic"].default_value = 0.623
    PRINC_SHAD.inputs["Anisotropic Rotation"].default_value = 1.0

    PRINC_SHAD.inputs["Transmission Weight"].default_value = 1
    PRINC_SHAD.inputs["Coat Weight"].default_value = 0
    PRINC_SHAD.inputs["Sheen Weight"].default_value = 0

    PRINC_SHAD.inputs["Emission Color"].default_value = (0.799, 0.908, 1, 1)
    PRINC_SHAD.inputs["Emission Strength"].default_value = 0.3

    # ----------------------------------------------------

    PRINC_REFLECT.inputs["Base Color"].default_value = (0.352, 0.352, 0.352, 1)
    PRINC_REFLECT.inputs["Metallic"].default_value = 0.0
    PRINC_REFLECT.inputs["Roughness"].default_value = 0.168
    PRINC_REFLECT.inputs["IOR"].default_value = 30.0
    PRINC_REFLECT.inputs["Alpha"].default_value = 1.0

    PRINC_REFLECT.inputs["Subsurface Weight"].default_value = 0.0
    PRINC_REFLECT.inputs["Subsurface Radius"].default_value = (13.6, 9.9, 5.4)
    PRINC_REFLECT.inputs["Subsurface Scale"].default_value = 3.85
    PRINC_REFLECT.inputs["Subsurface Anisotropy"].default_value = 0.432

    PRINC_REFLECT.inputs["Specular IOR Level"].default_value = 0
    PRINC_REFLECT.inputs["Specular Tint"].default_value = (0.986, 1.0, 0.880, 1.0)
    PRINC_REFLECT.inputs["Anisotropic"].default_value = 0.623
    PRINC_REFLECT.inputs["Anisotropic Rotation"].default_value = 1.0

    PRINC_REFLECT.inputs["Transmission Weight"].default_value = 1
    PRINC_REFLECT.inputs["Coat Weight"].default_value = 0
    PRINC_REFLECT.inputs["Sheen Weight"].default_value = 0

    PRINC_REFLECT.inputs["Emission Color"].default_value = (0.799, 0.908, 1, 1)
    PRINC_REFLECT.inputs["Emission Strength"].default_value = 0.3

    # ----------------------------------------------------

    mix = tree.nodes.new('ShaderNodeMixShader')
    mix.inputs["Fac"].default_value = 0.75
    tree.links.new(PRINC_SHAD.outputs['BSDF'], mix.inputs[1])
    tree.links.new(PRINC_REFLECT.outputs['BSDF'], mix.inputs[2])

    tree.links.new(mix.outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])
