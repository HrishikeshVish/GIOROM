import os
import json
import subprocess
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# --- DYNAMIC PATH CONFIGURATION ---
VIS_DIR = os.path.dirname(os.path.abspath(__file__))
BLENDER_EXE = os.environ.get("BLENDER_EXE", "blender")
RENDER_SCRIPT = os.path.join(VIS_DIR, "render_visuals.py")
RENDER_POINTS_SCRIPT = os.path.join(VIS_DIR, "render_points.py")

def save_obj_jax(vertices, frame_idx, output_dir, prefix="pred"):
    """
    Saves a point cloud/mesh frame as a standard .obj file.
    Supports both JAX arrays and PyTorch tensors (CPU or CUDA).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    filename = os.path.join(output_dir, f"{prefix}{frame_idx:04d}.obj")
    
    # --- [FIX] ROBUST TYPE CONVERSION ---
    # Handle PyTorch Tensors (including CUDA)
    if hasattr(vertices, "detach"):
        verts_np = vertices.detach().cpu().numpy()
    # Handle JAX/NumPy Arrays
    else:
        verts_np = np.array(vertices)
    # ------------------------------------
    
    with open(filename, 'w') as f:
        f.write(f"# Frame {frame_idx}\n")
        for v in verts_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

def run_render_pipeline(obj_dir, output_video_path, material):
    print(f"--- Rendering Video: {output_video_path} ---")
    render_dir = os.path.join(obj_dir, "rendered")
    os.makedirs(render_dir, exist_ok=True)
    
    if material in ["PLASTICINE", "ELASTIC", "WATER"]:
        cmd_blender = [
            BLENDER_EXE, "-b", "-P", RENDER_POINTS_SCRIPT, "--",
            material, obj_dir, render_dir
        ]
    else:
        config_path = os.path.join(obj_dir, "render_config.json")
        config_data = {
            "object": {"location": {"x": 0, "y": 0.0, "z": 0.0}, "rotation": {"x": 180, "y": 270, "z": 0}, "scale": {"x": 1, "y": 1, "z": 1}},
            "box": {"location": {"x": 0, "y": 1.0, "z": 1.0}, "rotation": {"x": 0, "y": 0, "z": 0}, "scale": {"x": 1, "y": 1, "z": 1}},
            "camera": {"location": {"x": 2.2, "y": -1.4, "z": 1.5}, "rotation": {"x": 60, "y": 0, "z": 60}},
            "pointsToVolumeDensity": 0.5, "pointsToVolumeVoxelAmount": 128, "pointsToVolumeRadius": 0.02
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
            
        cmd_blender = [
            BLENDER_EXE, "-b", "-P", RENDER_SCRIPT, "--",
            "-b", material, config_path, "1", "0", "0", obj_dir, render_dir
        ]
    
    try:
        subprocess.run(cmd_blender, check=True, stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"Error: Blender executable not found. Please ensure 'blender' is in your PATH or set the BLENDER_EXE environment variable.")
        return

    prefix = "pred" if "pred" in os.listdir(obj_dir)[0] else "gt"
    vf_string = "fps=24" if material in ["PLASTICINE", "ELASTIC", "WATER"] else "transpose=1"
    
    cmd_ffmpeg = [
        "ffmpeg", "-y", "-framerate", "30", "-start_number", "0",
        "-i", os.path.join(render_dir, f"{prefix}_%04d.obj.png"),
        "-vf", vf_string, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_video_path
    ]
    subprocess.run(cmd_ffmpeg, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved: {output_video_path}")

def generate_2d_video_stoc(params, video_traj, dmin, dmax, model, sparsity_factor, output_video="giorom_stochastic_demo.mp4"):
    frames_to_render = min(len(video_traj), 200) 
    print(f"\n--- Generating 2D Video (Stochastic) ---")
    
    video_key = jax.random.PRNGKey(100)

    @jax.jit
    def inference_step(x_q_ref, x_s_ref, x_s_curr, rng_key): 
        return model.apply(params, x_q_ref, x_s_ref, x_s_curr, rngs={'feynman_kac': rng_key})

    x_d_ref = jnp.array(video_traj[0])
    x_s_ref = x_d_ref[::int(sparsity_factor)]
    
    fig = plt.figure(figsize=(18, 7), facecolor='#0f0f0f')
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.1)
    
    titles = [f"Sparse Input (1/{sparsity_factor})", "Ground Truth", "GIOROM (Stochastic)"]
    colors = ['#00FFFF', '#00FF00', '#FF00FF']
    axes, scatters = [], []
    
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(titles[i], color=colors[i], fontsize=14, fontweight='bold')
        axes.append(ax)
        sc = ax.scatter([], [], s=1.0, alpha=0.6, c=colors[i], edgecolors='none')
        scatters.append(sc)

    text_ax = fig.add_subplot(gs[1, :])
    text_ax.set_facecolor('#0f0f0f'); text_ax.axis('off')
    hud_text = text_ax.text(0.5, 0.5, "Init...", ha='center', va='center', color='white', fontfamily='monospace', fontsize=12)

    def denorm(x): return x * (dmax - dmin) + dmin
    key_container = [video_key]

    def update(i):
        x_d_curr = jnp.array(video_traj[i])
        x_s_curr = x_d_curr[::int(sparsity_factor)]
        
        current_key, next_key = jax.random.split(key_container[0])
        key_container[0] = next_key
        
        pred = inference_step(x_d_ref, x_s_ref, x_s_curr, current_key)
        pred.block_until_ready()
        
        mse = jnp.mean((denorm(x_d_curr) - denorm(pred))**2)
        
        s_np = np.array(x_s_curr, dtype=np.float32)
        d_np = np.array(x_d_curr, dtype=np.float32)
        p_np = np.array(pred, dtype=np.float32)
        
        scatters[0].set_offsets(s_np[:, :2])
        scatters[1].set_offsets(d_np[:, :2])
        scatters[2].set_offsets(p_np[:, :2])
        
        hud_text.set_text(f"Frame: {i:03d} | MSE: {mse:.2e}")
        return scatters + [hud_text]

    ani = animation.FuncAnimation(fig, update, frames=tqdm(range(frames_to_render)), interval=1, blit=True)
    ani.save(output_video, fps=30, dpi=100, writer='ffmpeg')
    plt.close(fig)
    print("Video Done.")