import sys
import os

# Ensure the root directory is in sys.path so we can dynamically resolve paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from PIL import Image, ImageDraw, ImageFont, ImageOps

# ==========================================
# CONFIGURATION
# ==========================================
# We assume your raw Blender renders are in online_rom/outputs/
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRID_OUT_DIR = os.path.join(PROJECT_ROOT, "visualizations", "media", "comparison_grids")
os.makedirs(GRID_OUT_DIR, exist_ok=True)

# Layout Settings
IMG_W = 512
IMG_H = 512
PADDING_X = 80       
PADDING_Y = 60       
HEADER_H = 120       
LABEL_W = 450        

# Styling
CORNER_RADIUS = 40   
SEPARATOR_THICKNESS = 8
SEPARATOR_PADDING = 50

# Dotted Box Settings
DOT_WIDTH = 10       
DOT_LENGTH = 20      
DOT_GAP = 20         

# Models
MODELS = [
    "pca", "colora", "coral", "crom", "dino", "gno", "licrom", 
    "giorom"
]

MODEL_LABELS = {
    "pca": "GROUND TRUTH",
    "colora": "COLORA",
    "coral": "CORAL",
    "crom": "CROM",
    "dino": "DINO",
    "giorom": "GIOROM (OURS)",
    "gno": "GKI",
    "licrom": "LICROM"
}

DATASET_CONFIG = {
    "nclaw_Plasticine": [5, 50, 100, 150, 199],
    "nclaw_Water":      [5, 50, 100, 131, 150],
    "owl":              [5, 11, 22, 44, 60],
    "nclaw_Sand":       [5, 50, 100, 150, 199]
}

# Font Loading (Robust Fallback)
POSSIBLE_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"
]

FONT = ImageFont.load_default()
for font_path in POSSIBLE_FONTS:
    if os.path.exists(font_path):
        try:
            FONT = ImageFont.truetype(font_path, 55) 
            print(f"Loaded Font: {font_path}")
            break
        except: continue

# ==========================================
# HELPERS
# ==========================================
def get_image_path_robust(model, dataset, frame_idx):
    render_dir = os.path.join(BASE_OUTPUT_DIR, model, f"{dataset}_pred", "rendered")
    # Priority list for filenames
    candidates = [
        f"pred_{frame_idx:04d}.png",
        f"pred_{frame_idx:04d}.obj.png",
        f"pred{frame_idx:d}.obj.png",
        f"pred{frame_idx:d}.png"
    ]
    for fname in candidates:
        path = os.path.join(render_dir, fname)
        if os.path.exists(path): return path
    return os.path.join(render_dir, candidates[0])

def round_corners(img, radius):
    """ Rounds the corners of an image using a mask """
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), img.size], radius=radius, fill=255)
    output = Image.new('RGBA', img.size, (255, 255, 255, 0))
    output.paste(img, (0, 0), mask=mask)
    bg = Image.new('RGB', img.size, (255, 255, 255))
    bg.paste(output, mask=output.split()[3])
    return bg

def draw_dotted_rect(draw, bbox, color, width, dash_len, gap_len):
    x0, y0, x1, y1 = bbox
    # Top & Bottom
    for x in range(int(x0), int(x1), dash_len + gap_len):
        draw.line([(x, y0), (min(x + dash_len, x1), y0)], fill=color, width=width)
        draw.line([(x, y1), (min(x + dash_len, x1), y1)], fill=color, width=width)
    # Left & Right
    for y in range(int(y0), int(y1), dash_len + gap_len):
        draw.line([(x0, y), (x0, min(y + dash_len, y1))], fill=color, width=width)
        draw.line([(x1, y), (x1, min(y + dash_len, y1))], fill=color, width=width)

# ==========================================
# MAIN LOGIC
# ==========================================
def create_comparison_grid(dataset, frames):
    print(f"--- Processing Grid: {dataset} ---")
    
    rows = len(MODELS)
    cols = len(frames)
    
    total_h = HEADER_H + (rows * IMG_H) + ((rows + 1) * PADDING_Y) + (SEPARATOR_PADDING * 2)
    total_w = LABEL_W + (cols * IMG_W) + ((cols + 1) * PADDING_X)
    
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 1. Column Headers (Timesteps)
    for i, frame in enumerate(frames):
        cx = LABEL_W + PADDING_X + i * (IMG_W + PADDING_X) + (IMG_W // 2)
        cy = HEADER_H // 2
        text = f"t={frame}" # Added 't=' for clarity in the paper
        bbox = draw.textbbox((0, 0), text, font=FONT)
        draw.text((cx - (bbox[2]-bbox[0])//2, cy - (bbox[3]-bbox[1])//2), text, fill="black", font=FONT)

    current_y = HEADER_H + PADDING_Y
    
    for row_idx, model in enumerate(MODELS):
        
        # --- SEPARATOR FOR GIOROM ---
        if model == "giorom":
            line_y = current_y + SEPARATOR_PADDING // 2
            draw.line([(PADDING_X, line_y), (total_w - PADDING_X, line_y)], fill="black", width=SEPARATOR_THICKNESS)
            current_y += SEPARATOR_PADDING * 2
        
        # --- ROW LABEL ---
        label_text = MODEL_LABELS.get(model, model).upper()
        row_center_y = current_y + (IMG_H // 2)
        
        bbox = draw.textbbox((0,0), label_text, font=FONT)
        txt_h = bbox[3]-bbox[1]
        draw.text((30, row_center_y - txt_h//2), label_text, fill="black", font=FONT)
        
        # --- DOTTED BOX (Ground Truth) ---
        if model == "pca":
            box_x0 = LABEL_W + (PADDING_X // 2)
            box_y0 = current_y - (PADDING_Y // 2)
            box_x1 = total_w - (PADDING_X // 2)
            box_y1 = current_y + IMG_H + (PADDING_Y // 2)
            draw_dotted_rect(draw, (box_x0, box_y0, box_x1, box_y1), "black", DOT_WIDTH, DOT_LENGTH, DOT_GAP)
        
        # --- IMAGES ---
        for col_idx, frame in enumerate(frames):
            img_path = get_image_path_robust(model, dataset, frame)
            x_pos = LABEL_W + PADDING_X + col_idx * (IMG_W + PADDING_X)
            y_pos = current_y
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    
                    # FIXED SAND ROTATION LOGIC
                    if dataset == "nclaw_Sand":
                        img = img.transpose(Image.Transpose.ROTATE_270)
                    
                    img = img.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
                    img = round_corners(img, CORNER_RADIUS)
                    
                    canvas.paste(img, (x_pos, y_pos))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            else:
                draw.rectangle([x_pos, y_pos, x_pos+IMG_W, y_pos+IMG_H], fill=(240, 240, 240))
                draw.text((x_pos+50, y_pos+50), "MISSING", fill="red", font=FONT)
        
        current_y += IMG_H + PADDING_Y

    out_path = os.path.join(GRID_OUT_DIR, f"grid_{dataset}_polished.png")
    canvas.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    print(f"Reading renders from: {BASE_OUTPUT_DIR}")
    for ds_name, frame_list in DATASET_CONFIG.items():
        create_comparison_grid(ds_name, frame_list)