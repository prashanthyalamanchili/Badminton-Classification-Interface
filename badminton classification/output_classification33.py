import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from model_package import load_model, MatchChunkDataset, resize_tensor, normalize_clip

# -----------------------------------
# Device
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# Paths
# -----------------------------------
VIDEO_PATH = "/data/psxpy5/dataset/classification1.mp4"
OUTPUT_PATH = "/data/psxpy5/dataset/output_stage1_stage2_stage3.mp4"

MODEL_CFG = "/data/psxpy5/dataset/i3d_resnet50_v1_kinetics400.yaml"

# Stage 1 model: rally vs non-rally
RVSNR_MODEL_PATH = "/data/psxpy5/dataset/trained_model_rvsnr_2.pth"

# Stage 2 model: non-rally event classifier
NONRALLY_MODEL_PATH = "/data/psxpy5/dataset/cm/model_9_epoch_9.pth"

# Stage 3 model: shot classifier
SHOT_MODEL_PATH = "/data/psxpy5/dataset/shot_model_12.pth"  

CHUNK_FRAMES = 64
SAMPLE_FRAMES = 64
STAGE1_CONF_THRESHOLD = 0.70
BATCH_SIZE = 6

CI_THRESHOLD = 0.85
SH_THRESHOLD = 0.75

STAGE1_CLASSES = {
    0: "NON-RALLY",
    1: "RALLY"
}

STAGE2_CLASSES = {
    0: "INTERVAL",
    1: "SH",
    2: "SET BREAK",
    3: "COACH INTERACTION"
}

SHOT_CLASSES = {
    0: "SHORT SERVE",
    1: "LIFT",
    2: "DROP SHOT",
    3: "PUSH SHOT",
    4: "CUT",
    5: "CLEAR",
    6: "LONG SERVE"
}

shot_transform = transforms.Compose([
    transforms.Lambda(lambda x: resize_tensor(x, size=(224, 224))),
    transforms.Lambda(normalize_clip)
])

def load_checkpoint_model(cfg_path, checkpoint_path, num_classes):
    model = load_model(cfg_path, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model
def make_motion_clip(clip):
    motion = clip[:, :, 1:, :, :] - clip[:, :, :-1, :, :]
    motion = torch.cat([motion, motion[:, :, -1:, :, :].clone()], dim=2)
    return motion

stage1_model = load_checkpoint_model(MODEL_CFG, RVSNR_MODEL_PATH, num_classes=2)
stage2_model = load_checkpoint_model(MODEL_CFG, NONRALLY_MODEL_PATH, num_classes=4)
stage3_model = load_checkpoint_model(MODEL_CFG, SHOT_MODEL_PATH, num_classes=7)

print("Stage 1, Stage 2, Stage 3 models loaded")

dataset = MatchChunkDataset(
    video_path=VIDEO_PATH,
    chunk_frames=CHUNK_FRAMES,
    sample_frames=SAMPLE_FRAMES,
    resize=(224, 224)
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

predictions = []

with torch.no_grad():
    for clips, starts in loader:
        clips = clips.to(device)

        stage1_outputs = stage1_model(clips)
        stage1_probs = F.softmax(stage1_outputs, dim=1)
        stage1_preds = torch.argmax(stage1_probs, dim=1)

        for i in range(len(starts)):
            clip = clips[i].unsqueeze(0)   
            start_frame = starts[i].item()

            s1_pred = stage1_preds[i].item()
            s1_conf = stage1_probs[i][s1_pred].item()

            if s1_pred == 1 and s1_conf < STAGE1_CONF_THRESHOLD:
                s1_pred = 0
                s1_conf = stage1_probs[i][0].item()

            coarse_label = STAGE1_CLASSES[s1_pred]

            if s1_pred == 0:
                motion_clip = make_motion_clip(clip)

                stage2_outputs = stage2_model(motion_clip)
                stage2_probs = F.softmax(stage2_outputs, dim=1)
                stage2_pred = torch.argmax(stage2_probs, dim=1).item()
                stage2_conf = stage2_probs[0][stage2_pred].item()

                final_label = STAGE2_CLASSES[stage2_pred]

                if final_label == "COACH INTERACTION" and stage2_conf < CI_THRESHOLD:
                    final_label = "INTERVAL"

                if final_label == "SH" and stage2_conf < SH_THRESHOLD:
                    final_label = "INTERVAL"

                final_conf = stage2_conf

                print(f"[Frame {start_frame}] Stage 1: {coarse_label}")
                print(f"[Frame {start_frame}] Stage 2: {final_label} ({final_conf:.4f})")

            else:
                shot_clip = clip.clone()
                shot_clip = shot_transform(shot_clip.squeeze(0)).unsqueeze(0)

                stage3_outputs = stage3_model(shot_clip)
                stage3_probs = F.softmax(stage3_outputs, dim=1)
                stage3_pred = torch.argmax(stage3_probs, dim=1).item()
                stage3_conf = stage3_probs[0][stage3_pred].item()

                final_label = SHOT_CLASSES[stage3_pred]
                final_conf = stage3_conf

                print(f"[Frame {start_frame}] Stage 1: {coarse_label}")
                print(f"[Frame {start_frame}] Stage 3: {final_label} ({final_conf:.4f})")

            predictions.append({
                "start": start_frame,
                "coarse_label": coarse_label,
                "final_label": final_label,
                "confidence": final_conf
            })

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

current_chunk = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if (current_chunk + 1 < len(predictions) and
        frame_idx >= predictions[current_chunk + 1]["start"]):
        current_chunk += 1

    pred = predictions[current_chunk]

    coarse_text = pred["coarse_label"]
    final_text = pred["final_label"]
    conf_text = f"{pred['confidence']:.2f}"

    if coarse_text == "RALLY":
        color1 = (0, 255, 0)      
        color2 = (255, 255, 0)    
    else:
        color1 = (0, 0, 255)      
        color2 = (0, 255, 255)    

    cv2.rectangle(frame, (20, 20), (900, 140), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Stage 1: {coarse_text}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color1,
        3,
        cv2.LINE_AA
    )

    if coarse_text == "NON-RALLY":
        second_line = f"Stage 2: {final_text} ({conf_text})"
    else:
        second_line = f"Stage 3: {final_text} ({conf_text})"

    cv2.putText(
        frame,
        second_line,
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color2,
        3,
        cv2.LINE_AA
    )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f" Output saved to: {OUTPUT_PATH}")
