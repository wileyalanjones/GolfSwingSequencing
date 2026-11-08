from golfswing.model import load_trained_model
from golfswing.predict import TestVideo, compute_event_frames, extract_frames
import cv2

VIDEO = "data/videos_160/0.mp4"
MODEL = "models/weight_norm_loss_64_model_epoch_01_val_event_auc_pr0.3311.keras"
model = load_trained_model(MODEL)
video = TestVideo(VIDEO)
event_frames = compute_event_frames(model, video.frames)
print(event_frames)
images = extract_frames(VIDEO, event_frames)
for i, img in enumerate(images, start=1):
    cv2.imwrite(f"event_{i}.jpg", img)


