import mmcv
import os 
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

config = "/Users/arvindkruthiventy/mmdetection/configs/selfsup_pretrain/mask_rcnn_r50_fpn_mocov2-pretrain_1x_coco.py"
checkpoint = "/Users/arvindkruthiventy/mmdetection/research/checkpoints/mocov2_r50_800ep_pretrain.pth"
device = "mps"
config = mmcv.Config.fromfile(config)
model = build_detector(config.model)
checkpoint = load_checkpoint(model, checkpoint, map_location=device)
#model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()
img = '/Users/arvindkruthiventy/mmdetection/demo/demo.jpg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)