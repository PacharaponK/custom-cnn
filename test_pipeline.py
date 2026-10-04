"""One runnable check for polygon conversion, threshold boundaries and model gradients."""
import numpy as np
import torch
from PIL import Image
from prepare_data import rasterize
from evaluation import pixel_iou,summarize
from model import CompactUNet

def main():
    mask=np.asarray(rasterize([[(1,1),(4,1),(4,4),(1,4)]],8,8))
    assert set(np.unique(mask))=={0,255} and mask[2,2]==255 and mask[0,0]==0
    gt=np.array([1,1,1,1,1]);pred=np.array([1,1,1,0,0])
    intersection,union,iou=pixel_iou(pred,gt)
    assert (intersection,union,iou)==(3,5,.6)
    summary=summarize([{"iou":iou,"union_pixels":5}],.6)
    assert summary["detected_ge_0_6"]==1 and summary["detected_gt_0_6"]==0
    assert pixel_iou(np.zeros(4),np.zeros(4))==(0,0,1.)
    no_detect=summarize([{"iou":0.,"union_pixels":5}],.6)
    assert no_detect["mean_iou_detected"] is None
    torch.set_num_threads(2)
    model=CompactUNet()
    assert sum(p.numel() for p in model.parameters())==29761
    for width,height in [(64,36),(48,48)]:
        x=torch.rand(2,3,height,width);y=torch.zeros(2,1,height,width)
        out=model(x);assert out.shape==y.shape
        loss=torch.nn.functional.binary_cross_entropy_with_logits(out,y);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        model.zero_grad()
    print("PASS polygon rasterization, exact IoU boundary, empty policies, model shapes and gradients")

if __name__=="__main__":main()
