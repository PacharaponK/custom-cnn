"""Evaluate saved PNGs with pixel-wise IoU, not box IoU or model confidence."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,configuration,output,read_csv,record,write_csv,write_json

def pixel_iou(prediction,ground_truth):
    if prediction.shape!=ground_truth.shape:raise ValueError("Mask dimensions differ")
    p=prediction.astype(bool);g=ground_truth.astype(bool)
    intersection=int((p&g).sum());union=int((p|g).sum())
    return intersection,union,intersection/union if union else 1.

def summarize(rows,threshold):
    detected=[r["iou"] for r in rows if r["iou"]>=threshold]
    strict=[r["iou"] for r in rows if r["iou"]>threshold]
    values=[r["iou"] for r in rows]
    return {"test_images":len(rows),"detected_ge_0_6":len(detected),"detected_gt_0_6":len(strict),
            "exact_threshold_images":sum(x==threshold for x in values),
            "detected_rate":len(detected)/len(rows),"mean_iou_detected":float(np.mean(detected)) if detected else None,
            "mean_iou_all":float(np.mean(values)),"median_iou":float(np.median(values)),
            "min_iou":min(values),"max_iou":max(values),"empty_empty_images":sum(r["union_pixels"]==0 for r in rows),
            "threshold":threshold,"main_comparator":">=","empty_empty_policy":"IoU=1, count explicitly",
            "detection_flag_meaning":"GT-based post-inference success, not model confidence"}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="config.json")
    cfg=configuration(parser.parse_args().config);out=output(cfg);record(cfg,"evaluate","started")
    rows=[r for r in read_csv(ROOT/"data/manifest.csv") if r["split"]=="test"]
    saved=read_csv(out/"prediction-manifest.csv");index={r["basename"]:r for r in saved}
    if set(index)!={r["basename"] for r in rows}:raise ValueError("Prediction/test manifest mismatch")
    results=[]
    for row in rows:
        with Image.open(ROOT/index[row["basename"]]["prediction_path"]) as im:p=np.asarray(im).copy()
        with Image.open(ROOT/row["mask_path"]) as im:g=np.asarray(im).copy()
        if not set(np.unique(p)).issubset({0,255}):raise ValueError("Predicted PNG is not binary")
        intersection,union,iou=pixel_iou(p,g)
        results.append({"basename":row["basename"],"intersection_pixels":intersection,"union_pixels":union,"iou":iou,
                        "detected_ge_0_6":iou>=cfg["detection_iou_threshold"],
                        "detected_gt_0_6":iou>cfg["detection_iou_threshold"]})
    summary=summarize(results,cfg["detection_iou_threshold"])
    write_csv(out/"evaluation-per-image.csv",results);write_json(out/"evaluation-summary.json",summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4),dpi=180)
    ax.hist([r["iou"] for r in results],bins=np.linspace(0,1,21),color="#227c9d")
    ax.axvline(.6,color="#b23a48",linestyle="--",label="Detection threshold 0.6")
    ax.set(xlabel="Lane pixel IoU",ylabel="Test image count",title="Held-out test IoU | "+cfg["run_id"]);ax.legend()
    fig.tight_layout();fig.savefig(out/"figures/test-iou-histogram.png");plt.close(fig)
    # Two high, two nearest threshold and two low examples, without duplicate sample IDs.
    ordered=sorted(results,key=lambda r:r["iou"])
    chosen=[];seen=set()
    for kind,group in [("high",ordered[::-1]),("near-threshold",sorted(results,key=lambda r:abs(r["iou"]-.6))),("low",ordered)]:
        count=0
        for r in group:
            if r["basename"] not in seen:
                chosen.append((kind,r));seen.add(r["basename"]);count+=1
                if count==2:break
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",18) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    snapshots=[]
    row_index={r["basename"]:r for r in rows}
    for kind,result in chosen:
        row=row_index[result["basename"]]
        with Image.open(ROOT/row["image_path"]) as im:rgb=im.convert("RGB")
        with Image.open(ROOT/row["mask_path"]) as im:gt=im.copy()
        with Image.open(ROOT/index[row["basename"]]["prediction_path"]) as im:pred=im.copy()
        overlay=Image.composite(Image.blend(rgb,Image.new("RGB",rgb.size,(0,255,0)),.35),rgb,pred)
        canvas=Image.new("RGB",(1280,800),"white");draw=ImageDraw.Draw(canvas)
        for j,(title,im) in enumerate([("RGB",rgb),("Ground truth",gt.convert("RGB")),("Prediction",pred.convert("RGB")),("Prediction overlay",overlay)]):
            x=(j%2)*640;y=(j//2)*395
            draw.text((x+10,y+5),title,font=font,fill="black")
            canvas.paste(im.resize((640,360)),(x,y+30))
        path=out/"figures"/f"snapshot-{kind}-{Path(row['basename']).stem}.png";canvas.save(path)
        snapshots.append({"kind":kind,"basename":row["basename"],"iou":result["iou"],
                          "detected":result["detected_ge_0_6"],"figure":path.relative_to(ROOT).as_posix()})
    write_json(out/"snapshots.json",snapshots);record(cfg,"evaluate",**summary);print(summary)

if __name__=="__main__":main()
