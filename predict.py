"""Inference on the held-out temporal test block; save original-size binary PNGs."""
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from common import ROOT,configuration,load_model,output,read_csv,record,seed_all,sha256,write_csv,write_json

def predict_array(model, rgb, cfg, device):
    small=np.asarray(rgb.resize((cfg["width"],cfg["height"]),Image.Resampling.BILINEAR),dtype=np.float32)/255
    tensor=torch.from_numpy(small.transpose(2,0,1).copy()[None]).to(device)
    probability=model(tensor).sigmoid()
    probability=F.interpolate(probability,size=(rgb.height,rgb.width),mode="bilinear",align_corners=False)
    return (probability[0,0]>=cfg["probability_threshold"]).to(torch.uint8).cpu().numpy()*255

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="config.json")
    cfg=configuration(parser.parse_args().config);seed_all(cfg["seed"])
    out=output(cfg);record(cfg,"predict","started");start=time.perf_counter()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,checkpoint=load_model(cfg,device)
    rows=[r for r in read_csv(ROOT/"data/manifest.csv") if r["split"]=="test"]
    masks=out/"masks"
    if masks.exists() and any(masks.glob("*.png")):raise FileExistsError("Predictions already exist for this run")
    masks.mkdir(exist_ok=True);manifest=[]
    with torch.inference_mode():
        for row in rows:
            with Image.open(ROOT/row["image_path"]) as im:
                mask=predict_array(model,im.convert("RGB"),cfg,device)
            path=masks/Path(row["basename"]).with_suffix(".png")
            Image.fromarray(mask).save(path)
            with Image.open(path) as saved:
                array=np.asarray(saved)
                if saved.size!=(1280,720) or saved.mode!="L" or not set(np.unique(array)).issubset({0,255}):
                    raise ValueError("Invalid saved prediction mask")
            manifest.append({"basename":row["basename"],"image_path":row["image_path"],
                             "prediction_path":path.relative_to(ROOT).as_posix(),"prediction_sha256":sha256(path)})
    assert len(manifest)==400
    write_csv(out/"prediction-manifest.csv",manifest)
    summary={"test_images":len(manifest),"checkpoint_hash":sha256(checkpoint),"mask_size":[1280,720],
             "probability_threshold":cfg["probability_threshold"],"resize_policy":"bilinear probabilities before threshold; align_corners=False",
             "device":str(device),"elapsed_seconds":time.perf_counter()-start}
    write_json(out/"inference-summary.json",summary);record(cfg,"predict",**summary);print(summary)

if __name__=="__main__":main()
