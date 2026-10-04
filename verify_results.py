"""Fresh-process consistency check of supplied checkpoint, 400 PNGs and reported metrics."""
import numpy as np
from PIL import Image
import torch
from common import ROOT,configuration,load_model,output,read_csv,read_json,record,seed_all,sha256,write_json
from evaluation import pixel_iou,summarize
from predict import predict_array

def main():
    cfg=configuration();seed_all(cfg["seed"]);out=output(cfg)
    manifest=read_csv(ROOT/"data/manifest.csv")
    train={r["sha256"] for r in manifest if r["split"]=="train"}
    test={r["sha256"] for r in manifest if r["split"]=="test"}
    assert len(train)==743 and len(test)==400 and not train & test
    metric_rows=read_csv(out/"metrics.csv")
    assert len(metric_rows)==30 and all(int(r["samples"])==743 for r in metric_rows)
    history=read_json(out/"training-summary.json")
    checkpoint=ROOT/"checkpoints"/cfg["run_id"]/"last.pt"
    assert sha256(checkpoint)==history["checkpoint_hash"]
    assert sha256(ROOT/"checkpoints"/cfg["run_id"]/"initial.pt")==history["initial_hash"]
    byname={r["basename"]:r for r in manifest}
    results=[]
    prediction_manifest=read_csv(out/"prediction-manifest.csv")
    for row in prediction_manifest:
        path=ROOT/row["prediction_path"]
        assert sha256(path)==row["prediction_sha256"]
        with Image.open(path) as im:
            assert im.size==(1280,720) and im.mode=="L"
            p=np.asarray(im).copy()
        with Image.open(ROOT/byname[row["basename"]]["mask_path"]) as im:g=np.asarray(im).copy()
        intersection,union,iou=pixel_iou(p,g)
        results.append({"basename":row["basename"],"intersection_pixels":intersection,"union_pixels":union,"iou":iou})
    assert len(results)==400
    recomputed=summarize(results,.6);saved=read_json(out/"evaluation-summary.json")
    assert recomputed==saved
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,_=load_model(cfg,device)
    exact=[]
    with torch.inference_mode():
        for row in [prediction_manifest[0],prediction_manifest[-1]]:
            source=byname[row["basename"]]
            with Image.open(ROOT/source["image_path"]) as im:prediction=predict_array(model,im.convert("RGB"),cfg,device)
            with Image.open(ROOT/row["prediction_path"]) as im:expected=np.asarray(im)
            assert np.array_equal(prediction,expected),row["basename"]
            exact.append(row["basename"])
    result={"passed":True,"test_masks_recomputed":400,"checkpoint_reload_prediction_matches":exact,
            "train_test_hash_overlap":0,"epochs":30,"environment_lock_hash":sha256(ROOT/"uv.lock")}
    write_json(out/"verification.json",result);record(cfg,"verify_results",**result);print(result)

if __name__=="__main__":main()
