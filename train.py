"""Smoke check or final >=30-epoch training with TensorBoard and CSV."""
import argparse
import importlib.metadata
import json
import platform
import random
import subprocess
import time

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from common import ROOT, configuration, now, output, record, seed_all, sha256, write_csv, write_json
from model import CompactUNet

class LaneDataset(Dataset):
    def __init__(self, cfg, augment=True):
        saved = np.load(ROOT / "data/train-small.npz", allow_pickle=False)
        self.images, self.masks = saved["images"], saved["masks"]
        self.aug = cfg["augmentation"]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].astype(np.float32)/255
        if self.augment:
            a = self.aug
            if random.random()<a["white_balance_probability"]:
                image *= np.array([random.uniform(*a["white_balance_range"]) for _ in range(3)],dtype=np.float32)
            if random.random()<a["brightness_probability"]:
                image *= random.uniform(*a["brightness_range"])
            image=np.clip(image,0,1)
            if random.random()<a["blur_probability"]:
                im=Image.fromarray((image*255).astype(np.uint8))
                image=np.asarray(im.filter(ImageFilter.GaussianBlur(random.uniform(*a["blur_radius_range"]))),dtype=np.float32)/255
        return torch.from_numpy(image.transpose(2,0,1).copy()),torch.from_numpy(self.masks[index][None].astype(np.float32))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="config.json")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    cfg=configuration(args.config);seed_all(cfg["seed"])
    out=output(cfg);record(cfg,"smoke" if args.smoke else "train","started")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=CompactUNet().to(device)
    criterion=torch.nn.BCEWithLogitsLoss()
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==29761
    dataset=LaneDataset(cfg,augment=not args.smoke)
    x=torch.from_numpy(dataset.images[:8].astype(np.float32).transpose(0,3,1,2)/255).to(device)
    y=torch.from_numpy(dataset.masks[:8,None].astype(np.float32)).to(device)
    assert model(x).shape==y.shape==(8,1,36,64)
    if args.smoke:
        opt=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
        first=float(criterion(model(x),y).detach())
        for step in range(100):
            opt.zero_grad(set_to_none=True)
            loss=criterion(model(x),y)
            if not torch.isfinite(loss):raise ValueError("Nonfinite smoke loss")
            loss.backward();opt.step()
        last=float(criterion(model(x),y).detach())
        if last>=first*.8:raise ValueError("Overfit check did not lower loss enough")
        result={"device":str(device),"samples":8,"steps":100,"initial_loss":first,"final_loss":last,
                "parameters":parameters,"passed":True,"weights_reused_for_final":False}
        write_json(out/"smoke-check.json",result)
        record(cfg,"smoke",**result);print(result);return
    if cfg["epochs"]<30:raise ValueError("Final training requires >=30 epochs")
    checkpoint_dir=ROOT/"checkpoints"/cfg["run_id"]
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    if (checkpoint_dir/"last.pt").exists():raise FileExistsError("Final run already exists; use a new run id")
    torch.save(model.state_dict(),checkpoint_dir/"initial.pt")
    initial_hash=sha256(checkpoint_dir/"initial.pt")
    loader=DataLoader(dataset,batch_size=cfg["batch_size"],shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(cfg["seed"]),pin_memory=device.type=="cuda")
    opt=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    writer=SummaryWriter(str(ROOT/"runs"/cfg["run_id"]))
    history=[];start=time.perf_counter();global_step=0
    versions={name:importlib.metadata.version(name) for name in ["torch","numpy","pillow","tensorboard","matplotlib","psutil"]}
    environment={"python":platform.python_version(),"platform":platform.platform(),"uv":subprocess.check_output(["uv","--version"],text=True).strip(),
                 "packages":versions,"cuda_available":torch.cuda.is_available(),"cuda_runtime":torch.version.cuda,
                 "device":str(device),"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,
                 "cpu":platform.processor(),"logical_cpu_count":__import__("os").cpu_count(),
                 "ram_total_bytes":__import__("psutil").virtual_memory().total,
                 "lockfile_hash":sha256(ROOT/"uv.lock"),"started_at":now()}
    write_json(out/"environment.json",environment)
    log_path=out/"logs/training.log"
    for epoch in range(1,cfg["epochs"]+1):
        model.train();sum_loss=0.;count=0;intersection=0;union=0;epoch_start=time.perf_counter()
        for x,y in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y)
            if not torch.isfinite(loss):raise ValueError("Nonfinite final training loss")
            loss.backward();opt.step();n=x.shape[0];sum_loss+=float(loss.detach())*n;count+=n
            pred=logits.detach().sigmoid()>=cfg["probability_threshold"];gt=y.bool()
            intersection+=int((pred&gt).sum());union+=int((pred|gt).sum())
            writer.add_scalar("Loss/train_step",float(loss.detach()),global_step)
            global_step+=1
        if device.type=="cuda":torch.cuda.synchronize()
        row={"epoch":epoch,"train_loss":sum_loss/count,"train_pixel_iou":intersection/union if union else 1.,
             "learning_rate":opt.param_groups[0]["lr"],"samples":count,"epoch_seconds":time.perf_counter()-epoch_start}
        history.append(row)
        for tag,value in [("Loss/train",row["train_loss"]),("IoU/train",row["train_pixel_iou"]),("LR/train",row["learning_rate"])]:
            writer.add_scalar(tag,value,epoch)
        write_csv(out/"metrics.csv",history)
        torch.save({"state_dict":model.state_dict(),"epoch":epoch,"global_step":global_step,"config":cfg,
                    "manifest_hash":sha256(ROOT/"data/manifest.csv"),"initial_hash":initial_hash},checkpoint_dir/"last.pt")
        message=json.dumps(row);print(message,flush=True)
        with log_path.open("a",encoding="utf-8") as stream:stream.write(message+"\n")
        if epoch%5==0:
            model.eval()
            with torch.inference_mode():
                sample=torch.from_numpy(dataset.images[0].astype(np.float32).transpose(2,0,1)[None]/255).to(device)
                preview=(model(sample).sigmoid()[0,0].cpu().numpy()>=cfg["probability_threshold"]).astype(np.uint8)*255
            Image.fromarray(preview).resize((640,360),Image.Resampling.NEAREST).save(out/"figures"/f"train-preview-epoch-{epoch:02d}.png")
        writer.flush()
    writer.close()
    summary={"epochs":cfg["epochs"],"samples_per_epoch":len(dataset),"global_steps":global_step,
             "parameters":parameters,"parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
             "from_scratch":True,"pretrained_weights_loaded":False,"checkpoint_policy":"final epoch; no test selection",
             "initial_hash":initial_hash,"checkpoint_hash":sha256(checkpoint_dir/"last.pt"),"elapsed_seconds":time.perf_counter()-start,
             "initial_epoch_loss":history[0]["train_loss"],"final_epoch_loss":history[-1]["train_loss"],"completed_at":now()}
    write_json(out/"training-summary.json",summary);record(cfg,"train",**summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for field,title,ylabel in [("train_loss","Train BCE loss","Loss"),("train_pixel_iou","Train pixel IoU","IoU"),("learning_rate","Learning rate","LR")]:
        fig,ax=plt.subplots(figsize=(9,4),dpi=180);ax.plot([r["epoch"] for r in history],[r[field] for r in history])
        ax.set(xlabel="Epoch",ylabel=ylabel,title=title+" | "+cfg["run_id"])
        fig.tight_layout();fig.savefig(out/"figures"/f"{field}.png");plt.close(fig)
    # Show the exact photometric augmentation pipeline on the first train sample.
    preview=Image.new("RGB",(768,265),"white")
    from PIL import ImageDraw
    preview.paste(Image.fromarray(dataset.images[0]).resize((384,216)),(0,25))
    augmented=dataset[0][0].numpy().transpose(1,2,0)
    preview.paste(Image.fromarray((augmented*255).astype(np.uint8)).resize((384,216)),(384,25))
    ImageDraw.Draw(preview).text((10,5),"Train sample: original / random photometric augmentation",fill="black")
    preview.save(out/"figures/augmentation-example.png")
    print(summary)

if __name__=="__main__":
    main()
