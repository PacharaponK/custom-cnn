"""Read CVAT lane polygons, deduplicate, exclude, convert and persist the split."""
import argparse
import math
import re
import shutil
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, configuration, output, read_csv, record, sha256, write_csv, write_json

def rasterize(polygons, width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for points in polygons:
        clipped = [(min(width-1, max(0, x)), min(height-1, max(0, y))) for x, y in points]
        draw.polygon(clipped, fill=255)
    return mask

def parse_lane(element, width, height):
    polygons = []
    for shape in element.findall("polygon"):
        if shape.get("label") != "lane":
            continue
        points = [tuple(map(float, pair.split(","))) for pair in shape.get("points", "").split(";")]
        if len(points) < 3 or any(len(p) != 2 for p in points):
            raise ValueError("Invalid lane polygon")
        if any(not math.isfinite(v) for p in points for v in p):
            raise ValueError("Nonfinite lane coordinates")
        if any(x < 0 or x > width or y < 0 or y > height for x, y in points):
            raise ValueError("Lane polygon out of bounds")
        area = abs(sum(points[i][0]*points[(i+1)%len(points)][1] -
                       points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))) / 2
        if area <= 0:
            raise ValueError("Degenerate lane polygon")
        polygons.append(points)
    return polygons

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config.json")
    cfg = configuration(args.parse_args().config)
    start = time.perf_counter()
    record(cfg, "prepare_data", "started")
    source = (ROOT / cfg["source_dataset"]).resolve()
    dest = ROOT / "data"
    if (dest / "manifest.csv").exists():
        raise FileExistsError("Prepared manifest already exists; do not overwrite data silently")
    paths = defaultdict(list)
    for p in sorted(source.rglob("*.jpg")):
        paths[p.name].append(p)
    annotations = defaultdict(list)
    for xml in sorted(source.rglob("*.xml")):
        text = xml.read_text(encoding="utf-8")
        if "<!DOCTYPE" in text or "<!ENTITY" in text:
            raise ValueError("DTD/entity declarations are not accepted")
        for image in ET.fromstring(text).findall("image"):
            name = image.get("name")
            if Path(name).name != name or not re.fullmatch(r"frame_\d+_f\d+\.jpg", name):
                raise ValueError("Unexpected image basename")
            annotations[name].append((xml, image))
    rows, excluded, duplicate_rows = [], [], []
    tensors = {"train": ([], [], []), "test": ([], [], [])}
    selected_previews = []
    for name in sorted(paths, key=lambda n: int(re.search(r"frame_(\d+)", n).group(1))):
        frame = int(re.search(r"frame_(\d+)", name).group(1))
        candidates = paths[name]
        hashes = {sha256(p) for p in candidates}
        if len(hashes) != 1:
            raise ValueError(f"Same basename has differing images: {name}")
        for p in candidates[1:]:
            duplicate_rows.append({"basename": name, "duplicate_path": p.relative_to(source).as_posix(),
                                   "kept_path": candidates[0].relative_to(source).as_posix(), "sha256": next(iter(hashes))})
        if frame in cfg["excluded_frames"]:
            reason = "conflicting_gt" if frame in [400, 401] else "uncertain_lane_label"
            excluded.append({"basename": name, "frame": frame, "reason": reason})
            continue
        entries = annotations.get(name, [])
        if len(entries) != 1:
            raise ValueError(f"Expected one canonical annotation: {name}")
        xml, element = entries[0]
        width, height = int(element.get("width")), int(element.get("height"))
        polygons = parse_lane(element, width, height)
        if not polygons:
            raise ValueError(f"Unexpected empty lane target: {name}")
        image_path = candidates[0]
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            if im.size != (width, height):
                raise ValueError("XML/image dimensions mismatch")
            rgb = im.copy()
        gt = rasterize(polygons, width, height)
        yolo_rows = ["0 " + " ".join(f"{v:.8f}" for x,y in polygon for v in [x/width,y/height]) for polygon in polygons]
        roundtrip = []
        for line in yolo_rows:
            fields = line.split()
            values = list(map(float, fields[1:]))
            roundtrip.append([(round(values[i]*width, 2),round(values[i+1]*height, 2)) for i in range(0,len(values),2)])
        if not np.array_equal(np.asarray(gt), np.asarray(rasterize(roundtrip,width,height))):
            raise ValueError(f"XML/YOLO roundtrip mismatch: {name}")
        split = "train" if frame <= cfg["train_last_frame"] else "test"
        if split == "test" and frame < cfg["test_first_frame"]:
            raise ValueError("Unexpected intermediate frame")
        image_out = dest / "images" / split / name
        label_out = dest / "labels" / split / Path(name).with_suffix(".txt")
        mask_out = dest / "masks-original" / split / Path(name).with_suffix(".png")
        for p in [image_out,label_out,mask_out]: p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image_path,image_out)
        label_out.write_text("\n".join(yolo_rows)+"\n",encoding="utf-8")
        gt.save(mask_out)
        small_rgb = np.asarray(rgb.resize((cfg["width"],cfg["height"]),Image.Resampling.BILINEAR),dtype=np.uint8).copy()
        small_gt = (np.asarray(gt.resize((cfg["width"],cfg["height"]),Image.Resampling.NEAREST)) > 0).astype(np.uint8)
        tensors[split][0].append(small_rgb)
        tensors[split][1].append(small_gt)
        tensors[split][2].append(name)
        rows.append({"basename":name,"frame":frame,"split":split,"sha256":next(iter(hashes)),
                     "source_image":image_path.relative_to(source).as_posix(),"source_xml":xml.relative_to(source).as_posix(),
                     "image_path":image_out.relative_to(ROOT).as_posix(),"label_path":label_out.relative_to(ROOT).as_posix(),
                     "mask_path":mask_out.relative_to(ROOT).as_posix(),"lane_polygons":len(polygons),
                     "width":width,"height":height,"foreground_fraction":round(float(np.asarray(gt).mean()/255),6)})
        if split == "train" and frame in [2,100,200,300,399,450,500,600,700,749]:
            selected_previews.append((name,rgb,gt))
    if len(rows) != 1143 or len(excluded) != 5 or len(duplicate_rows) != 52:
        raise ValueError("Source counts differ from confirmed audit")
    for split, (images,masks,names) in tensors.items():
        np.savez_compressed(dest / f"{split}-small.npz", images=np.stack(images), masks=np.stack(masks), names=np.array(names))
    train_hashes={r["sha256"] for r in rows if r["split"]=="train"}
    test_hashes={r["sha256"] for r in rows if r["split"]=="test"}
    assert len(train_hashes)==743 and len(test_hashes)==400 and not train_hashes & test_hashes
    write_csv(dest/"manifest.csv",rows)
    write_csv(dest/"excluded-images.csv",excluded)
    write_csv(dest/"duplicate-copies.csv",duplicate_rows)
    split={"train":[r["basename"] for r in rows if r["split"]=="train"],
           "test":[r["basename"] for r in rows if r["split"]=="test"],
           "strategy":"temporal blocks: train<=749, test>=803","intersection_hashes":[],
           "manifest_hash":sha256(dest/"manifest.csv"),"dataset_version":cfg["dataset_version"]}
    write_json(dest/"split.json",split)
    summary={"source_image_files":sum(map(len,paths.values())),"unique_image_names":len(paths),
             "duplicate_copies_removed":len(duplicate_rows),"excluded_images":excluded,"candidate_images":len(rows),
             "train_images":743,"test_images":400,"polygon_target":"lane area only",
             "mask_values":[0,255],"rasterization":"PIL polygon fill; clip x/y to pixel bounds; YOLO roundtrip rounded to source 0.01 px",
             "roundtrip_mismatches":0,"manifest_hash":split["manifest_hash"],"elapsed_seconds":time.perf_counter()-start}
    write_json(output(cfg)/"preparation-summary.json",summary)
    for name,rgb,gt in selected_previews:
        green=Image.new("RGB",rgb.size,(0,255,0))
        overlay=Image.composite(Image.blend(rgb,green,.35),rgb,gt)
        from PIL import ImageDraw
        canvas=Image.new("RGB",(1440,305),"white")
        for j,pic in enumerate([rgb,overlay,gt.convert("RGB")]):
            canvas.paste(pic.resize((480,270)),(j*480,25))
        ImageDraw.Draw(canvas).text((10,5),name+" | RGB / lane polygon overlay / binary GT",fill="black")
        canvas.save(output(cfg)/"figures"/f"conversion-{Path(name).stem}.png")
    record(cfg,"prepare_data",**summary)
    print(summary)

if __name__ == "__main__":
    main()
