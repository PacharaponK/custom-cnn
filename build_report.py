"""Build a UTF-8 standalone HTML report from measured outputs; images are embedded."""
import argparse
import base64
import html
import shutil
from pathlib import Path
from common import ROOT,configuration,now,output,read_csv,read_json,record,sha256,write_json

def esc(value):
    return html.escape(str(value))

def table(headers,rows):
    head="<thead><tr>"+"".join("<th scope='col'>"+esc(h)+"</th>" for h in headers)+"</tr></thead>"
    body="<tbody>"+"".join("<tr>"+"".join("<td>"+esc(v)+"</td>" for v in row)+"</tr>" for row in rows)+"</tbody>"
    return "<div class='table-wrap'><table>"+head+body+"</table></div>"

def figure(path,caption):
    data=base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return "<figure><img loading='eager' src='data:image/png;base64,"+data+"' alt='"+esc(caption)+"'><figcaption>"+esc(caption)+"</figcaption></figure>"

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="config.json")
    cfg=configuration(parser.parse_args().config);out=output(cfg)
    required=["preparation-summary.json","training-summary.json","environment.json","evaluation-summary.json","inference-memory.json","verification.json"]
    docs={name:read_json(out/name) for name in required}
    prep=docs["preparation-summary.json"];train=docs["training-summary.json"];env=docs["environment.json"]
    evaluation=docs["evaluation-summary.json"];memory=docs["inference-memory.json"]
    assert docs["verification.json"]["passed"] and train["epochs"]>=30
    if train["checkpoint_hash"]!=memory["checkpoint_hash"]:raise ValueError("Memory/checkpoint mismatch")
    source=Path(cfg["vault_assignment"])/"results"
    for name in ["01-ground-truth-preview.png","02-input-resolution-comparison.png","03-annotation-label-issues.png","04-conflicting-ground-truth.png"]:
        dest=out/"figures"/("audit-"+name)
        if not dest.exists():shutil.copy2(source/"figures"/name,dest)
    audit_dest=out/"source-audit.json"
    if not audit_dest.exists():shutil.copy2(source/"audit/dataset-audit.json",audit_dest)
    snapshots=read_json(out/"snapshots.json");metrics=read_csv(out/"metrics.csv")
    section_names=[("overview","ภาพรวม"),("dataset","Dataset"),("prepare","เตรียมข้อมูล"),("architecture","Architecture"),
                   ("training","Training"),("evaluation","Evaluation"),("snapshots","Inference snapshots"),
                   ("memory","Memory"),("discussion","ข้อจำกัด"),("reproduce","ทำซ้ำและหลักฐาน")]
    nav="".join("<a href='#"+key+"'>"+esc(title)+"</a>" for key,title in section_names)
    parts=[]
    def section(key,title,content):
        parts.append("<section id='"+key+"'><h2>"+esc(title)+"</h2>"+content+"</section>")
    overview="<p>ออกแบบ compact U-Net สำหรับแบ่งพื้นที่ lane class เดียว โดย train ทุก layer ใหม่บน PSU reservoir dataset ใช้เฉพาะ polygon label lane และประเมินบน temporal test block ที่แยกไว้ก่อน train</p>"
    overview+="<div class='kpis'>"
    for label,value in [("Mean test IoU",f"{evaluation['mean_iou_all']:.4f}"),("Detected IoU ≥0.6",f"{evaluation['detected_ge_0_6']}/{evaluation['test_images']}"),
                        ("Trainable parameters",f"{train['parameters']:,}"),("E2E GPU peak allocated",f"{memory['end_to_end']['gpu']['peak_allocated_mib']:.3f} MiB")]:
        overview+="<div class='kpi'><strong>"+esc(value)+"</strong><span>"+esc(label)+"</span></div>"
    overview+="</div><p class='note'>Detection flag เป็นการวัดความสำเร็จด้วย ground truth หลัง inference ไม่ใช่ confidence ของโมเดล และ 400/400 ไม่ได้หมายถึง pixel accuracy 100%</p>"
    section("overview","1. Objective และผลรวม",overview)
    dataset=table(["รายการ","จำนวน/รายละเอียด"],[
        ["ภาพ source","1,200 ไฟล์, 1280×720; unique SHA-256 1,148"],
        ["สำเนาที่ deduplicate",prep["duplicate_copies_removed"]],
        ["ภาพที่แยกออก","400, 401: GT conflict; 451, 452, 544: uncertain lane labels"],
        ["ภาพใช้จริง",prep["candidate_images"]],
        ["Train",f"{prep['train_images']} ภาพ; frame 2–749 หลัง clean; 65.004%"],
        ["Test",f"{prep['test_images']} ภาพ; frame 803–1202; 34.996%"],
        ["Train/test content-hash overlap","0"],
        ["ช่องว่าง frame","750–802 ไม่มีใน source; ไม่อยู่ใน denominator"],
        ["Target","พื้นที่เลนสองฝั่งของเส้นกลาง รวม foreground class เดียว; sideway/track-line/boxes/polylines ไม่ใช้"]])
    dataset+="<p>แยกภาพที่มีปัญหาตามคำยืนยันของผู้ใช้ โดยเก็บ JPG/XML ต้นฉบับครบ การตรวจ geometry ตรวจ bounds/finite/area แต่ไม่ได้พิสูจน์ semantic correctness หรือ self-intersection ทุก polygon</p>"
    dataset+=figure(out/"figures/audit-01-ground-truth-preview.png","รูป 1: Source RGB, polygon lane overlay และ binary ground truth ของ frame 2/475/700/803; ภาพนี้เป็น GT preparation")
    dataset+=figure(out/"figures/audit-03-annotation-label-issues.png","รูป 2: frame 451/452/544 มี label ที่ทำให้ lane-only mask ว่าง จึงแยกออกก่อน train")
    dataset+=figure(out/"figures/audit-04-conflicting-ground-truth.png","รูป 3: ภาพซ้ำ frame 400/401 มี GT ต่างกันจากสอง XML; IoU annotation–annotation 0.983949/0.978455")
    section("dataset","2. Dataset และการคัดกรอง",dataset)
    preparation="<ol><li>จับคู่ JPG/XML ด้วย basename และตรวจ SHA-256 ของสำเนา ไม่ใช้ XML id เป็น global ID</li><li>เลือกเฉพาะ polygon label lane; เขียน YOLO class 0 และ normalized coordinates 8 ตำแหน่ง</li><li>สร้าง original-resolution GT 1280×720 ด้วย union polygons; clip พิกัดที่ขอบภาพและใช้ convention เดียวกันใน conversion/evaluation</li><li>YOLO roundtrip reconstruct พิกัดถึง 0.01 pixel ตาม source; mismatch count = 0 จากภาพใช้จริงทั้ง 1,143 ภาพ</li><li>Resize RGB bilinear เป็น W×H=64×36 และ GT nearest-neighbor; cached arrays เก็บไว้เพื่อ train</li><li>ล็อก split ก่อน train; final run ไม่มี test-driven tuning และไม่มี validation split แยก</li></ol>"
    preparation+=figure(out/"figures/audit-02-input-resolution-comparison.png","รูป 4: 48×48 ใน TXT เทียบ 64×36 ใน PDF; ผู้ใช้ยืนยัน 64×36 ซึ่งรักษา aspect ratio 16:9")
    section("prepare","3. Data preparation",preparation)
    architecture="<p>โมเดลประกอบด้วยสอง encoder/pooling stages, bottleneck และสอง decoder stages พร้อม skip connections ทุก Conv ใช้ bias; Conv 3×3 padding 1 และ ReLU ยกเว้น head 1×1 เป็น logits ไม่มี pretrained backbone, BatchNorm หรือ Dropout</p>"
    architecture+="<svg role='img' aria-label='Compact U-Net encoder decoder with skip connections' viewBox='0 0 1050 310' xmlns='http://www.w3.org/2000/svg'><defs><marker id='arrow' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' fill='#125e69'/></marker></defs>"
    nodes=[(15,"RGB","3 × 36 × 64"),(160,"Encoder 1","8 × 36 × 64"),(305,"Encoder 2","16 × 18 × 32"),(450,"Bottleneck","32 × 9 × 16"),(595,"Decoder 2","16 × 18 × 32"),(740,"Decoder 1","8 × 36 × 64"),(885,"Head logits","1 × 36 × 64")]
    for x,label,shape in nodes:
        architecture+=f"<rect x='{x}' y='160' width='130' height='70' rx='8' fill='#eaf3f4' stroke='#125e69'/><text x='{x+65}' y='187' text-anchor='middle' font-size='17' fill='#14313a'>{esc(label)}</text><text x='{x+65}' y='211' text-anchor='middle' font-size='14' fill='#14313a'>{esc(shape)}</text>"
    for x,_,_ in nodes[:-1]:architecture+=f"<path d='M{x+130},195 H{x+143}' fill='none' stroke='#125e69' stroke-width='2' marker-end='url(#arrow)'/>"
    architecture+="<path d='M370,160 V110 H660 V160' fill='none' stroke='#125e69' stroke-width='2' marker-end='url(#arrow)'/><text x='510' y='100' text-anchor='middle' font-size='16'>skip E2 → concat 48 channels</text><path d='M225,160 V55 H805 V160' fill='none' stroke='#125e69' stroke-width='2' marker-end='url(#arrow)'/><text x='510' y='43' text-anchor='middle' font-size='16'>skip E1 → concat 24 channels</text><text x='510' y='275' text-anchor='middle' font-size='16'>MaxPool on encoder / bilinear upsample on decoder</text></svg>"
    architecture+=table(["Stage","Output N,C,H,W"],[["Input","N,3,36,64"],["E1 / Pool1","N,8,36,64 / N,8,18,32"],["E2 / Pool2","N,16,18,32 / N,16,9,16"],["Bottleneck","N,32,9,16"],["D2 (concat 48→16→16)","N,16,18,32"],["D1 (concat 24→8→8)","N,8,36,64"],["Head logits","N,1,36,64"]])
    architecture+=f"<p>Parameters ที่นับจาก model จริง = {train['parameters']:,}; FP32 parameter bytes = {train['parameter_bytes']:,} ({train['parameter_bytes']/1024**2:.5f} MiB) สอง pooling stages ให้ขนาดลงตัวถึง 9×16; skip connections ส่งรายละเอียดต้นทางกลับ decoder; bilinear upsampling ไม่มี weights เพิ่ม ขนาด channels 8/16/32 ช่วยลดการใช้ memory โดยประสิทธิภาพอ่านจากผล test ที่วัดจริง</p>"
    section("architecture","4. Custom network architecture",architecture)
    training=table(["รายการ","ค่าจริง"],[["Device",env["gpu"]],["Python / uv",env["python"]+" / "+env["uv"]],["PyTorch / CUDA",env["packages"]["torch"]+" / "+str(env["cuda_runtime"])],
        ["Batch / epochs",str(cfg["batch_size"])+" / "+str(train["epochs"])],["Optimizer / LR","Adam / 0.001 fixed"],["Loss","BCEWithLogitsLoss; logits input, GT float 0/1"],
        ["Epoch samples / steps",f"{train['samples_per_epoch']} / {train['global_steps']} total steps"],["Training elapsed",f"{train['elapsed_seconds']:.3f} s"],
        ["From scratch","ทุก layer random initialization; ไม่มี pretrained weights; smoke weights ไม่ใช้ต่อ"],
        ["Checkpoint policy","Final epoch 30; ไม่เลือกจาก test score"]])
    training+="<p>Train-only augmentation: white-balance multiplier 0.95–1.05 (p=0.3), brightness 0.9–1.1 (p=0.3), Gaussian blur radius 0.2–0.8 (p=0.2) เป็น photometric transforms จึงไม่เปลี่ยน mask</p>"
    training+=figure(out/"figures/augmentation-example.png","รูป 5: ตัวอย่าง photometric augmentation จาก train sample; RGB เท่านั้นที่ถูกเปลี่ยน")
    training+=figure(out/"figures/train_loss.png","รูป 6: Train BCE loss จาก CSV/TensorBoard ของ final run 30 epochs; average แบบ sample-weighted")
    training+=figure(out/"figures/train_pixel_iou.png","รูป 7: Train pixel IoU แบบ aggregate บน resized/augmented batches; ไม่ใช่ macro test IoU ที่ original resolution")
    training+=figure(out/"figures/learning_rate.png","รูป 8: Learning rate 0.001 คงที่ตลอด final run")
    training+=f"<p>Loss ลดจาก epoch 1 = {train['initial_epoch_loss']:.6f} เป็น epoch 30 = {train['final_epoch_loss']:.6f}; ช่วงท้ายมีการแกว่งเล็กน้อยราว 0.027–0.030 หลักฐานนี้แสดงการ optimize training objective ไม่ใช้เพียงจำนวน epochs เป็นข้อสรุปว่า generalize ได้กับทุกถนน</p>"
    section("training","5. Training และ convergence",training)
    evaltext="<p>โหลด predicted PNG และ GT PNG ที่ save จริงขนาด 1280×720 แล้วคำนวณ pixel-wise lane IoU = intersection / union ของ foreground ใช้ comparator ≥0.6 เป็นหลักตาม PDF และเก็บ >0.6 เพิ่มตาม TXT</p>"
    evaltext+=table(["Metric","ค่าจริง"],[["Test images",evaluation["test_images"]],["Detected IoU≥0.6",evaluation["detected_ge_0_6"]],["Detected IoU>0.6",evaluation["detected_gt_0_6"]],["Exact IoU=0.6",evaluation["exact_threshold_images"]],
        ["Detected rate",f"{evaluation['detected_rate']*100:.2f}%"],["Mean IoU detected",f"{evaluation['mean_iou_detected']:.8f}"],["Mean IoU all",f"{evaluation['mean_iou_all']:.8f}"],
        ["Median / min / max IoU",f"{evaluation['median_iou']:.6f} / {evaluation['min_iou']:.6f} / {evaluation['max_iou']:.6f}"],["Empty-empty images",evaluation["empty_empty_images"]]])
    evaltext+="<p>Mean IoU detected เท่ากับ mean IoU all ใน run นี้เพราะทุกภาพผ่านเกณฑ์ หากไม่มี detected images จะรายงาน N/A; empty-empty policy คือ IoU=1 พร้อมนับกรณี ซึ่ง run นี้ไม่มีกรณีดังกล่าว</p>"
    evaltext+=figure(out/"figures/test-iou-histogram.png","รูป 9: Distribution ของ per-image IoU ใน test 400 ภาพและเส้นเกณฑ์ 0.6")
    section("evaluation","6. Quantitative evaluation",evaltext)
    snapshottext="<p>Inference ใช้ RGB→64×36→logits→sigmoid→bilinear probability upscale→threshold 0.5→PNG 0/255 เลือก 6 ตัวอย่างตามผลทั้งชุด: สูงสุด 2, ใกล้เกณฑ์ที่สุด 2 และผลต่ำถัดมา 2 เพื่อไม่ซ้ำกลุ่มใกล้เกณฑ์ ทุกตัวอย่างยังผ่านเกณฑ์ 0.6; กลุ่มผลต่ำจึงเป็นผลที่แย่กว่าในชุดนี้ ไม่ใช่ detection failure</p>"
    kinds={"high":"IoU สูง","near-threshold":"ใกล้เกณฑ์ที่สุด","low":"IoU ต่ำในชุดนี้"}
    for i,snapshot in enumerate(snapshots,10):
        snapshottext+=figure(ROOT/snapshot["figure"],f"รูป {i}: {snapshot['basename']} — {kinds[snapshot['kind']]}; IoU={snapshot['iou']:.6f}, detection=Yes; RGB/GT/prediction/overlay")
    section("snapshots","7. Inference snapshots ก่อนและหลัง",snapshottext)
    rows=[]
    for title,key in [("Model forward","forward"),("End-to-end","end_to_end")]:
        result=memory[key];gpu=result["gpu"]
        rows.append([title,f"{gpu['peak_allocated_mib']:.6f}",f"{gpu['peak_reserved_mib']:.3f}",f"{gpu['incremental_peak_allocated_mib']:.6f}",f"{result['sampled_peak_cpu_rss_mib']:.3f}",f"{result['median_latency_ms']:.3f}"])
    mem="<p>วัดใน process แยกจาก training, CUDA batch=1, FP32, input 64×36, warm-up 10 ครั้ง และวัด 100 ครั้งต่อ mode โดย synchronize GPU รอบการจับเวลา</p>"
    mem+=table(["Mode","GPU peak allocated MiB","GPU peak reserved MiB","GPU incremental allocated MiB","CPU sampled peak RSS MiB","Median latency ms"],rows)
    mem+=f"<p>Parameter storage {memory['parameter_bytes']:,} bytes; buffers {memory['buffer_bytes']} bytes; checkpoint {memory['checkpoint_bytes']:,} bytes เป็นค่าคนละประเภทกับ inference footprint GPU peak allocated นับเฉพาะ PyTorch tensors และ reserved นับ allocator cache ไม่รวม CUDA context/driver/non-PyTorch allocations</p>"
    mem+="<p>CPU RSS เป็นทั้ง process รวม framework และ sampled ทุก 5 ms จึงอาจพลาด transient spike End-to-end รวมอ่าน JPEG, RGB resize, model, mask resize/threshold, copy กลับ CPU และ encode PNG ใน memory ไม่รวมเขียน disk ค่า timing เป็นการวัดบนภาพ test แรกซ้ำ 100 ครั้ง ไม่ใช่ latency distribution ของ test ทั้ง 400 ภาพ</p>"
    section("memory","8. Inference memory footprint และเวลา",mem)
    discussion="<ul><li>ข้อมูลมาจากเส้นทางเดียวและช่วง frame ของ sequence เดียว ผล temporal test ยังไม่ยืนยันการใช้งานกับถนน/กล้อง/แสงหรือวันที่เก็บข้อมูลอื่น</li><li>Input 64×36 ลดรายละเอียด boundary; ภาพขนาดเล็กเหมาะกับ baseline memory ต่ำ แต่ความละเอียดของ mask มีข้อจำกัดหลังขยายกลับ 1280×720</li><li>Train IoU คำนวณ micro บน batch ที่ resize/augment ส่วน test เป็น macro per-image ที่ original resolution จึงไม่ตีความช่องว่างของสองค่าตรง ๆ เป็น overfitting โดยไม่มีหลักฐานเพิ่ม</li><li>แยกภาพ label ผิด/GT conflict 5 ภาพตามที่ยืนยัน แต่ยังไม่ได้พิสูจน์ว่า annotation ที่เหลือถูก semantic ทุกจุด</li><li>ทุกภาพผ่าน threshold 0.6 แต่ mean IoU=0.8121 และค่าต่ำสุดประมาณ 0.6856 แสดงว่ายังมี segmentation error; ดู snapshot/CSV ประกอบ ไม่สรุปว่า segmentation สมบูรณ์</li></ul>"
    section("discussion","9. Discussion และข้อจำกัด",discussion)
    prefix="result/"+cfg["run_id"]+"/"
    evidence_files=["preparation-summary.json","metrics.csv","training-summary.json","evaluation-per-image.csv","evaluation-summary.json","inference-memory.json","verification.json","environment.json","prediction-manifest.csv","pipeline-history.json"]
    links="<ul>"+"".join("<li><a href='"+esc(prefix+name)+"'>"+esc(name)+"</a></li>" for name in evidence_files)+"</ul>"
    reproduce="<p>ใช้ uv project และ lockfile ที่ส่ง: uv sync --locked จากนั้น uv run --locked python verify_results.py ตรวจ checkpoint, PNG 400 ภาพ, recomputed metrics และ fresh inference สองภาพจริง โดยไม่ทับ outputs</p>"
    reproduce+="<pre>uv sync --locked\nuv run --locked python verify_results.py\nuv run --locked tensorboard --logdir runs --host 127.0.0.1</pre>"
    reproduce+=table(["Provenance","Value"],[["Run ID",cfg["run_id"]],["Dataset version",cfg["dataset_version"]],["Seed",cfg["seed"]],["Manifest SHA-256",prep["manifest_hash"]],["Initial state SHA-256",train["initial_hash"]],
        ["Final checkpoint SHA-256",train["checkpoint_hash"]],["uv.lock SHA-256",env["lockfile_hash"]],["Generated",now()],["Publication","ชุดส่งอยู่ในเครื่อง; ยังไม่ได้ upload หรือส่งใน Teams"]])
    reproduce+=links+"<p>HTML ฝังภาพทั้งหมดและ CSS ไว้ในไฟล์เดียว จึงอ่านเนื้อหาหลัก offline ได้ ส่วน raw evidence ใช้ relative links ในชุดไฟล์ที่ส่งและ TensorBoard ต้องรันจาก code project</p>"
    section("reproduce","10. Reproducibility และภาคผนวก",reproduce)
    style="""*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f8;color:#182c38;font:16px/1.75 'Segoe UI',Tahoma,sans-serif}main{max-width:1180px;margin:auto;background:#fff;padding:48px 56px}header{border-bottom:3px solid #125e69;padding-bottom:28px}.eyebrow{color:#125e69;font-weight:650;letter-spacing:.05em}h1{font-size:38px;line-height:1.25;margin:12px 0}h2{font-size:25px;line-height:1.4;color:#125e69;margin:0 0 20px}section{padding:38px 0;border-bottom:1px solid #d9e3e9;scroll-margin-top:20px}nav{display:flex;flex-wrap:wrap;gap:8px 20px;margin:28px 0}a{color:#125e69;text-decoration:underline;text-underline-offset:3px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.kpi{padding:18px;background:#eaf3f4;border-radius:8px}.kpi strong{display:block;font-size:27px;line-height:1.3}.kpi span{display:block;font-size:14px;margin-top:6px}.note{padding:14px 18px;border-left:4px solid #b77c22;background:#fff8ed}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:15px;margin:16px 0}th,td{text-align:left;vertical-align:top;padding:11px 13px;border-bottom:1px solid #d7e1e7;overflow-wrap:anywhere}th{background:#eaf3f4;color:#123e47}td:first-child{font-weight:550}figure{margin:26px 0}figure img{display:block;width:100%;height:auto;border:1px solid #e1e7eb}figcaption{font-size:14px;color:#405968;margin:9px 0}svg{width:100%;height:auto}pre{white-space:pre-wrap;background:#f2f5f8;padding:18px;border-radius:6px;font-size:14px;overflow-wrap:anywhere}footer{font-size:13px;color:#405968;padding:26px 0}li{margin:8px 0}@media(max-width:650px){main{padding:24px 18px}h1{font-size:29px}h2{font-size:22px}.kpis{grid-template-columns:repeat(2,1fr)}.kpi strong{font-size:22px}table{font-size:13px}th,td{padding:9px 8px}}@page{size:A4;margin:15mm}@media print{body{background:white;font-size:10.5pt;line-height:1.55}main{max-width:none;padding:0}h1{font-size:25pt}h2{font-size:16pt}nav{display:none}section{padding:18px 0}figure,table,svg,.kpis{break-inside:avoid}h2{break-after:avoid}figure img{max-height:225mm;object-fit:contain}table{font-size:9pt}.table-wrap{overflow:visible}pre{font-size:9pt}a{color:inherit}.kpi{background:white;border:1px solid #cbd7dc}.kpi strong{font-size:18pt}}"""
    document="<!doctype html><html lang='th'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Assignment 10 — Custom Lane Segmentation — 6610110190</title><style>"+style+"</style></head><body><main><header><div class='eyebrow'>241-353 AI ECOSYSTEM · ASSIGNMENT 10</div><h1>Custom Lane Segmentation<br>Training from Scratch</h1><p>รหัสนักศึกษา 6610110190 · P.Fern · รายงานผลทดลองวันที่ 2026-10-04</p><p>"+esc(cfg["run_id"])+"</p></header><nav aria-label='สารบัญ'>"+nav+"</nav>"+"".join(parts)+"<footer>รายงานประกอบจากผลรันและหลักฐานจริง; code/runtime/checkpoint อยู่ใน code workspace และชุดส่ง ใช้ HTML นี้ตรวจทานก่อน upload</footer></main></body></html>"
    report_dir=out/"report";report_dir.mkdir(exist_ok=True)
    path=report_dir/"Assignment10-6610110190.html";path.write_text(document,encoding="utf-8")
    shutil.copy2(path,ROOT/path.name)
    write_json(out/"report-build.json",{"created_at":now(),"report_hash":sha256(path),"report_bytes":path.stat().st_size,
             "run_id":cfg["run_id"],"checkpoint_hash":train["checkpoint_hash"],
             "input_hashes":{name:sha256(out/name) for name in required},"embedded_images":document.count("data:image/png;base64,"),
             "external_runtime_resources":0})
    record(cfg,"build_report",report_hash=sha256(path));print("Built report:",path,path.stat().st_size,"bytes")

if __name__=="__main__":main()
