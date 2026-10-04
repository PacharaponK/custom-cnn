"""Fresh-process batch-1 FP32 GPU tensor memory and sampled process RSS."""
import argparse
import io
import statistics
import threading
import time
import psutil
import torch
from PIL import Image
from common import ROOT,configuration,load_model,output,read_csv,record,sha256,write_csv,write_json
from predict import predict_array

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="config.json")
    cfg=configuration(parser.parse_args().config);out=output(cfg);record(cfg,"measure_memory","started")
    process=psutil.Process();cold_rss=process.memory_info().rss
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats()
    model,checkpoint=load_model(cfg,device)
    x=torch.zeros(1,3,cfg["height"],cfg["width"],device=device,dtype=torch.float32)
    first=next(r for r in read_csv(ROOT/"data/manifest.csv") if r["split"]=="test")
    sample_path=ROOT/first["image_path"]
    def sync():
        if device.type=="cuda":torch.cuda.synchronize()
    def gpu_baseline():
        if device.type=="cuda":
            return {"allocated_bytes":torch.cuda.memory_allocated(),"reserved_bytes":torch.cuda.memory_reserved()}
        return None
    def gpu_peak(baseline):
        if device.type!="cuda":return None
        return {"baseline_allocated_mib":baseline["allocated_bytes"]/1024**2,
                "baseline_reserved_mib":baseline["reserved_bytes"]/1024**2,
                "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
                "peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2,
                "incremental_peak_allocated_mib":(torch.cuda.max_memory_allocated()-baseline["allocated_bytes"])/1024**2}
    with torch.inference_mode():
        for _ in range(10):model(x)
    sync()
    cold_gpu={"peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
              "peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2} if device.type=="cuda" else None
    def benchmark(name,operation):
        baseline=gpu_baseline();baseline_rss=process.memory_info().rss;rss=[baseline_rss];done=threading.Event()
        def sample():
            while not done.wait(.005):rss.append(process.memory_info().rss)
        watcher=threading.Thread(target=sample,daemon=True);watcher.start()
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
        durations=[]
        try:
            with torch.inference_mode():
                for i in range(100):
                    sync();start=time.perf_counter();operation();sync();durations.append((time.perf_counter()-start)*1000)
        finally:
            done.set();watcher.join();rss.append(process.memory_info().rss)
        result={"mode":name,"repetitions":100,"rss_sampling_interval_ms":5,
                "baseline_cpu_rss_mib":baseline_rss/1024**2,"sampled_peak_cpu_rss_mib":max(rss)/1024**2,
                "gpu":gpu_peak(baseline),"median_latency_ms":statistics.median(durations),
                "mean_latency_ms":statistics.mean(durations),"min_latency_ms":min(durations),"max_latency_ms":max(durations)}
        write_csv(out/f"memory-{name}-latencies.csv",[{"iteration":i+1,"latency_ms":v} for i,v in enumerate(durations)])
        return result
    forward=benchmark("forward",lambda:model(x))
    def end_to_end():
        with Image.open(sample_path) as im:mask=predict_array(model,im.convert("RGB"),cfg,device)
        buffer=io.BytesIO();Image.fromarray(mask).save(buffer,format="PNG");buffer.close()
    e2e=benchmark("end-to-end",end_to_end)
    summary={"device":str(device),"dtype":"float32","batch_size":1,"input_wh":[cfg["width"],cfg["height"]],
             "original_image_wh":[1280,720],"warmup_iterations":10,"checkpoint_hash":sha256(checkpoint),
             "parameters":sum(p.numel() for p in model.parameters()),
             "parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
             "buffer_bytes":sum(b.numel()*b.element_size() for b in model.buffers()),
             "checkpoint_bytes":checkpoint.stat().st_size,"cold_pre_model_cpu_rss_mib":cold_rss/1024**2,
             "cold_load_and_warmup_gpu":cold_gpu,"forward":forward,"end_to_end":e2e,
             "gpu_scope":"PyTorch tensors/reserved allocator; excludes CUDA context/driver/non-PyTorch allocations",
             "rss_scope":"whole process after imports; sampled peak may miss transient spikes",
             "end_to_end_scope":"read JPEG, RGB resize, forward, sigmoid/upscale/threshold, CPU copy, in-memory PNG encoding; no disk write"}
    write_json(out/"inference-memory.json",summary);record(cfg,"measure_memory",**summary);print(summary)

if __name__=="__main__":main()
