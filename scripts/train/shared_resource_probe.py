"""共享模型资源小检查：真实本地 Qwen3.5 + 随机解码器，不是 Lens 成绩。

命令：python scripts/train/shared_resource_probe.py --help
输入：原有只读模型清单及最多 8 条新训练样本；输出：新的检查目录。
最多 2 次参数更新，10 分钟截止；不会下载模型或改写原权重。
"""
from pathlib import Path
import argparse
import json
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'kind':'untrained_mask_decoder_resource_probe','scientific_acceptance':False,'max_pixels':1024,'max_steps':2}
    start=time.monotonic()
    def deadline(*_):raise TimeoutError('600 second probe limit')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    try:
        import torch
        from PIL import Image
        from oa_groundrag.vlm.shared_segmentation.diagnostics import load_base_for_diagnostics
        from oa_groundrag.data.shared_segmentation.dataset import load_samples
        from oa_groundrag.training.shared_segmentation.engine import train_bounded,collate_one
        from oa_groundrag.vlm.shared_segmentation.session import restore_mask
        torch.manual_seed(20261026)
        free,total=torch.cuda.mem_get_info()
        report['initial_free_bytes']=free;report['device_total_bytes']=total
        if free<13*1024**3:raise RuntimeError('Insufficient free GPU memory for the bounded probe')
        session,identity=load_base_for_diagnostics(args.base,args.ledger)
        report['identity']=identity
        samples=load_samples(args.data)
        report['training']=train_bounded(session,samples,args.output/'delta',base_identity=identity,
            timeout_seconds=max(1,600-(time.monotonic()-start)))
        sample=samples[-1];batch=collate_one(session,sample)
        torch.cuda.reset_peak_memory_stats();tic=time.monotonic()
        with torch.inference_mode(),session.model.auxiliary_context([sample.auxiliaries],[(sample.image.height,sample.image.width)]):
            out=session.model(**batch)
        grid=batch['image_grid_thw'][0]
        mask=restore_mask(out.pred_masks[0,0],(int(grid[1])*16,int(grid[2])*16),(sample.image.height,sample.image.width))
        Image.fromarray(mask.astype('uint8')*255).save(args.output/'teacher_forced_untrained_mask.png')
        report['teacher_forced']={'not_free_prediction':True,'mask_shape':list(mask.shape),'foreground_pixels':int(mask.sum()),
                                  'elapsed_seconds':time.monotonic()-tic,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        _,tokens,text,complete=session._generate([sample.image],'Locate candidate landslide regions and report masks.',[sample.auxiliaries])
        report['generation']={'text':text,'tokens':int(tokens.numel()),'complete':complete,'mask_tokens':int((tokens==session.mask_token_id).sum())}
        report['status']='computational_check_passed'
    except Exception as error:
        report['status']='failed';report['error']=str(error)
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
