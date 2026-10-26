"""共享模型命令入口：准备小样本、训练、推理；只写新的输出目录。

示例：python -m oa_groundrag.runtime.shared_cli --help
输入为已核验的本地模型/数据；输出为独立研究目录，不修改原始资产。
"""
from pathlib import Path
import argparse
import json


def parser():
    p=argparse.ArgumentParser(description='Geo-GroundRAG experimental shared model')
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare',help='Export at most 8 train/val samples into a new directory')
    prepare.add_argument('--benchmark',type=Path,required=True)
    prepare.add_argument('--compact',type=Path)
    prepare.add_argument('--split',choices=['train','val'],default='train')
    prepare.add_argument('--indices',type=int,nargs='+',default=[0])
    prepare.add_argument('--modalities',nargs='*',choices=['dem','insar_velocity','slope'],default=[])
    prepare.add_argument('--output',type=Path,required=True)
    score=sub.add_parser('score',help='Score one identified val prediction; not a formal evaluation')
    score.add_argument('--data',type=Path,required=True)
    score.add_argument('--prediction',type=Path,required=True)
    score.add_argument('--index',type=int,default=0)
    score.add_argument('--output',type=Path,required=True)
    for name in ('infer','train'):
        cmd=sub.add_parser(name)
        cmd.add_argument('--model',type=Path,required=True)
        cmd.add_argument('--identity',type=Path,required=True)
        cmd.add_argument('--device',choices=['cpu','cuda'],default='cpu')
        cmd.add_argument('--seed',type=int,default=20261026)
        cmd.add_argument('--max-pixels',type=int,default=65536)
        cmd.add_argument('--output',type=Path,required=True)
        cmd.add_argument('--data',type=Path)
        cmd.add_argument('--delta',type=Path)
        if name=='infer':
            cmd.add_argument('--image',type=Path)
            cmd.add_argument('--index',type=int,default=0)
            cmd.add_argument('--question',default='Locate and describe candidate landslide regions.')
            cmd.add_argument('--mode',choices=['joint','feedback'],default='feedback')
            cmd.add_argument('--rag-config',type=Path)
            cmd.add_argument('--knowledge-only',action='store_true')
        else:
            cmd.add_argument('--steps',type=int,default=2)
            cmd.add_argument('--formal-run',action='store_true',help='Explicit long-training mode; requires separate owner authorization')
            cmd.add_argument('--time-limit',type=int,default=600)
            cmd.add_argument('--learning-rate',type=float,default=1e-4)
            cmd.add_argument('--visual-last-block',action='store_true')
            cmd.add_argument('--description-supervision',choices=['joint','feedback'],default='joint')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    from oa_groundrag.runtime.contracts import reject_test_or_sealed_path
    if args.output.exists():raise FileExistsError(args.output)
    reject_test_or_sealed_path(args.output,label='shared output')
    if args.command=='score':
        from oa_groundrag.data.shared_segmentation.dataset import load_samples
        from oa_groundrag.evaluation.shared_segmentation import score_artifact
        from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
        result=score_artifact(args.prediction,load_samples(args.data)[args.index])
        with AtomicArtifactDirectory(args.output) as writer:
            writer.write_json('metrics.json',result);writer.publish()
        print(json.dumps(result,indent=2));return
    if args.command=='prepare':
        from oa_groundrag.data.shared_segmentation.dataset import SegmentationDataset,CompactRegionDataset,export_samples
        if not 1<=len(args.indices)<=8 or any(i<0 for i in args.indices):raise ValueError('Choose 1–8 nonnegative indices')
        dataset=CompactRegionDataset(args.compact,args.benchmark) if args.compact else SegmentationDataset(args.benchmark,args.split,tuple(args.modalities))
        samples=[dataset[i] for i in args.indices]
        if any(s.split!=args.split for s in samples):raise ValueError('Selected record does not belong to requested split')
        print(json.dumps(export_samples(samples,args.output),indent=2));return
    if args.command=='train' and not args.formal_run and (not 1<=args.steps<=2 or not 0<args.time_limit<=600):
        raise ValueError('Default training allows at most 2 updates and 600 seconds')
    import torch
    torch.manual_seed(args.seed)
    from oa_groundrag.vlm.shared_segmentation.session import SharedModelSession
    from oa_groundrag.training.shared_segmentation.engine import load_delta
    reject_test_or_sealed_path(args.model,label='shared model')
    session=SharedModelSession.load(args.model,args.identity,device=args.device,max_pixels=args.max_pixels)
    identity=session.identity
    if args.delta:
        load_delta(session.model,args.delta,identity)
        from oa_groundrag.artifacts.identity import sha256_file
        session.identity={'kind':'geo_groundrag_delta','base_identity':identity,'delta_manifest_sha256':sha256_file(args.delta/'manifest.json')}
    if args.command=='train':
        if args.delta:raise ValueError('Bounded training starts from the declared base; resume is not implicit')
        if args.data is None:raise ValueError('--data is required for training')
        from oa_groundrag.data.shared_segmentation.dataset import load_samples
        from oa_groundrag.training.shared_segmentation.engine import train_bounded,train_joint
        train=train_joint if args.formal_run else train_bounded
        print(json.dumps(train(session,load_samples(args.data),args.output,base_identity=identity,
                         timeout_seconds=args.time_limit,learning_rate=args.learning_rate,
                         steps=args.steps,visual_last_block=args.visual_last_block,description_supervision=args.description_supervision),indent=2));return
    from oa_groundrag.runtime.shared_model import GeoGroundRuntime,ExistingTextRAG,save_result
    retriever=ExistingTextRAG(args.rag_config,device='cpu') if args.rag_config else None
    runtime=GeoGroundRuntime(session,retriever)
    if args.knowledge_only:
        if not retriever:raise ValueError('Knowledge QA requires --rag-config')
        from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
        with AtomicArtifactDirectory(args.output) as writer:
            result=runtime.knowledge_qa(args.question)
            writer.write_json('result.json',result);writer.publish()
    else:
        from PIL import Image
        if args.data and args.image:raise ValueError('Choose --data or --image')
        if args.data:
            from oa_groundrag.data.shared_segmentation.dataset import load_samples
            sample=load_samples(args.data)[args.index];image=sample.image;aux=sample.auxiliaries
        elif args.image:
            reject_test_or_sealed_path(args.image,label='input image')
            with Image.open(args.image) as im:image=im.convert('RGB').copy()
            aux={}
        else:raise ValueError('Specify --data or --image')
        result=save_result(runtime.run(image,args.question,auxiliaries=aux,mode=args.mode,record_id=sample.record_id if args.data else 'user-image'),args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
