"""Read-only adapters for Benchmark and compact region supervision."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from PIL import Image
from oa_groundrag.vlm.shared_segmentation.contracts import AuxiliaryObservation, SCHEMA, validate_auxiliaries
from oa_groundrag.runtime.contracts import reject_test_or_sealed_path

@dataclass
class JointSample:
    record_id: str
    parent_id: str
    source: str
    split: str
    image: Image.Image
    masks: list[np.ndarray]
    instruction: str
    description: dict | None = None
    authority: str = "segmentation_label"
    auxiliaries: dict[str, AuxiliaryObservation] = field(default_factory=dict)
    mask_origin: str = "ground_truth"
    provenance: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.split not in {"train", "val"} or not all((self.record_id,self.parent_id,self.source)):
            raise ValueError("Only identified train/val records are permitted")
        if self.mask_origin not in {"ground_truth", "prediction", "counterfactual"}:
            raise ValueError("Unknown mask origin")
        if self.description is not None and self.mask_origin != "ground_truth":
            raise ValueError("GT descriptions cannot supervise predicted/counterfactual masks")
        hw=(self.image.height,self.image.width)
        for mask in self.masks:
            if mask.shape != hw or mask.dtype != np.bool_:
                raise ValueError("Masks must be boolean and match image dimensions")
        if self.description is not None:
            from oa_groundrag.grounding.outputs import parse_region_model_output
            description=parse_region_model_output(self.description).to_dict()
            if len(self.masks)!=1 or (description['target_status']=='target_present') != bool(self.masks[0].any()):
                raise ValueError('Description/GT mask state mismatch')
        validate_auxiliaries(self.auxiliaries, hw)


def assert_parent_isolation(samples: Sequence[JointSample]) -> None:
    seen={}
    records=set()
    for sample in samples:
        sample.validate()
        if sample.record_id in records:
            raise ValueError("Duplicate record identity")
        records.add(sample.record_id)
        key=sample.parent_id
        if key in seen and seen[key] != sample.split:
            raise ValueError(f"Parent crosses train/val: {key}")
        seen[key]=sample.split


def parent_split(parent_id: str, validation_fraction: float=0.1) -> str:
    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation fraction")
    value=int(hashlib.sha256(parent_id.encode()).hexdigest()[:16],16)/2**64
    return "val" if value < validation_fraction else "train"


def benchmark_parent_splits(root: Path | str) -> dict[str, str]:
    from oa_groundrag.data.grounded.region_contracts import parent_identity
    assignments = {}
    with (Path(root)/"index.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] not in {"train", "val"}:
                continue  # Shared index only; never open test payloads.
            parent = parent_identity(row)[0]
            split = "val" if row["split"] == "val" else parent_split(parent)
            assignments[parent] = "val" if "val" in {split, assignments.get(parent)} else "train"
    return assignments


class SegmentationDataset:
    def __init__(self, root: Path | str, split: str, modalities: tuple[str,...]=()):
        if split not in {"train","val"}:
            raise ValueError("test/sealed are not available")
        reject_test_or_sealed_path(Path(root), label="shared training input")
        from oa_groundrag.data.oa_auxseg.dataset import BenchmarkDataset
        self.dataset=BenchmarkDataset(root,split=split,auxiliary_policy="all",normalization="none")
        from oa_groundrag.data.grounded.region_contracts import parent_identity
        self.parent_assignments = benchmark_parent_splits(root)
        if split == "val":
            train = BenchmarkDataset(root,split="train",auxiliary_policy="all",normalization="none")
            self.dataset.rows += [row for row in train.rows if self.parent_assignments[parent_identity(row)[0]] == "val"]
        self.dataset.rows = [row for row in self.dataset.rows if self.parent_assignments[parent_identity(row)[0]] == split]
        self.split = split
        self.modalities=modalities

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index: int) -> JointSample:
        from oa_groundrag.data.grounded.pilot import render_optical
        from oa_groundrag.data.grounded.region_contracts import parent_identity
        row=self.dataset.rows[index]
        shard=Path(self.dataset.root)/row["storage"]["shard"]
        reject_test_or_sealed_path(shard,label="training shard")
        sample=self.dataset[index]
        image=render_optical(sample)
        aux={}
        for name in self.modalities:
            if name not in sample["auxiliaries"]:
                continue
            item=sample["auxiliaries"][name]
            valid=item["pixel_valid"].bool() & item["channel_valid"].bool().reshape(-1,1,1)
            aux[name]=AuxiliaryObservation(item["values"],valid,str(row["source"]),
                "rgb_grid_verified","source_native",False)
        target=sample["mask"][0].numpy().astype(bool)
        result=JointSample(str(row["sample_id"]),parent_identity(row)[0],str(row["source"]),self.split,
                           image,[target],"Locate all candidate landslide regions. Return masks; do not confirm landslides.",auxiliaries=aux,provenance={"source_split":row["split"],"benchmark_record_sha256":row["record_sha256"],"parent_identity_status":parent_identity(row)[1].value})
        result.validate()
        return result


class CompactRegionDataset:
    def __init__(self, root: Path | str, benchmark_root: Path | str):
        self.parent_assignments = benchmark_parent_splits(benchmark_root)
        from oa_groundrag.artifacts.identity import sha256_file
        self.root=Path(root)
        reject_test_or_sealed_path(self.root,label="compact supervision")
        manifest=json.loads((self.root/'manifest.json').read_text())
        if manifest.get('split') != 'train':
            raise ValueError('Compact source must be train-only')
        path=self.root/'messages.jsonl'
        if sha256_file(path) != manifest['messages']['sha256']:
            raise ValueError('Compact messages identity mismatch')
        self.rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(self.rows)!=manifest['record_count']:
            raise ValueError('Compact count mismatch')
        self.manifest=manifest

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index: int) -> JointSample:
        from oa_groundrag.grounding.outputs import parse_region_model_output
        row=self.rows[index]
        images=[Path(c['image']) for c in row['messages'][0]['content'] if c['type']=='image']
        if len(images) not in {2,3}:
            raise ValueError('Expected original image, GT region and optional context crop')
        for path in images:
            reject_test_or_sealed_path(path,label='compact image')
        # Verify selected bytes and their source-record binding, without rereading all image payloads.
        from oa_groundrag.data.grounded.supervision.model_assisted import _load_collection_context
        from oa_groundrag.data.grounded.region import region_asset_identity
        from oa_groundrag.grounding.messages import build_mask_grounded_region_messages
        from oa_groundrag.artifacts.identity import canonical_json, sha256_text
        if not hasattr(self,'_collection'):
            source=self.manifest['source_collection']
            self._collection=_load_collection_context(source['root'],verify_members=False,eval_exclusion_policy='retired_identity_only')
            if self._collection.manifest_sha256!=source['manifest_sha256']:
                raise ValueError('Region collection identity mismatch')
            self._entries={e.record['record_id']:e for e in self._collection.entries}
        entry=self._entries[row['record_id']]
        if entry.record['split']!='train' or entry.record['parent_id']!=row['parent_id'] or entry.record['source']!=row['source']:
            raise ValueError('Region parent/source/split mismatch')
        if region_asset_identity(entry.member_root,entry.record['assets'])!=row['asset_identity_sha256']:
            raise ValueError('Region asset identity mismatch')
        target=row['messages'][1]['content'][0]['text']
        if sha256_text(target)!=row['assistant_target_sha256']:
            raise ValueError('Region text identity mismatch')
        expected=build_mask_grounded_region_messages(entry.record,asset_root=entry.member_root,assistant_target=target)
        if row['messages']!=expected:
            raise ValueError('Region images/text binding mismatch')
        with Image.open(images[0]) as image:
            rgb=image.convert('RGB').copy()
        with Image.open(images[1]) as mask:
            binary=np.asarray(mask.convert('L')).copy()
        if not set(np.unique(binary)).issubset({0,255}):
            raise ValueError('Compact mask is not binary')
        description=parse_region_model_output(row['messages'][1]['content'][0]['text']).to_dict()
        if (description['target_status']=='target_present') != bool(binary.any()):
            raise ValueError('Region text contradicts the GT mask')
        if len(images)!=(3 if binary.any() else 2):
            raise ValueError('Unexpected crop for target/no-target state')
        result=JointSample(row['record_id'],row['parent_id'],row['source'],self.parent_assignments[row['parent_id']],
                           rgb,[binary>0],"Describe and segment this candidate landslide region.",description,
                           row['supervision_authority'],provenance={"source_split":"train","asset_identity_sha256":row['asset_identity_sha256'],"supervision_identity_sha256":row['source_supervision_identity_sha256']})
        result.validate()
        return result


def export_samples(samples: Sequence[JointSample], output: Path) -> dict:
    """New self-contained small artifact; never overwrite an existing directory."""
    from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
    from oa_groundrag.artifacts.identity import sha256_file
    assert_parent_isolation(samples)
    with AtomicArtifactDirectory(Path(output)) as writer:
        output=writer.staging
        records=[]
        for i,s in enumerate(samples):
            s.image.save(output/f'{i}.png')
            arrays={f'mask_{j}':mask for j,mask in enumerate(s.masks)}
            for name,obs in s.auxiliaries.items():
                arrays[f'{name}_values']=obs.values.cpu().numpy()
                arrays[f'{name}_valid']=obs.valid.cpu().numpy()
            np.savez_compressed(output/f'{i}.npz',**arrays)
            records.append(dict(schema_version=SCHEMA,record_id=s.record_id,parent_id=s.parent_id,source=s.source,
                split=s.split,image=f'{i}.png',arrays=f'{i}.npz',mask_count=len(s.masks),mask_origin=s.mask_origin,provenance=s.provenance,
                instruction=s.instruction,description=s.description,authority=s.authority,
                auxiliaries={name:dict(source=o.source,alignment=o.alignment,unit=o.unit,unit_verified=o.unit_verified,
                                       sign_convention=o.sign_convention,sign_verified=o.sign_verified) for name,o in s.auxiliaries.items()}))
        (output/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
        ledger={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}
        manifest=dict(schema_version=SCHEMA,record_count=len(records),files=ledger,scientific_acceptance=False)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        writer.publish()
        return manifest


def load_samples(root: Path) -> list[JointSample]:
    from oa_groundrag.artifacts.identity import sha256_file
    root=Path(root)
    reject_test_or_sealed_path(root,label='joint artifact')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['schema_version']!=SCHEMA:
        raise ValueError('Unsupported joint artifact')
    rows=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    if any(r.get('split') not in {'train','val'} for r in rows):
        raise ValueError('Forbidden split before payload read')
    if set(p.name for p in root.iterdir() if p.is_file()) != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Joint artifact inventory mismatch')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or (root/name).is_symlink() or sha256_file(root/name)!=digest:
            raise ValueError('Joint artifact identity mismatch')
    result=[]
    for r in rows:
        if r['split'] not in {'train','val'}:
            raise ValueError('Forbidden split before payload read')
        if r['image'] not in manifest['files'] or r['arrays'] not in manifest['files']:
            raise ValueError('Unregistered payload')
        with Image.open(root/r['image']) as im:
            image=im.convert('RGB').copy()
        with np.load(root/r['arrays'],allow_pickle=False) as arrays:
            masks=[arrays[f'mask_{j}'].copy() for j in range(r['mask_count'])]
            aux={name:AuxiliaryObservation(torch.from_numpy(arrays[f'{name}_values'].copy()),
                 torch.from_numpy(arrays[f'{name}_valid'].copy()),**meta) for name,meta in r['auxiliaries'].items()}
        result.append(JointSample(r['record_id'],r['parent_id'],r['source'],r['split'],image,masks,r['instruction'],
                                  r['description'],r['authority'],aux,r['mask_origin'],r['provenance']))
    if len(result)!=manifest['record_count']:
        raise ValueError('Joint artifact count mismatch')
    assert_parent_isolation(result)
    return result
