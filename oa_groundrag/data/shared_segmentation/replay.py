"""Reuse the existing task-aware RS-GeneralDesc renderer without inventing masks."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image
from oa_groundrag.runtime.contracts import reject_test_or_sealed_path

@dataclass
class TextReplaySample:
    record_id: str
    parent_id: str
    source: str
    split: str
    messages: list[dict[str,Any]]
    images: list[Image.Image]
    authority: str = 'rs_generaldesc_text_only'

    def validate(self):
        if self.split not in {'train','val'} or not self.parent_id:
            raise ValueError('Replay must preserve an identified train/val parent')
        if not self.messages or self.messages[-1]['role']!='assistant':
            raise ValueError('Replay requires an explicit text target')
        if any('<|mask|>' in str(c.get('text','')) for m in self.messages for c in m['content']):
            raise ValueError('General text replay cannot claim mask supervision')

class RSGeneralReplay:
    def __init__(self,root: Path,derived_root: Path, *, split='train'):
        from oa_groundrag.data.rs_general.dataset import RSGeneralDescDataset
        if split not in {'train','val'}:raise ValueError('Replay rejects test/sealed')
        reject_test_or_sealed_path(root,label='general replay input')
        self.dataset=RSGeneralDescDataset(root,roles=[f'external_{split}'],load_assets=False)
        self.root=Path(root).resolve()
        self.derived_root=Path(derived_root).resolve()
        if self.derived_root.exists():raise FileExistsError(self.derived_root)
        self.derived_root.mkdir(parents=True)
        self.split=split

    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        from oa_groundrag.data.rs_general.exporter import render_canonical_messages
        record=self.dataset.records[index]
        if record['logical_role']!=f'external_{self.split}':raise ValueError('Replay role mismatch')
        rendered,_=render_canonical_messages(record,purpose='training' if self.split=='train' else 'validation',
            seed=20261026,dataset=self.dataset,derived_root=self.derived_root)
        messages=rendered['messages'];images=[]
        for message in messages:
            for item in message['content']:
                if item['type']=='image':
                    if 'benchmark_asset' in item:
                        path=self.dataset.resolve_verified_asset_path(str(item['benchmark_asset']))
                    elif 'export_asset' in item:
                        relative=Path(item['export_asset'])
                        if relative.is_absolute() or '..' in relative.parts:
                            raise ValueError('Derived replay path escape')
                        path=self.derived_root/relative
                        if path.is_symlink() or not path.is_file():
                            raise ValueError('Derived replay image must be a regular file')
                    else:
                        raise ValueError('Replay image has no registered asset')
                    reject_test_or_sealed_path(path,label='replay image')
                    with Image.open(path) as image:im=image.convert('RGB').copy()
                    item['image']=im;images.append(im)
        result=TextReplaySample(record['record_id'],record['parent_id'],str(record.get('source','rs_generaldesc')),
                                self.split,messages,images)
        result.validate();return result
