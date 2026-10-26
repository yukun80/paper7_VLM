"""Small real Qwen processor/model fixtures; never pretrained quality evidence."""
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from oa_groundrag.vlm.shared_segmentation.model import tiny_model
from oa_groundrag.vlm.shared_segmentation.session import SharedModelSession

TEMPLATE="""{% for message in messages %}{{ '<s> ' + message['role'] + ' ' }}{% for c in message['content'] %}{% if c['type']=='image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ c['text'] }}{% endif %}{% endfor %}{{ ' </s>' }}{% endfor %}{% if add_generation_prompt %}{{ '<s> assistant ' }}{% endif %}"""

def tiny_session():
    vocab={f'w{i}':i for i in range(64)}
    for key,value in {'<pad>':0,'<unk>':1,'</s>':2,'<s>':3,'user':4,'assistant':5,
                      '<|video_pad|>':59,'<|image_pad|>':60,'<|vision_start|>':61,
                      '<|vision_end|>':62,'<|mask|>':63}.items():
        del vocab[f'w{value}'];vocab[key]=value
    tokenizer=Tokenizer(models.WordLevel(vocab,unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='<unk>',pad_token='<pad>',eos_token='</s>',
        additional_special_tokens=['<s>','<|image_pad|>','<|video_pad|>','<|vision_start|>','<|vision_end|>','<|mask|>'])
    processor=Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(patch_size=16,temporal_patch_size=2,merge_size=2),
                              tokenizer=fast,video_processor=Qwen3VLVideoProcessor(patch_size=16),chat_template=TEMPLATE)
    model=tiny_model()
    model.generation_config.eos_token_id=2
    model.generation_config.pad_token_id=0
    return SharedModelSession(model,processor,max_pixels=1024,max_new_tokens=8)


def tensor_batch(two_images=False):
    ids=torch.tensor([[1,61,60,62,5,63,2]])
    grid=torch.tensor([[1,2,2]])
    if two_images:
        ids=torch.tensor([[1,61,60,62,61,60,60,62,5,63,63,2]])
        grid=torch.tensor([[1,2,2],[1,2,4]])
    pos=(ids[0]==63).nonzero().flatten()[None]
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=torch.randn(int(grid.prod(1).sum()),1536),
                image_grid_thw=grid,mm_token_type_ids=(ids==60).long(),mask_positions=pos,
                mask_image_index=torch.tensor([[0,1]]) if two_images else torch.tensor([[0]]),
                labels=ids.clone(),use_cache=False)
