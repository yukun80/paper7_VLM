"""Development-only paired segmentation metrics; no scientific Gate promotion."""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label


def segmentation_metrics(predicted: np.ndarray, truth: np.ndarray, *, boundary_tolerance=1) -> dict:
    if predicted.shape!=truth.shape or predicted.dtype!=np.bool_ or truth.dtype!=np.bool_ or predicted.ndim!=2:
        raise ValueError('Expected aligned boolean masks')
    if boundary_tolerance<0:raise ValueError('Negative boundary tolerance')
    tp=int((predicted & truth).sum());fp=int((predicted & ~truth).sum());fn=int((~predicted & truth).sum())
    pborder=predicted ^ binary_erosion(predicted);tborder=truth ^ binary_erosion(truth)
    if not predicted.any() and not truth.any():bf1=1.0
    elif not pborder.any() or not tborder.any():bf1=0.0
    else:
        p_near=binary_dilation(pborder,iterations=boundary_tolerance) if boundary_tolerance else pborder
        t_near=binary_dilation(tborder,iterations=boundary_tolerance) if boundary_tolerance else tborder
        precision=float((pborder&t_near).sum()/pborder.sum());recall=float((tborder&p_near).sum()/tborder.sum())
        bf1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    components,n=label(truth)
    sizes=[int((components==i).sum()) for i in range(1,n+1)]
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 1.0,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
                false_positive_pixels=fp,false_negative_pixels=fn,boundary_f1=bf1,
                no_target_correct=bool(not predicted.any()) if not truth.any() else None,
                truth_component_sizes=sizes,component_recalls=[float((predicted&(components==i)).sum()/sizes[i-1]) for i in range(1,n+1)])


def compare_paired(rows_a,rows_b):
    """Require identical sample/truth identities; fail rather than drop incomplete pairs."""
    a={r['record_id']:r for r in rows_a};b={r['record_id']:r for r in rows_b}
    if len(a)!=len(rows_a) or len(b)!=len(rows_b) or set(a)!=set(b) or not a:
        raise ValueError('Paired records differ or contain duplicates')
    for key in a:
        if a[key]['truth_sha256']!=b[key]['truth_sha256']:raise ValueError('Paired truth differs')
        if a[key]['split']!='val' or b[key]['split']!='val':raise ValueError('Development comparison accepts val only')
    return {'record_count':len(a),'iou_delta':float(np.mean([b[k]['metrics']['iou']-a[k]['metrics']['iou'] for k in a])),
            'scientific_acceptance':False}


def score_artifact(root,sample):
    from pathlib import Path
    import json,hashlib
    from PIL import Image
    from oa_groundrag.artifacts.identity import sha256_file
    from oa_groundrag.runtime.contracts import reject_test_or_sealed_path
    sample.validate()
    if sample.split!='val' or sample.mask_origin!='ground_truth' or not sample.masks:
        raise ValueError('Development scoring requires val ground truth')
    root=Path(root);reject_test_or_sealed_path(root,label='development prediction')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or (root/name).is_symlink() or sha256_file(root/name)!=digest:
            raise ValueError('Prediction artifact identity mismatch')
    prediction=json.loads((root/'result.json').read_text())
    image_sha=hashlib.sha256(np.asarray(sample.image.convert('RGB')).tobytes()).hexdigest()
    if prediction['record_id']!=sample.record_id or prediction['image_sha256']!=image_sha:
        raise ValueError('Prediction belongs to a different input')
    if prediction['status'] not in {'target_present','no_target'}:
        raise ValueError('Incomplete/failed predictions cannot be scored as no-target')
    mask=np.zeros((sample.image.height,sample.image.width),bool)
    if (prediction['status']=='target_present')!=bool(prediction['regions']):
        raise ValueError('Prediction region/status mismatch')
    for region in prediction['regions']:
        name=region['mask_path']
        if name not in manifest['files']:raise ValueError('Unregistered prediction mask')
        with Image.open(root/name) as image:array=np.asarray(image.convert('L'))
        if array.shape!=mask.shape or not set(np.unique(array)).issubset({0,255}):
            raise ValueError('Invalid prediction mask')
        mask|=array>0
    truth=np.logical_or.reduce(sample.masks)
    return {'record_id':sample.record_id,'split':'val','truth_sha256':hashlib.sha256(truth.tobytes()).hexdigest(),
            'prediction_manifest_sha256':sha256_file(root/'manifest.json'),'metrics':segmentation_metrics(mask,truth),
            'scientific_acceptance':False}
