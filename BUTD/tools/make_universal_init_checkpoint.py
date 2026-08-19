import argparse
from collections import OrderedDict
import torch

p=argparse.ArgumentParser()
p.add_argument('--base',required=True)
p.add_argument('--template',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
base=torch.load(a.base,map_location='cpu')
tpl=torch.load(a.template,map_location='cpu')
bs=base['model']; ts=tpl['model']
def strip(k): return k[7:] if k.startswith('module.') else k
bmap={strip(k):v for k,v in bs.items()}
new=OrderedDict(); copied=[]; fresh=[]; mismatch=[]
for tk,tv in ts.items():
    sk=strip(tk)
    if sk in bmap and getattr(bmap[sk],'shape',None)==getattr(tv,'shape',None):
        new[tk]=bmap[sk]; copied.append(sk)
    else:
        new[tk]=tv; fresh.append(sk)
        if sk in bmap: mismatch.append((sk,tuple(bmap[sk].shape),tuple(tv.shape)))
allowed_prefixes=('structured_slot_builder.','sacr_head.','reliability_fusion.','quality_head.')
disallowed=[k for k in fresh if not k.startswith(allowed_prefixes)]
unused=[k for k in bmap if k not in {strip(x) for x in ts}]
if disallowed:
    raise SystemExit('fresh keys outside universal modules: '+repr(disallowed))
out={'config':tpl.get('config'),'save_path':a.output,'model':new,'optimizer':None,'scheduler':None,'epoch':-1,
     'initialization_audit':{'base':a.base,'template':a.template,'copied_tensor_count':len(copied),'fresh_tensor_count':len(fresh),'fresh_keys':fresh,'shape_mismatch':mismatch,'unused_base_keys':unused}}
torch.save(out,a.output)
print('copied',len(copied),'fresh',len(fresh),'unused_base',len(unused),'mismatch',len(mismatch))
for k in fresh: print('FRESH',k)
