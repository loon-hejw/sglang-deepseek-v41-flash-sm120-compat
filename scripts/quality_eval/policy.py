"""Predeclared selection, inference settings and family-level uncertainty."""
import hashlib
import json
import random
import statistics
from .cases import CATEGORIES


def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def baseline(category):
    effort={'reasoning':'high','coding':'low'}.get(category,'none')
    return {'temperature':1,'top_p':1,'reasoning_effort':effort,
            'max_tokens':12288 if category in ['reasoning','coding'] else 4096 if category=='tools' else 2048}


def profile_id(profile):
    return digest(profile)[:16]


def initial_profiles(category):
    efforts=['low','medium','high','max'] if category in ['reasoning','coding'] else ['none','low','medium']
    return [{**baseline(category),'reasoning_effort':e} for e in efforts]


def smoke_valid(rows):
    """A model's wrong answer is evidence to retain, not an API startup fault."""
    return len(rows)==5 and all(
        not r.get('error_type') and r.get('done') and r.get('checks',{}).get('complete')
        and r.get('usage',{}).get('completion_tokens',0)>0
        and (r['reasoning_chars']>0)==(r['profile']['reasoning_effort']!='none') for r in rows)


def rank_profiles(category, profiles, rows):
    def rank(p):
        r=[r for r in rows if r['category']==category and r['profile_id']==profile_id(p)]
        assert r, 'Cannot rank an unmeasured profile'
        return (-statistics.mean(x['passed'] for x in r),sum('length' in x.get('finish',[]) for x in r),
                p!=baseline(category),p['max_tokens'],['none','low','medium','high','max'].index(p['reasoning_effort']),abs(p['temperature']-1))
    return sorted(profiles,key=rank)


def quality_gate(rows, bootstrap_repeats=10000):
    """Paired by case/repetition, stratified by category, clustered by family."""
    pairs={}
    for r in rows:
        key=(r['category'],r['family'],r['case'],r['repeat'])
        assert r['arm'] not in pairs.setdefault(key,{}), 'Duplicate observation'
        pairs[key][r['arm']]=float(r['passed'])
    assert pairs and all(set(v)=={'baseline','candidate'} for v in pairs.values()),'Incomplete paired holdout'
    grouped={cat:{} for cat in CATEGORIES}
    for (cat,family,_,_), arms in pairs.items():
        grouped[cat].setdefault(family,[]).append(arms['candidate']-arms['baseline'])
    assert all(grouped.values()), 'Missing category'
    means={cat:{f:statistics.mean(v) for f,v in fam.items()} for cat,fam in grouped.items()}
    per_category={cat:statistics.mean(means[cat].values()) for cat in CATEGORIES}
    delta=statistics.mean(per_category.values())
    rng=random.Random(20261105);samples=[]
    for _ in range(bootstrap_repeats):
        samples.append(statistics.mean(statistics.mean(rng.choices(list(means[cat].values()),k=len(means[cat]))) for cat in CATEGORIES))
    samples.sort();lo=samples[int(.025*bootstrap_repeats)];hi=samples[min(len(samples)-1,int(.975*bootstrap_repeats))]
    return {'macro_delta':delta,'per_category_delta':per_category,'ci95':[lo,hi],
            'bootstrap_repetitions':bootstrap_repeats,'independent_resampling_unit':'problem family within category',
            'holdout_families':sum(len(x) for x in means.values()),
            'passed':delta>=.03 and lo>0 and all(x>=-.02-1e-9 for x in per_category.values())}
