#!/usr/bin/env python3
"""Verify numerical consistency of the curated review package (stdlib only)."""
import csv,json,math,statistics as st
from collections import Counter
from pathlib import Path

data=Path(__file__).resolve().parent.parent/'data'/'curated'
def rows(n):
    with (data/n).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def js(n):return json.loads((data/n).read_text(encoding='utf8'))
def close(actual,expected,tol=1e-6):
    if abs(actual-expected)>tol:raise AssertionError((actual,expected,tol))
cases=rows('benchmark_55_summary.csv');reps=rows('benchmark_all_repetitions.csv')
assert len(cases)==55 and len({c['case_id'] for c in cases})==55 and len(reps)==168
assert Counter(c['workload'] for c in cases)=={'main':12,'code':3,'humaneval':20,'jfleg':20}
for c in cases:
    rr=[r for r in reps if r['case_id']==c['case_id'] and r['selected_by_original_analysis']=='True']
    assert len(rr)==int(c['repetitions'])==3
    close(st.median(float(r['base_ms']) for r in rr),float(c['base_median_ms']))
    close(st.median(float(r['spec_ms']) for r in rr),float(c['spec_median_ms']))
    close(float(c['base_median_ms'])/float(c['spec_median_ms']),float(c['speedup']),1e-12)
    assert all(r['spec_eq_base']==c['spec_eq_base'] for r in rr)
assert Counter(r['selected_by_original_analysis'] for r in reps)=={'True':165,'False':3}
assert sum(c['spec_eq_base']=='True' for c in cases)==51
margins=js('margins_55.json');div=js('divergence_margins.json')
assert set(margins)=={c['case_id'] for c in cases}
assert set(div)=={c['case_id'] for c in cases if c['spec_eq_base']=='False'}
assert all(margins[c['case_id']]['eq_base']==(c['spec_eq_base']=='True') for c in cases)
assert sum(v['min_margin']<=0.012 for v in margins.values())==12
for g,expect in [('main',1.45),('code',1.16),('humaneval',1.13),('jfleg',2.21)]:
    cs=[c for c in cases if c['workload']==g]; speed=st.median(float(c['speedup']) for c in cs);close(speed,expect,.005)
    print(g,'n',len(cs),'speedup_median',round(speed,4),'copy_median',round(st.median(float(c['copy_4gram_full']) for c in cs),4))
y=[math.log(float(c['speedup'])) for c in cases]
for suffix,want in [('full',.940),('prefix_4',.627),('prefix_8',.721),('prefix_16',.806),('prefix_32',.914)]:
    v=st.correlation([float(c['copy_4gram_'+suffix]) for c in cases],y);close(v,want,.0006);print('correlation',suffix,round(v,6))

pre=js('prefill_measurements.json');warm=[v for k,v in pre.items() if k!='copy_r1'];xs=[v['launches'] for v in warm];ys=[v['blocked_ms'] for v in warm];mx,my=st.mean(xs),st.mean(ys)
slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs);intercept=my-slope*mx
close(slope,115,1);print('prefill_fit_ms_per_launch',round(slope,4),'intercept_ms',round(intercept,4),'warm_cases',len(warm),'cold_case_retained',True)
for group,expected in [('main',1.42),('code',1.15),('humaneval',1.12),('jfleg',1.99)]:
    vals=[]
    for c in cases:
        if c['workload']!=group:continue
        base=float(c['base_median_ms']);spec=float(c['spec_median_ms']);p=intercept+slope*math.ceil(int(c['prompt_tokens'])/32)
        vals.append((p+base)/(p+spec))
    close(st.median(vals),expected,.005)

e=js('energy_numeric.json');events=e['harness_events'];samples=e['power_samples_time_s_watts'];assert len(samples)==688
def event(arm,rep):return next(x['time_s'] for x in events if x['arm']==arm and x['rep_index']==rep)
b0,s0,s7=event('base',0),event('spec',0),event('spec',7)
bw=[p for t,p in samples if b0<=t<s0];sw=[p for t,p in samples if s0<=t<s7]
assert len(bw)==119 and len(sw)==62
n=(e['generated_tokens']-1)*e['repetitions'];bt=sum(e['base_repetitions_ms'])/1000;stotal=sum(e['spec_repetitions_ms'])/1000
eb=st.mean(bw)*bt/n;es=st.mean(sw)*stotal/n
close(eb,.9051,.00006);close(es,.5164,.00006);close(eb/es,1.752,.001)
print('energy_sample_counts',len(bw),len(sw),'joules_per_token',round(eb,4),round(es,4))
print('PASS: curated numerical consistency; token-level verification and full raw provenance remain outside this package')
