#!/usr/bin/env python3
"""Analyze sustained KV-pressure tests and causally replay a learned rate correction.

Reads recorded files only. The experimental learner is never a serving change.
"""
from pathlib import Path
from collections import Counter, deque
from bisect import bisect_right
import argparse
import csv
import json
import math
import os
import statistics
from datetime import datetime

from plot_wait_metrics import recalculate

WINDOW=32
MIN_SAMPLES=4
CAPACITY_FLOOR=2000

def finite(value):
    try:return math.isfinite(float(value))
    except (ValueError,TypeError):return False

def number(value):return float(value) if finite(value) else math.nan
def yes(value):return value is True or value=='True'
def median(values,default=math.nan):return statistics.median(values) if len(values)>=MIN_SAMPLES else default
def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value

def events(path):
    forecasts={};texts={}
    for line in (path/'router-events.jsonl').read_text().splitlines():
        try:e=json.loads(line)
        except ValueError:continue
        e=e.get('fields',{})|e
        dest=forecasts if e.get('event')=='forecast' else texts if e.get('event')=='first_text' else None
        if dest is not None:
            rid=e['request_id']
            if rid in dest:raise ValueError(f'Duplicate event {rid}')
            dest[rid]=e
    return forecasts,texts

def ratio(f,actual):
    """Only identify native queue-rate bias where that term dominates the floor."""
    q=f['token_work']/f['effective_capacity']
    if f['ready'] and f.get('capacity_learned_ready',f['native_ready']) and not f.get('measured_delay_only',False) and f['baseline_available'] and f['native_waiting']>0 and q>0 and q>=f['observed_delay_floor']:
        return max(0,actual-f['baseline_seconds'])/q
    return None

def prior_state(path):
    values=deque(maxlen=WINDOW)
    if path and path.exists():
        forecasts,_=events(path)
        requests=list(csv.DictReader((path/'requests.tsv').open(),delimiter='\t'))
        completed={r['opc_request_id']:float(r['started_elapsed_seconds'])+float(r['e2e_seconds']) for r in requests}
        paired=json.loads((path/'paired-requests.json').read_text())
        for r in sorted(paired,key=lambda r:completed[r['request_id']]):
            v=ratio(forecasts[r['request_id']],r['actual_ttft_seconds'])
            if v is not None:values.append(v)
    return values

def confusion(rows,key):
    counts=Counter();unavailable=0
    for r in rows:
        y=r['deadline_miss']
        if y is None:continue
        p=r.get(key)
        if p is None:unavailable+=1;p=False
        counts['tp' if y and p else 'fn' if y else 'fp' if p else 'tn']+=1
    neg=counts['tn']+counts['fp'];pos=counts['tp']+counts['fn']
    return dict(counts,healthy=neg,late=pos,unknown_prediction_allowed=unavailable,
                fpr=counts['fp']/neg if neg else None,fnr=counts['fn']/pos if pos else None)

def high_duration(metrics,threshold=.9):
    longest=0;total=0;start=None;previous=None
    for row in metrics:
        now=number(row['sample_completed_elapsed_seconds'])
        high=number(row.get('vllm_kv_usage'))>=threshold and number(row.get('vllm_running'))>0
        if high:
            if start is None or previous is None or now-previous>3:start=now
            else:total+=now-previous
            longest=max(longest,now-start)
        else:start=None
        previous=now
    return dict(seconds=total,longest_contiguous_seconds=longest)

def analyze(path,prior=None):
    meta=json.loads((path/'summary.json').read_text())
    if meta['error']:raise ValueError(f'Incomplete run: {path}: {meta["error"]}')
    metrics=list(csv.DictReader((path/'metrics.tsv').open(),delimiter='\t'))
    if prior and prior.exists():
        old_metrics=list(csv.DictReader((prior/'metrics.tsv').open(),delimiter='\t'))
        if datetime.fromisoformat(old_metrics[-1]['timestamp'])>=datetime.fromisoformat(metrics[0]['timestamp']):
            raise ValueError('Historical calibration must end before this test begins')
    requests=list(csv.DictReader((path/'requests.tsv').open(),delimiter='\t'))
    forecasts,texts=events(path)
    for f in forecasts.values():
        if f['threshold_seconds']!=300:raise ValueError('Unexpected router admission budget')
        calculated,_=recalculate(f)
        if abs(calculated-f['predicted_ttft_seconds'])>1e-8:raise ValueError('Formula replay mismatch')
    times=[number(r['sample_completed_elapsed_seconds']) for r in metrics]
    rates=[]
    for i,row in enumerate(metrics):
        value=math.nan
        if i:
            dt=times[i]-times[i-1]
            delta=number(row.get('generation_tokens'))-number(metrics[i-1].get('generation_tokens'))
            if dt>0 and delta>0:value=delta/dt
        rates.append(value)
    history=prior_state(prior)
    initial_factor=median(history,1.0)
    idle_decode=deque(maxlen=WINDOW)
    previous=[];rows=[];last_update=None
    for request in sorted(requests,key=lambda r:float(r['started_elapsed_seconds'])):
        started=float(request['started_elapsed_seconds'])
        # Completion ordering prevents using a slow request's outcome prematurely.
        matured=[r for r in previous if r['end']<=started]
        previous=[r for r in previous if r['end']>started]
        for old in sorted(matured,key=lambda r:r['end']):
            last_update=old['end']
            if old['f'] and finite(old['actual_ttft']):
                value=ratio(old['f'],old['actual_ttft'])
                if value is not None:history.append(value)
                if old['native_running_at_start']<=1 and old['f']['native_waiting']==0 and old['output_tokens']>0:
                    decode=max(0,old['e2e']-old['client_ttft'])/old['output_tokens']
                    if finite(decode) and decode>0:idle_decode.append(decode)
        rid=request['opc_request_id'];f=forecasts.get(rid);text=texts.get(rid)
        index=bisect_right(times,started)-1
        native_running=number(metrics[index].get('vllm_running')) if index>=0 else math.nan
        background=request['case']=='background'
        complete=yes(request['complete']);cancelled=yes(request['cancelled'])
        status=int(request['status']);elapsed=float(request['e2e_seconds'])
        if status==503 and (f is None or not f['would_reject']):raise ValueError('503 did not match an SMG admission rejection')
        timeout='TimeoutError' in request.get('error','')
        truth=(elapsed>300) if (status==200 or timeout) and (complete or elapsed>300) and (not cancelled or elapsed>300) else None
        row=dict(request_id=rid,case=request['case'],phase=request['phase_at_start'],start=started,
            end=float(request['completed_elapsed_seconds']),status=status,complete=complete,cancelled=cancelled,
            background=background,e2e=elapsed,deadline_miss=truth,
            client_ttft=number(request['ttft_seconds']),actual_ttft=text['ttft_seconds'] if text else math.nan,
            output_tokens=number(request['actual_output_tokens']),output_limit=int(request['output_tokens']),
            prompt_tokens=number(request['actual_prompt_tokens']),f=f,native_running_at_start=native_running,
            native_sample_available_elapsed_seconds=times[index] if index>=0 else None)
        if index>=0 and times[index]>started:raise ValueError('Future metric sample leaked into prediction')
        if not background and status in (200,503) and f is None:raise ValueError(f'Missing probe forecast {rid}')
        factor=median(history,1.0);row.update(learned_factor=factor,learned_samples=len(history),latest_training_completion=last_update)
        if last_update is not None and last_update>started:raise ValueError('Future completion leaked into calibration')
        if f:
            capacity=max(CAPACITY_FLOOR,f['effective_capacity']/factor) if factor>0 else math.inf
            learned=(f['baseline_seconds']+max(f['token_work']/capacity,f['observed_delay_floor'])
                     if not f['measured_delay_only'] else f['predicted_ttft_seconds'])
            row.update(raw_ttft=f['predicted_ttft_seconds'],learned_ttft=learned,learned_capacity=capacity,
                smg_reject=f['would_reject'],learned_queue_reject=bool(f['ready'] and learned-f['baseline_seconds']>=300))
            aggregate=rates[index] if index>=0 else math.nan
            seconds_per_token=(max(1,native_running)/aggregate if finite(aggregate) and aggregate>0 and finite(native_running) else median(idle_decode))
            predicted_e2e=learned+row['output_limit']*seconds_per_token
            row.update(predicted_e2e=predicted_e2e,
                learned_completion_reject=bool(f['ready'] and predicted_e2e>300) if finite(predicted_e2e) else None,
                generation_tps_available_before_start=aggregate)
        if not background and complete and status==200:previous.append(row)
        rows.append(row)
    probes=[r for r in rows if r['case'].startswith(('low','pressure','recovery'))]
    grouped={}
    for prefix in ['low','pressure','recovery']:
        subset=[r for r in probes if r['case'].startswith(prefix)]
        paired=[r for r in subset if finite(r['actual_ttft']) and r['f'] and r['f']['baseline_available']]
        grouped[prefix]=dict(requests=len(subset),completed=sum(r['complete'] for r in subset),
            rejected=sum(r['status']==503 for r in subset),late=sum(r['deadline_miss'] is True for r in subset),
            unknown_outcomes=sum(r['deadline_miss'] is None for r in subset),
            smg_gate=confusion(subset,'smg_reject'),learned_queue_gate=confusion(subset,'learned_queue_reject'),
            experimental_completion_gate=confusion(subset,'learned_completion_reject'),
            ready=sum(bool(r['f'] and r['f']['ready']) for r in subset),
            native_capacity_ready=sum(bool(r['f'] and r['f']['capacity_learned_ready']) for r in subset),
            learned_factor_min=min((r['learned_factor'] for r in subset),default=None),
            learned_factor_max=max((r['learned_factor'] for r in subset),default=None),
            actual_prompt_usage_samples=sum(finite(r['prompt_tokens']) for r in subset),
            actual_prompt_tokens_median=statistics.median(r['prompt_tokens'] for r in subset if finite(r['prompt_tokens'])) if any(finite(r['prompt_tokens']) for r in subset) else None,
            queue_proxy_tokens_per_waiting_median=statistics.median(r['f']['queued_tokens']/r['f']['native_waiting'] for r in subset if r['f'] and r['f']['native_waiting']>0) if any(r['f'] and r['f']['native_waiting']>0 for r in subset) else None,
            observed_first_text_over_deadline=sum(finite(r['actual_ttft']) and r['actual_ttft']>300 for r in subset),
            first_text_unobserved=sum(not finite(r['actual_ttft']) for r in subset if r['status']!=503),
            ttft_pairs=len(paired),raw_ttft_mae=statistics.mean(abs(r['raw_ttft']-r['actual_ttft']) for r in paired) if paired else None,
            learned_ttft_mae=statistics.mean(abs(r['learned_ttft']-r['actual_ttft']) for r in paired) if paired else None,
            actual_ttft_mean=statistics.mean(r['actual_ttft'] for r in paired) if paired else None,
            estimated_ttft_mean=statistics.mean(r['raw_ttft'] for r in paired) if paired else None,
            admission_snapshot_age_max=max((r['f']['snapshot_age_seconds'] for r in subset if r['f']),default=None),
            completed_e2e_p50=statistics.median(r['e2e'] for r in subset if r['complete']) if any(r['complete'] for r in subset) else None)
        bands=[('well_within',0,240),('near_healthy',240,300),('near_late',300,360),('strongly_late',360,math.inf)]
        grouped[prefix]['deadline_bands']={name:confusion([r for r in subset if r['deadline_miss'] is not None and low<r['e2e']<=high and (r['complete'] or name=='strongly_late')],'smg_reject') for name,low,high in bands}
    peak=max((number(r.get('vllm_kv_usage')) for r in metrics if finite(r.get('vllm_kv_usage'))),default=math.nan)
    memory_key='retractions_sum' if meta['engine']=='sglang' else 'preemptions'
    memory_values=[number(r.get(memory_key)) for r in metrics if finite(r.get(memory_key))]
    queued_high_kv_seconds=0;inactive_learning_seconds=0
    for before,after in zip(metrics,metrics[1:]):
        dt=number(after['sample_completed_elapsed_seconds'])-number(before['sample_completed_elapsed_seconds'])
        if 0<dt<=3 and number(before.get('vllm_waiting'))>0 and number(before.get('vllm_kv_usage'))>=.9:
            queued_high_kv_seconds+=dt
            if number(before.get('capacity_learning_active'))==0:inactive_learning_seconds+=dt
    result=dict(model=path.parent.name,mode=path.name,requests=len(rows),probe_requests=len(probes),
        status_counts=dict(Counter(r['status'] for r in rows)),max_kv_usage=peak,
        rejected_by_case=dict(Counter(r['case'] for r in rows if r['status']==503)),
        high_kv=high_duration(metrics),deadline_seconds=300,router_extra_ttft_budget_seconds=300,
        learning=dict(window=WINDOW,min_samples=MIN_SAMPLES,capacity_floor=CAPACITY_FLOOR,
            initial_factor=initial_factor,last_factor=rows[-1]['learned_factor'] if rows else None,
            prior_path=str(prior) if prior else None,policy='Analysis-only causal median native-queue rate correction; decode estimate uses aggregate generation rate divided by native running count.'),
        phases=grouped,forecast_events=len(forecasts),first_text_events=len(texts),
        preemption_metric=memory_key,
        memory_events_delta=memory_values[-1]-memory_values[0] if memory_values and memory_values[-1]>=memory_values[0] else None,
        capacity_context=dict(queued_high_kv_seconds=queued_high_kv_seconds,
            without_active_positive_prefill_learning_seconds=inactive_learning_seconds),
        client_minus_smg_first_text_p50=statistics.median(r['client_ttft']-r['actual_ttft'] for r in probes if finite(r['client_ttft']) and finite(r['actual_ttft'])) if any(finite(r['client_ttft']) and finite(r['actual_ttft']) for r in probes) else None,
        notes=['All formula forecasts replayed exactly.',
               'Completion-time evaluation waits for completed requests; a request still open beyond 300s is a known deadline miss.',
               'Background cancellations and enforced rejections have no counterfactual full completion label.',
               'The online learner and completion estimator were not deployed. Native fair-share decode is a diagnostic approximation, not a deadline guarantee.'])
    return meta,metrics,rows,result

def draw(meta,metrics,rows,result,out):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    label=f"{result['model']} / {meta['engine']} / {result['mode']}"
    probes=[r for r in rows if r['case'].startswith(('low','pressure','recovery'))]
    xx=[number(r['sample_completed_elapsed_seconds']) for r in metrics]
    fig,axes=plt.subplots(9,1,figsize=(16,24),sharex=True,layout='constrained')
    def sampled(ax,key,name,color=None,zero_at_start=False):
        # These gauges and counters are nonnegative; -1 means unavailable.
        yy=[number(r.get(key)) if number(r.get(key))>=0 else math.nan for r in metrics]
        if zero_at_start:
            origin=next((v for v in yy if finite(v)),0)
            yy=[v-origin if finite(v) else v for v in yy]
        if any(finite(v) for v in yy):ax.step(xx,yy,where='post',label=name,color=color,linewidth=1)
    for key,name,color in [('actual_ttft','Actual SMG first-text TTFT','#2563eb'),('raw_ttft','Recorded SMG TTFT estimate','#d97706'),('learned_ttft','Learned rate: analysis replay','#059669')]:
        rs=[r for r in probes if finite(r.get(key))]
        axes[0].scatter([r['start'] for r in rs],[r[key] for r in rs],s=10,label=name,alpha=.6,color=color)
    done=[r for r in probes if r['complete']]
    axes[1].scatter([r['start'] for r in done],[r['e2e'] for r in done],s=10,label='Actual full completion',color='#2563eb')
    predicted=[r for r in probes if finite(r.get('predicted_e2e'))]
    axes[1].scatter([r['start'] for r in predicted],[r['predicted_e2e'] for r in predicted],s=10,label='Experimental completion estimate',color='#059669',alpha=.5)
    axes[1].axhline(300,color='#b91c1c',label='Full-response deadline: 300s')
    rejected=[r for r in probes if r['status']==503]
    axes[1].scatter([r['start'] for r in rejected],[300]*len(rejected),marker='x',color='#b91c1c',label='Rejected: actual completion unknown',s=22)
    censored=[r for r in probes if not r['complete'] and r['deadline_miss'] is True]
    axes[1].scatter([r['start'] for r in censored],[r['e2e'] for r in censored],marker='^',facecolors='none',edgecolors='#2563eb',label='Observed lower bound: incomplete',s=22)
    sampled(axes[2],'vllm_kv_usage','Native KV/token usage','#7c3aed');sampled(axes[2],'kv_usage','SMG cached KV/token usage','#94a3b8')
    axes[2].axhline(.9,color='#b91c1c',linestyle='--',label='90% test target')
    for key,name in [('vllm_waiting','Native waiting'),('vllm_running','Native running'),('client_background_active','Client background in flight')]:sampled(axes[3],key,name)
    for key,name in [('generation_tps','SMG generation TPS'),('prefill_tps','SMG measured prefill TPS')]:sampled(axes[4],key,name)
    traced=[r for r in probes if r['f']]
    for key,name in [('queued_tokens','Q: recorded queue proxy'),('token_work','W: recorded Q + 0.05 D')]:
        axes[5].scatter([r['start'] for r in traced],[r['f'][key] for r in traced],s=8,label=name,alpha=.5)
    sampled(axes[6],'effective_capacity','SMG effective prefill C')
    sampled(axes[6],'learned_capacity','SMG learned prefill capacity')
    corrected=[r for r in traced if finite(r['learned_capacity'])]
    axes[6].scatter([r['start'] for r in corrected],[r['learned_capacity'] for r in corrected],s=8,label='Analysis-only corrected C',alpha=.5)
    for key,name in [('baseline_seconds','B: measured low-load TTFT'),('observed_delay_floor','F: observed delay floor'),('native_queue_delay','Native recent queue delay')]:sampled(axes[7],key,name)
    sampled(axes[8],'retractions_sum' if meta['engine']=='sglang' else 'preemptions','Native retractions/preemptions since test start',zero_at_start=True)
    sampled(axes[8],'retracted','Currently retracted requests')
    titles=[('Same request at admission: predicted and actual first text','Seconds'),('Full response completion: different target from TTFT','Seconds'),
        ('Reported native KV/token usage','Fraction'),('Resident work and queue','Requests'),('Engine rates','Tokens/s'),
        ('Recorded arithmetic numerator','Tokens'),('Prefill rate: actual and experimental','Tokens/s'),('Delay components','Seconds'),('Memory pressure events','Count')]
    for ax,(title,unit) in zip(axes,titles):
        ax.set_title(title,loc='left',pad=45);ax.set_ylabel(unit);ax.grid(alpha=.2)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncol=3,fontsize=8)
        if isinstance(ax.yaxis.get_major_formatter(),ScalarFormatter):ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.tick_params(axis='x',labelbottom=True)
        for index,stage in enumerate(meta['stages']):
            end=meta['stages'][index+1]['elapsed_seconds'] if index+1<len(meta['stages']) else xx[-1]
            if stage['name']=='sustained-pressure':ax.axvspan(stage['elapsed_seconds'],end,color='#fed7aa',alpha=.25)
    axes[-1].set_xlabel('Seconds since test start; durations plotted at their request admission')
    fig.suptitle(label+' · Sustained KV pressure / 300-second completion deadline\nLearning curves are causal analysis replays; orange shading marks sustained background load',fontsize=15)
    fig.savefig(out/f"{result['model']}-{result['mode']}-pressure-timeline.png",dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for ax,key,title,target in [(axes[0],'raw_ttft','Recorded SMG TTFT','actual_ttft'),(axes[1],'learned_ttft','Causal learned-rate TTFT replay','actual_ttft'),(axes[2],'predicted_e2e','Experimental completion forecast','e2e')]:
        selected=[r for r in probes if r['complete'] and finite(r.get(key)) and finite(r.get(target))]
        limit=max([1]+[max(r[key],r[target]) for r in selected])*1.03
        for prefix,color in [('low','#059669'),('pressure','#d97706'),('recovery','#2563eb')]:
            rs=[r for r in selected if r['case'].startswith(prefix)]
            ax.scatter([r[target] for r in rs],[r[key] for r in rs],s=15,label=prefix,color=color,alpha=.6)
        ax.plot([0,limit],[0,limit],color='#64748b',linewidth=1)
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Actual seconds',ylabel='Predicted seconds',title=title)
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle(label+' · Accuracy by traffic phase; completed probes only')
    fig.savefig(out/f"{result['model']}-{result['mode']}-accuracy.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(16,14),sharex=True,layout='constrained')
    axes[0].scatter([r['start'] for r in probes],[r['learned_factor'] for r in probes],s=10,label='Causal queue-delay scale from prior completions')
    axes[0].axhline(1,color='#64748b',linestyle='--',label='Identity correction')
    for key,name in [('capacity_samples','Qualified capacity intervals'),('prefill_size_samples','Completed-prompt size samples')]:sampled(axes[1],key,name)
    for key,name in [('capacity_ready','SMG capacity/admission readiness'),('capacity_learning_active','Current saturated learning interval')]:sampled(axes[2],key,name)
    for key,name in [('mean_prefill_tokens','SMG mean uncached prompt size'),('median_prefill_tokens','SMG recent prompt size'),('blended_prompt_tokens','Q proxy tokens per waiting request')]:sampled(axes[3],key,name)
    sized=[r for r in probes if r['complete'] and finite(r['prompt_tokens'])]
    axes[3].scatter([r['start'] for r in sized],[r['prompt_tokens'] for r in sized],s=8,color='#7c3aed',label='Actual input tokens after completion: diagnostic')
    observed=[r for r in probes if r['f']]
    axes[4].scatter([r['start'] for r in observed],[r['f']['snapshot_age_seconds'] for r in observed],s=8,label='Exact SMG snapshot age at admission')
    sampled(axes[4],'snapshot_age','Scraped age gauge (last refresh value)')
    for ax,title,unit in zip(axes,['Learned relative gap: no model-specific assigned factor','Learning evidence available to SMG','Readiness and learning activity','Queue-size history versus completed-request input size','Snapshot freshness'],['Scale','Samples','0/1','Tokens','Seconds']):
        ax.set_title(title,loc='left',pad=45);ax.set_ylabel(unit);ax.grid(alpha=.2)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncol=3,fontsize=8);ax.tick_params(axis='x',labelbottom=True)
    axes[-1].set_xlabel('Seconds since test start')
    fig.suptitle(label+' · Learning evidence and causal calibration',fontsize=15)
    fig.savefig(out/f"{result['model']}-{result['mode']}-learning-context.png",dpi=130);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--prior',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    os.environ['MPLBACKEND']='Agg';os.environ.setdefault('MPLCONFIGDIR',str(a.output_dir/'.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(a.output_dir/'.cache'))
    results=[]
    for file in sorted(a.input.glob('*/*/summary.json')):
        path=file.parent;prior=a.prior/path.parent.name/'train' if a.prior else None
        meta,metrics,rows,result=analyze(path,prior);draw(meta,metrics,rows,result,a.output_dir)
        (a.output_dir/f"{result['model']}-{result['mode']}-analysis.json").write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
        probes=[r for r in rows if not r['background']]
        (a.output_dir/f"{result['model']}-{result['mode']}-paired-probes.json").write_text(json.dumps(clean(probes),indent=2,allow_nan=False)+'\n')
        results.append(result)
        print(json.dumps(dict(model=result['model'],mode=result['mode'],peak_kv=result['max_kv_usage'],
            high_kv_seconds=result['high_kv']['seconds'],phases={name:dict(requests=v['requests'],late=v['late'],rejected=v['rejected'],smg=v['smg_gate']) for name,v in result['phases'].items()})),flush=True)
    (a.output_dir/'pressure-analysis.json').write_text(json.dumps(clean(results),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
