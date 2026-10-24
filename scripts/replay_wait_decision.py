#!/usr/bin/env python3
"""Causal audit of decision-shadow telemetry; no workload or network operations."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
try:
    from . import wait_decision_calibration as d
    from .replay_wait_joint_gains import close, read_events, require, sha
except ImportError:
    import wait_decision_calibration as d
    from replay_wait_joint_gains import close, read_events, require, sha

EVENTS = {'decision_native_poll','decision_poll','decision_forecast','decision_admission','decision_label','decision_terminal'}


def payload(e):
    require(e.get('decision_method') == d.METHOD, 'Wrong or missing decision method')
    p=json.loads(e['decision_json'])
    require(isinstance(p,dict),'Decision payload is not an object')
    return p


def audit_schema(events, expected=d.METHOD):
    errors=[];counts=Counter()
    for e in events:
        if e.get('event') not in EVENTS:continue
        counts[e['event']]+=1
        try:
            require(expected==d.METHOD,'Unsupported required decision method')
            p=payload(e);kind=e['event']
            required={'decision_native_poll':('expected_rank_count','ranks'),'decision_poll':('snapshot','predictions','state','fresh','profile','max_gap','available'),
                      'decision_forecast':('snapshot','predictions','available'),
                      'decision_admission':('id','tracked','started','snapshot','predictions'),
                      'decision_label':('label',),'decision_terminal':('id','observed_at','outcome')}[kind]
            require(all(k in p for k in required),'Incomplete '+kind+' schema')
            if kind=='decision_poll':
                require(isinstance(p['state'],dict) and all(k in p['state'] for k in d.State().diagnostics()),'Missing decision state')
            if 'predictions' in p and p['predictions'] is not None:
                require([v['arm'] for v in p['predictions']]==list(d.ARMS),'Missing/reordered shadow arms')
                require(all(type(v['available']) is bool for v in p['predictions']),'Invalid availability')
        except (ValueError,KeyError,TypeError) as exc:errors.append(str(exc))
    if not counts['decision_poll']:errors.append('No decision poll method proof')
    return dict(ok=not errors,method=expected,counts=dict(counts),errors=errors[:20])


def verify_snapshot(s, state, max_gap, selected=None):
    if state.profile is not None:
        require(s['profile']==state.profile,'Snapshot profile differs from initialized state')
    else:
        require(not state.rows and all(m.anchor==d.PRIOR and m.target==d.PRIOR for m in state.models),'Nonneutral state lacks a profile')
    require(s['source'] in ('live_n_scalar','original_n_scalar_fallback'),'Unknown base source')
    live=s['live_queue_extra']
    close(s['x'],s['scalar_gain']*(live if live is not None else s['original_queue_extra']),'original X')
    require((live is not None)==(s['source']=='live_n_scalar'),'Incorrect fallback source')
    require(s['fresh']==(max(0.,s['at']-s['poll_started'])<=max_gap),'Freshness not clock-derived')
    eligible=s['fresh'] and s['native_ready'] and s['native_waiting']>0 and math.isfinite(s['baseline']) and math.isfinite(s['x']) and s['x']>=.001
    require(s['eligible']==eligible,'Eligibility mismatch')
    c=s['context']
    # Native progress provenance is independently logged by the monitor. This
    # arithmetic audit treats the snapshot rate/capacity/backlog as recorded inputs.
    actual=state.context(s['at'],c['progress_tokens_per_second'],c['progress_capacity'],c['values'][2] is not None)
    close(actual,c,'causal pending context')
    if selected:
        for a,b in [('at','diagnostic_time_seconds'),('baseline','calibration_baseline_seconds'),
                    ('original_queue_extra','queue_excess_seconds'),('scalar_gain','calibration_scale'),
                    ('native_ready','native_ready'),('native_waiting','native_waiting')]:close(s[a],selected[b],'selected '+a)


def verify_native(snapshot, event, evidence, max_gap):
    ranks=evidence['ranks'] or []
    finite=lambda x: type(x) in (int,float) and math.isfinite(x)
    complete=bool(ranks) and evidence['expected_rank_count']==len(ranks) and len({r['dp_rank'] for r in ranks})==len(ranks)
    valid=complete and all(r['native_request_counts_available'] is True and r['native_request_counts_profile'] is not None
        and r['timely_prefill_progress_source'] in ('local_compute_tokens','prefill_started_requests')
        and finite(r['timely_prefill_progress_interval']) and 0<r['timely_prefill_progress_interval']<=max_gap
        and finite(r['timely_prefill_progress']) and r['timely_prefill_progress']>=0 for r in ranks)
    progress=sum(r['timely_prefill_progress'] for r in ranks) if valid and snapshot['fresh'] else None
    learned=sum(r['learned_prefill_capacity'] for r in ranks) if ranks and all(finite(r['learned_prefill_capacity']) and r['learned_prefill_capacity']>0 for r in ranks) else None
    ready=not any(r['prefill_capacity_learning_active'] is not None for r in ranks) or learned is not None
    close(snapshot['native_ready'],ready,'native capacity readiness')
    close(snapshot['poll_started'],event['poll_started_time_seconds'],'native poll source')
    close(snapshot['context']['progress_tokens_per_second'],progress,'qualified native rate')
    close(snapshot['context']['progress_capacity'],learned if ready else None,'capacity normalization')
    backlog=any(r['num_waiting_reqs']>0 or r['num_waiting_uncached_tokens_available'] is not False and r['num_waiting_uncached_tokens']>0 for r in ranks)
    c=snapshot['context'];deficit=max(0.,min(1.,1-progress/learned)) if c['complete'] and c['pending']>0 and backlog and progress is not None and learned else None
    close(c['values'][2],deficit,'native progress deficit')


def bind_native_profile(bindings, worker, profile, evidence):
    if profile is None:return
    require(type(profile) is int and profile>=0,'Invalid decision profile')
    require(isinstance(evidence['ranks'],list),'Profile lacks native rank evidence')
    signature=tuple(sorted(((r['dp_rank'],r['native_request_counts_profile'],r['timely_prefill_progress_source'])
                           for r in evidence['ranks']),key=lambda r:(r[0],json.dumps(r[1:]))))
    by_profile,by_signature=bindings.setdefault(worker,({},{}))
    require(profile not in by_profile or by_profile[profile]==signature,'Native signature changed without decision profile reset')
    require(signature not in by_signature or by_signature[signature]==profile,'Unchanged native signature changed opaque profile')
    by_profile[profile]=signature;by_signature[signature]=profile


def verify_recovery_input(snapshot, proof, max_gap, source_poll):
    require(proof is not None,'Missing causal live-N source')
    close(snapshot['poll_started'],source_poll,'Live-N native sample source')
    original=proof['original_queue_excess_seconds'];debt=proof['dispatch_debt_seconds']
    require(type(original) in (int,float) and type(debt) in (int,float) and original>=0 and 0<=debt<=original+1e-7,'Invalid queue/debt evidence')
    queue=max(original-debt,0.)
    require(snapshot['original_queue_extra']>=queue-1e-7,'Selected original queue is below same-poll base queue')
    selected_debt=max(snapshot['original_queue_extra']-queue,0.)
    live=None
    if proof['available']:
        rate=proof['net_population_depletion_per_second'];population=proof['native_population'];latest=proof['latest_poll_started_time_seconds']
        require(type(rate) in (int,float) and math.isfinite(rate) and rate>0 and type(population) in (int,float) and population>=0,'Missing recovery depletion evidence')
        require(proof['window_seconds']>=30 and proof['window_poll_count']>=5,'Insufficient recovery support')
        close(proof['queue_excess_seconds'],min(queue,population/rate)+debt,'Independent recovery arithmetic')
        if max(0.,snapshot['at']-latest)<=max_gap:live=min(queue,population/rate)+selected_debt
    else:
        close(proof['queue_excess_seconds'],original,'Recovery fallback queue')
        require(not (proof.get('reason')=='stale' and proof.get('latest_poll_started_time_seconds') is not None
                     and max(0.,snapshot['at']-proof['latest_poll_started_time_seconds'])<=max_gap),
                'Later stale diagnostic cannot prove earlier recovery availability')
    close(snapshot['live_queue_extra'],live,'Live-N source binding')


def replay(events):
    states={};epochs={};selected={};gaps={};native={};profiles={};recovery={};pending_recovery={};counts=Counter();rows=[];labels=[];pending_labels={};admissions={}
    def state(e):
        worker=e['worker'];epoch=e['calibration_epoch']
        if worker not in states:states[worker]=d.State();epochs[worker]=epoch
        require(epochs[worker]==epoch,'Decision epoch changed without reset')
        return states[worker]
    for index,e in enumerate(events):
        kind=e.get('event');worker=e.get('worker')
        if kind=='calibration_reset':
            states[worker]=d.State();epochs[worker]=e['calibration_epoch'];counts['resets']+=1
            for key in [key for key in selected if key[0]==worker]:selected.pop(key)
            profiles.pop(worker,None);recovery.pop(worker,None)
            continue
        if kind=='calibration_forecast':
            s=state(e);selected[(worker,e['request_id'])]=(deepcopy(s),e,deepcopy(native.get(worker)),deepcopy(recovery.get(worker)))
        if kind=='feature_admission_decision' and e.get('decision')=='admitted' and e.get('outcome_tracking_supported') is False:
            state(e).mark_incomplete()
        if kind in ('recovery_forecast','recovery_poll'):
            transaction='decision_forecast' if kind=='recovery_forecast' else 'decision_poll'
            key=(worker,transaction,e.get('request_id',''))
            item=pending_recovery.pop(key,None)
            if item:
                snapshot,max_gap,source_poll=item
                if transaction=='decision_forecast':close(snapshot['at'],e['diagnostic_time_seconds'],'Same-clock recovery forecast')
                require(e['calibration_epoch']==epochs[worker],'Recovery epoch mismatch')
                verify_recovery_input(snapshot,e,max_gap,source_poll)
            if worker in native:
                recovery[worker]=(e,native[worker][0]['poll_started_time_seconds'])
        if kind not in EVENTS:continue
        p=payload(e);s=state(e);now=e['diagnostic_time_seconds'];counts[kind]+=1
        if kind=='decision_native_poll':native[worker]=(e,p)
        if kind=='decision_poll':
            require(worker in native,'Missing exact native evidence')
            native_event,native_payload=native[worker]
            close(native_event['diagnostic_time_seconds'],now,'native publication clock')
            require(native_event['calibration_epoch']==e['calibration_epoch'],'Native epoch mismatch')
            bind_native_profile(profiles,worker,p['profile'],native_payload)
            if p['snapshot'] is not None:verify_native(p['snapshot'],native_event,native_payload,p['max_gap'])
            gaps[worker]=p['max_gap'];s.poll(now,p['profile'],p['fresh'],p['max_gap'])
            close(s.diagnostics(),p['state'],'poll state')
        if kind in ('decision_poll','decision_forecast') and p['snapshot'] is not None:
            verify_snapshot(p['snapshot'],s,p['max_gap'])
            bind_native_profile(profiles,worker,p['snapshot']['profile'],native[worker][1])
            pending_recovery[(worker,kind,e.get('request_id',''))]=(p['snapshot'],p['max_gap'],native[worker][0]['poll_started_time_seconds'])
            if kind=='decision_forecast':
                require(worker in native,'Missing native provenance')
                verify_native(p['snapshot'],*native[worker],p['max_gap'])
            close(s.predictions(p['snapshot'],now,p['available']),p['predictions'],'published predictions')
        if kind=='decision_admission':
            key=(worker,e['request_id']);require(key not in admissions,'Duplicate admission')
            require(key in selected or p['snapshot'] is None,'No selected calibration snapshot')
            if p['snapshot'] is not None:
                saved,source,source_native,source_recovery=selected.pop(key)
                verify_snapshot(p['snapshot'],saved,gaps[worker],source)
                require(source_native is not None,'Missing selected native provenance')
                verify_native(p['snapshot'],*source_native,gaps[worker])
                bind_native_profile(profiles,worker,p['snapshot']['profile'],source_native[1])
                if source_recovery is None:
                    require(saved.profile is None and p['snapshot']['live_queue_extra'] is None,'Missing selected recovery source outside neutral reset')
                else:
                    proof,source_poll=source_recovery
                    require(proof['calibration_epoch']==e['calibration_epoch'],'Selected recovery epoch mismatch')
                    verify_recovery_input(p['snapshot'],proof,gaps[worker],source_poll)
                available=p['snapshot']['fresh'] and p['snapshot']['native_ready']
                close(saved.predictions(p['snapshot'],p['snapshot']['at'],available),p['predictions'],'frozen admission prediction')
            selected.pop(key,None)
            require(s.register(p['id'],p['started'],p['snapshot'])==p['tracked'],'Tracking mismatch')
            row=dict(event_index=index,worker=worker,epoch=e['calibration_epoch'],request_id=e['request_id'],**p)
            rows.append(row);admissions[key]=row
        if kind=='decision_label':
            label=p['label'];identity=label['id'];reason=e['reason'];terminal=reason!='poll_deadline'
            require(reason in ('poll_deadline','first_text','upstream_timeout','pretext_downstream_disconnect'),'Unclean label reason')
            require(label['exact']==(reason=='first_text'),'Censor/exact category mismatch')
            if not terminal:
                require(identity in s.pending and label['observed_at']-s.pending[identity]['started']>=300-1e-7,'Premature deadline bound')
            value=s.label(identity,label['observed_at'],now,label['exact'],terminal,300. if not terminal else None)
            require(value is not None,'Logged label was ineligible');close(value,label,'original saved label')
            labels.append(dict(event_index=index,worker=worker,reason=reason,label=label))
            if terminal:pending_labels[(worker,identity)]=(now,label['observed_at'],reason)
        if kind=='decision_terminal':
            key=(worker,p['id']);published=pending_labels.pop(key,None)
            if published:close(list(published),[now,p['observed_at'],p['outcome']],'terminal label provenance')
            elif p['id'] in s.pending:
                if p['outcome'] in ('first_text','upstream_timeout','pretext_downstream_disconnect'):
                    require(s.label(p['id'],p['observed_at'],now,p['outcome']=='first_text',True) is None,'Missing eligible terminal label')
                else:s.terminal_unknown(p['id'])
    require(not pending_recovery,'Decision snapshot lacks paired recovery transaction')
    require(not pending_labels,'Labels without terminal provenance')
    identity={(r['worker'],r['epoch'],r['id']):r for r in rows if r['id'] is not None}
    first={e['request_id']:e for e in events if e.get('event')=='first_text'}
    outcomes={(e['worker'],e['request_id']):e for e in events if e.get('event')=='feature_outcome'}
    resolutions={(e['worker'],e['request_id']):e for e in events if e.get('event')=='feature_deadline_resolution'}
    for item in labels:
        label=item['label'];event=events[item['event_index']]
        admission=identity[(item['worker'],event['calibration_epoch'],label['id'])]
        key=(item['worker'],admission['request_id'])
        if item['reason']=='poll_deadline':
            proof=resolutions.get(key);require(proof is not None and proof['resolution_source']=='poll','Bound lacks deadline ledger proof')
            close(proof['diagnostic_time_seconds'],label['at'],'bound publication')
        else:
            proof=outcomes.get(key);require(proof is not None and proof['outcome']==item['reason'],'Label lacks actual terminal proof')
            close(proof['diagnostic_time_seconds'],label['at'],'terminal publication')
        if label['exact']:
            require(admission['request_id'] in first,'Exact label lacks first-text proof')
            close(first[admission['request_id']]['ttft_seconds'],label['elapsed'],'first text elapsed')
    require(counts['decision_poll'] and counts['decision_admission'],'Incomplete decision capture')
    return dict(ok=True,method=d.METHOD,counts=dict(counts),admissions=rows,labels=labels)


def score_admissions(admissions, paired, capture_name, model):
    try:
        from .wait_decision_selection import compare,score
    except ImportError:
        from wait_decision_selection import compare,score
    owned={r['opc_request_id']:r for r in paired};rows=[]
    require(len(owned)==len(paired),'Duplicate paired request ID')
    expected={rid for rid,p in owned.items() if not p.get('local_drop') and not p.get('controlled_rejection')}
    recorded=[r['request_id'] for r in admissions if r['request_id'] in owned]
    require(len(recorded)==len(set(recorded)),'Duplicate authoritative admission')
    require(set(recorded)==expected,'Missing or unexpected authoritative owned admission')
    for admission in admissions:
        p=owned.get(admission['request_id'])
        if p is None:continue
        snapshot=admission['snapshot'];candidates=admission['predictions'] or []
        available=bool(snapshot and candidates and candidates[0]['available'])
        threshold=p.get('feature_context_candidate',{}).get('snapshot',{}).get('threshold_seconds')
        require(type(threshold) in (int,float) and math.isfinite(threshold),'Missing recorded EXTRA threshold')
        fallback=p.get('pre_admission_recorded_extra_flags',{}).get('calibrated')
        require(type(fallback) in (bool,type(None)),'Invalid scalar fallback')
        predictions={'live_n_scalar':snapshot['x']>=threshold if available else fallback}
        predictions.update({v['arm']:v['extra']>=threshold if v['available'] else fallback for v in candidates})
        fallback_total=p.get('pre_admission_total_ttft_predictions',{}).get('calibrated')
        totals={'live_n_scalar':snapshot['baseline']+snapshot['x'] if available else fallback_total}
        totals.update({v['arm']:v['total'] if v['available'] else fallback_total for v in candidates})
        for arm in d.ARMS:totals.setdefault(arm,fallback_total)
        for arm in d.ARMS:predictions.setdefault(arm,fallback)
        truth=None if p.get('controlled_rejection') or p.get('local_drop') else p.get('first_text_late')
        require(type(truth) in (bool,type(None)),'Invalid truth')
        rows.append(dict(**{k:v for k,v in admission.items() if k!='predictions'},candidates=candidates,predictions=predictions,
            total_predictions=totals,
            capture=capture_name,model=model,phase=p['phase'],truth=truth,actual_ttft=p.get('smg_ttft_seconds'),
            arrival=p['started_elapsed_seconds'],actual_rejected=p.get('controlled_rejection',False),threshold=threshold,
            comparison='recorded EXTRA policy; TOTAL>=300 is separate',scalar_fallback_used=not available))
    for rid,p in owned.items():
        if rid in expected:continue
        rows.append(dict(request_id=rid,capture=capture_name,model=model,phase=p['phase'],truth=None,
            actual_rejected=p.get('controlled_rejection',False),local_drop=p.get('local_drop',False),
            predictions={arm:None for arm in ['live_n_scalar',*d.ARMS]},
            total_predictions={arm:None for arm in ['live_n_scalar',*d.ARMS]},candidates=[],snapshot=None))
    def evaluation(values):
        return dict(scores={arm:{phase:score([r for r in values if phase=='all' or r['phase']==phase],arm)[0]
            for phase in ['all']+sorted({r['phase'] for r in values})} for arm in ['live_n_scalar',*d.ARMS]},
            selection={arm:compare(values,'live_n_scalar',arm) for arm in d.ARMS})
    total_rows=[dict(r,predictions={arm:value>=300 if value is not None else None for arm,value in r['total_predictions'].items()}) for r in rows]
    extra=evaluation(rows);total=evaluation(total_rows)
    return rows,dict(extra=extra,total_ttft_300=total,
        promotion_requires_both_contracts=True,expected_admitted=len(expected),actual_rejected=sum(p.get('controlled_rejection',False) for p in paired),
        owned_rows=len(rows),selection={arm:extra['selection'][arm]['eligible'] and total['selection'][arm]['eligible'] for arm in d.ARMS})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--router',type=Path,required=True);parser.add_argument('--state',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(not args.out.exists(),'Output directory already exists')
    state_digest=sha(args.state);capture=json.loads(args.state.read_text());require(capture.get('results_valid') is True,'Capture invalid')
    root=Path(capture['result_path']).resolve();require(args.router.resolve()==root/'router-events.jsonl','Noncanonical router source')
    require({'router-events.jsonl','paired-production.jsonl'}<=capture['result_files'].keys(),'Consumed canonical sources missing from hash manifest')
    bindings={}
    for name,digest in capture['result_files'].items():
        path=(root/name).resolve();require(path.is_relative_to(root),'Result path escaped root');require(sha(path)==digest,'Source hash mismatch: '+name);bindings[name]=digest
    events=read_events(args.router);schema=audit_schema(events);require(schema['ok'],str(schema))
    result=replay(events);require(result==replay(events),'Replay nondeterministic')
    paired=[json.loads(line) for line in (root/'paired-production.jsonl').read_text().splitlines()]
    try:
        from . import wait_feature_context,wait_feature_deadline
    except ImportError:
        import wait_feature_context,wait_feature_deadline
    audits=dict(activity=wait_feature_context.audit_activity(events,paired),deadlines=wait_feature_deadline.audit_deadlines(events,paired))
    require(all(v['ok'] for v in audits.values()),'Independent terminal/deadline audit failed')
    result['admissions'],evaluation=score_admissions(result['admissions'],paired,root.name,root.parent.name)
    result.update(evaluation=evaluation,audits=audits)
    require(sha(args.state)==state_digest and all(sha(root/name)==digest for name,digest in bindings.items()),'Source changed during audit')
    args.out.mkdir(parents=True)
    for key in ('admissions','labels'):
        (args.out/(key+'.jsonl')).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in result.pop(key)))
    result.update(schema=schema,deterministic_repeat_equal=True,source_state=str(args.state),source_state_sha256=state_digest,source_files=bindings,
                  source_router=str(args.router),helper_sha256=sha(__file__),math_sha256=sha(d.__file__),
                  limitations=['Native per-rank rate provenance, completeness, interval and normalization are audited; monitor counter derivatives are separately unit-tested.','Opaque native profile hashes are checked for signature consistency, not reimplemented with Python hashing.',
                                'Recovery rate/window eligibility is a separately recorded prerequisite; selected live-N arithmetic, same-poll debt and clock freshness are independently bound.',
                                'Capture closure and restoration are host-owner gates in addition to hash/results_valid checks.'])
    (args.out/'decision-replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['counts']))

if __name__=='__main__':main()
