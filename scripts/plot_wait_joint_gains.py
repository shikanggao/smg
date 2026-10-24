#!/usr/bin/env python3
"""Linear TTFT, signed contributions and independent weights from replay rows."""
import argparse
import json
import math
from pathlib import Path


def series(rows,key):
    """Keep unknown observations in the line as gaps, never bridge over them."""
    return [r[key] if type(r.get(key)) in (int,float) and math.isfinite(r[key]) else math.nan for r in rows]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--admissions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--title',required=True)
    parser.add_argument('--mode',choices=('historical','fresh'),required=True)
    parser.add_argument('--censor-bound',type=float,choices=(300.,420.))
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted((json.loads(line) for line in args.admissions.read_text().splitlines()),key=lambda r:r['arrival_seconds'])
    x=[r['arrival_seconds'] for r in rows]
    fig,axes=plt.subplots(4,1,figsize=(14,14),sharex=True,gridspec_kw={'height_ratios':[3,1,1.5,1.5]})
    for key,label,color in [('scalar','Scalar','#777777'),('feature_v2','Historical feature v2','#aa77aa'),
                             ('live_same_scalar','Live remaining work + same scalar','#167ba6'),('joint_v3','Joint feature gains v3','#d55e00')]:
        if any(key in r for r in rows):axes[0].plot(x,series(rows,key),label=label,color=color,lw=1.25)
    exact=[r for r in rows if r['actual_ttft_seconds'] is not None and r['truth'] is not None]
    axes[0].scatter([r['arrival_seconds'] for r in exact],[r['actual_ttft_seconds'] for r in exact],s=5,c='black',label='Actual first text (known later)',zorder=5)
    censored=[r for r in rows if r['actual_ttft_seconds'] is None and r['truth'] is True and r.get('censor_bound_seconds') is not None]
    if args.censor_bound is not None and any(r['censor_bound_seconds']!=args.censor_bound for r in censored):
        raise ValueError('Per-request lower bound differs from requested arm label')
    axes[0].scatter([r['arrival_seconds'] for r in censored],[r['censor_bound_seconds'] for r in censored],s=12,c='black',marker='^',label='No first text: observed lower bound',zorder=5)
    axes[0].axhline(300,color='#b2182b',ls='--',lw=1,label='300s decision threshold')
    axes[0].set_ylabel('Total TTFT (seconds)');axes[0].legend(fontsize=8,ncol=2)
    axes[1].plot(x,[r['prediction']['contribution_seconds'][0] for r in rows],color='#555555',label='Queue contribution N × effective gQ')
    axes[1].set_ylabel('Queue delay (s)');axes[1].legend(fontsize=8)
    colors=['#555555','#009e73','#cc79a7','#e69f00']
    labels=['Queue','KV','Deadline/cutoff','Rejection']
    for j in (1,2,3):
        axes[2].plot(x,[r['prediction']['contribution_seconds'][j] for r in rows],label=labels[j],color=colors[j],lw=1.3)
    for j in range(4):
        axes[3].plot(x,[r['prediction']['applied_coefficients'][j] for r in rows],label='g '+labels[j],color=colors[j],lw=1.3)
        axes[3].plot(x,[r['prediction']['target_coefficients'][j] for r in rows],color=colors[j],lw=.7,ls='--',alpha=.45)
    axes[2].axhline(0,color='black',lw=.6);axes[2].legend(fontsize=8,ncol=3)
    axes[2].set_ylabel('Signed added delay (s)')
    axes[3].set_ylabel('Applied coefficients\n(dashed: target)');axes[3].legend(fontsize=8,ncol=4)
    axes[3].set_xlabel('Request arrival time (seconds)')
    for ax in axes:
        ax.grid(alpha=.2)
        for edge in (60,480,660):ax.axvline(edge,color='#bbbbbb',ls=':',lw=.8)
    mode='historical workload, shadow counterfactual' if args.mode=='historical' else 'fresh capture, recorded shadow forecasts'
    axes[0].set_title(args.title+' — '+mode)
    missing=sum(r['truth'] is True and r['actual_ttft_seconds'] is None and r.get('censor_bound_seconds') is None for r in rows)
    fig.text(.07,.01,'Actual labels arrive later. Censored bounds are not exact TTFT. Linear axes; missing predictions break lines.'+(f' {missing} late rows lack a verified numeric lower bound and are not plotted.' if missing else ''),fontsize=9)
    fig.tight_layout(rect=(0,.025,1,1));args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=160);plt.close(fig)


if __name__=='__main__':main()
