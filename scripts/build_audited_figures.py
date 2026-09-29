#!/usr/bin/env python3
"""Paper figures derived directly from audited records; whiskers show seed ranges."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':8, 'axes.titlesize':9, 'legend.fontsize':7,
                     'pdf.fonttype':42, 'ps.fonttype':42})
LABELS = {'exact':'NumPy exact','lsh':'Hyperplane LSH','ivf_static':'NumPy IVF fixed',
          'ivf_rebuild':'NumPy IVF rebuilt','faiss_flat':'Faiss Flat',
          'faiss_ivf_static':'Faiss IVF fixed','faiss_ivf_rebuild':'Faiss IVF rebuilt'}
COLORS = {'faiss_flat':'#0072B2','faiss_ivf_static':'#009E73','faiss_ivf_rebuild':'#D55E00',
          'exact':'#999999','lsh':'#CC79A7','ivf_static':'#56B4E9','ivf_rebuild':'#E69F00'}


def save(fig, name):
    fig.savefig(ROOT/f'results/{name}.pdf', bbox_inches='tight', metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(ROOT/f'results/{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    rows=json.loads((ROOT/'results/validation_v3/raw.json').read_text())
    fig, axes=plt.subplots(1,2,figsize=(7.2,2.8))
    for m in sorted(LABELS):
        xs=[10,40,160]
        for ax,metric in zip(axes,['recall','service_ms']):
            values=np.array([[r[metric] for r in rows if r['method']==m and r['queries_per_batch']==q] for q in xs])
            y=values.mean(1)
            ax.errorbar(xs,y,yerr=[y-values.min(1),values.max(1)-y],capsize=2,
                        marker='o',markersize=3,label=LABELS[m],color=COLORS[m],linewidth=1)
            ax.set_xscale('log',base=4)
            ax.set_xticks(xs, labels=[str(q) for q in xs])
            ax.set_xlabel('Queries per 100-vector batch')
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Recall@10')
    axes[1].set_ylabel('Measured service work (ms)')
    axes[1].set_yscale('log')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.05),frameon=False)
    fig.tight_layout(rect=(0,.1,1,1))
    save(fig,'ratio_sensitivity')

    audit=json.loads((ROOT/'results/validation_audit.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.5))
    for col,c in enumerate(audit['conditions']):
        for p in c['policies']:
            r=np.asarray(p['round_recalls']); y=r.mean(0)
            axes[0,col].errorbar(range(1,6),y,yerr=[y-r.min(0),r.max(0)-y],marker='o',
                                markersize=3,capsize=2,color=COLORS[p['method']],label=LABELS[p['method']])
        axes[0,col].set_ylim(.94,1.005)
        axes[0,col].set_xticks(range(1,6))
        axes[0,col].set_xlabel('Post-update round')
        axes[0,col].set_title(f'Query factor {c["query_shift_factor"]:g}\nUpdated neighbors: {100*c["mean_exposure"]:.2f}%')
        axes[0,col].grid(alpha=.2)
        x=np.arange(3); bottom=np.zeros(3)
        for key,label,color in [('insert_ms','Insert','#56B4E9'),('rebuild_ms','Rebuild','#D55E00'),('query_total_ms','Query','#009E73')]:
            v=np.array([np.mean(p[key]) for p in c['policies']])
            axes[1,col].bar(x,v,bottom=bottom,label=label,color=color,width=.65)
            bottom+=v
        axes[1,col].set_xticks(x,labels=['Flat','Fixed IVF','Rebuilt IVF'],rotation=15)
        axes[1,col].set_ylim(0,650)
        axes[1,col].grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Recall@10')
    axes[1,0].set_ylabel('Service work (ms)')
    axes[0,0].legend(fontsize=6,loc='lower left',frameon=False)
    axes[1,0].legend(fontsize=6,loc='upper left',frameon=False)
    fig.tight_layout()
    save(fig,'alignment_diagnostics')


if __name__=='__main__':
    main()
