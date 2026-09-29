import json
from pathlib import Path

def main():
    rows=json.load(open('results/summary.json')); methods=['exact','lsh','ivf_static','ivf_rebuild','faiss_flat','faiss_ivf_rebuild']; colors=['#1f77b4','#d62728','#2ca02c','#9467bd','#17becf','#8c564b']
    pts=[(r['method'],r['update_ms'],r['recall_mean']) for r in rows if r['dataset']=='digits']
    W,H=900,520; out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Digits workload: recall--update-cost trade-off</text>']
    xmin,xmax=0,60; ymin,ymax=.70,1.01
    def x(v): return 90+(v-xmin)/(xmax-xmin)*760
    def y(v): return 430-(v-ymin)/(ymax-ymin)*350
    for i in range(6):
        v=xmin+(xmax-xmin)*i/5; xx=x(v); out += [f'<line x1="{xx}" x2="{xx}" y1="80" y2="430" stroke="#eee"/><text x="{xx}" y="455" text-anchor="middle" font-size="12">{v:.0f}</text>']
    for i in range(5):
        v=ymin+(ymax-ymin)*i/4; yy=y(v); out += [f'<line x1="90" x2="850" y1="{yy}" y2="{yy}" stroke="#eee"/><text x="80" y="{yy+4}" text-anchor="end" font-size="12">{v:.2f}</text>']
    out += ['<line x1="90" x2="850" y1="430" y2="430" stroke="#222"/><line x1="90" x2="90" y1="80" y2="430" stroke="#222"/>','<text x="470" y="490" text-anchor="middle" font-size="14">mean update latency (ms)</text>','<text x="18" y="255" transform="rotate(-90 18 255)" text-anchor="middle" font-size="14">Recall@10</text>']
    for i,m in enumerate(methods):
        for _,u,r in [p for p in pts if p[0]==m]: out.append(f'<circle cx="{x(u):.1f}" cy="{y(r):.1f}" r="5" fill="{colors[i]}" opacity=".75"/>')
        out += [f'<rect x="{610+i*65}" y="50" width="10" height="10" fill="{colors[i]}"/><text x="{625+i*65}" y="60" font-size="11">{m}</text>']
    out.append('</svg>'); Path('results/recall_update_tradeoff.svg').write_text(''.join(out))
if __name__=='__main__': main()
