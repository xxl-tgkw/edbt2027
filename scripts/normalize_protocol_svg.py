#!/usr/bin/env python3
"""Make the supplied method diagram portable across LaTeX renderers.

The Office-exported SVG uses DengXian-specific offsets and a clip rectangle
that is not robust when the font is unavailable.  The normalizer keeps the
author-supplied shapes/arrows, removes the fragile clip, and replaces text
with centered Liberation Sans blocks.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dist/1234.svg"
OUTPUT = ROOT / "results/uvrbench_protocol.svg"

TEXT_BLOCKS = [
    '<text x="155" y="247" text-anchor="middle" font-family="Liberation Sans" font-weight="700" font-size="24">Controlled roles<tspan x="155" y="271" font-weight="400" font-size="19">base / updates /</tspan><tspan x="155" y="294" font-weight="400" font-size="19">queries frozen IDs and</tspan><tspan x="155" y="316" font-weight="400" font-size="19">seeds</tspan></text>',
    '<text x="398" y="247" text-anchor="middle" font-family="Liberation Sans" font-weight="700" font-size="24">Arrival orders<tspan x="398" y="271" font-weight="400" font-size="19">shifted stream shuffled</tspan><tspan x="398" y="294" font-weight="400" font-size="19">control same final</tspan><tspan x="398" y="316" font-weight="400" font-size="19">corpus</tspan></text>',
    '<text x="1126" y="247" text-anchor="middle" font-family="Liberation Sans" font-weight="700" font-size="24">Audited output<tspan x="1126" y="271" font-weight="400" font-size="19">Recall@10, Q95 insert</tspan><tspan x="1126" y="294" font-weight="400" font-size="19">+ maintenance C /</tspan><tspan x="1126" y="316" font-weight="400" font-size="19">C+V and Pareto</tspan></text>',
    '<text x="641" y="247" text-anchor="middle" font-family="Liberation Sans" font-weight="700" font-size="24">Replay rounds<tspan x="641" y="271" font-weight="400" font-size="19">initial build insert +</tspan><tspan x="641" y="294" font-weight="400" font-size="19">checkpoint query</tspan><tspan x="641" y="316" font-weight="400" font-size="19">optional rebuild</tspan></text>',
    '<text x="883" y="247" text-anchor="middle" font-family="Liberation Sans" font-weight="700" font-size="24">Index policies<tspan x="883" y="271" font-weight="400" font-size="19">IVF-fixed / IVF-rebuilt</tspan><tspan x="883" y="294" font-weight="400" font-size="19">HNSW / dynamic</tspan><tspan x="883" y="316" font-weight="400" font-size="19">graph exact reference</tspan></text>',
    '<text x="641" y="377" text-anchor="middle" font-family="Liberation Sans" font-weight="400" font-size="24">paired evidence: exact neighbors + event log + hashes</text>',
    '<text x="641" y="456" text-anchor="middle" font-family="Liberation Sans" font-weight="700" font-size="24">Recall-floor selection<tspan x="641" y="485" font-weight="400" font-size="19">held-out validation -&gt; test quality / service-work boundary</tspan></text>',
]


def main():
    source = SOURCE.read_text(encoding="utf-8")
    # The source canvas is 1190x297 after translating the Office coordinates
    # (45,205).  A white background makes the figure deterministic in PDF
    # viewers that otherwise display transparent pixels as black.
    source = source.replace(
        '<svg width="1190" height="297"',
        '<svg width="1190" height="297" viewBox="0 0 1190 297"',
        1,
    )
    source = source.replace(
        '<defs><clipPath id="clip0"><rect x="45" y="205" width="1190" height="297"/></clipPath></defs><g clip-path="url(#clip0)" transform="translate(-45 -205)">',
        '<rect width="1190" height="297" fill="#FFFFFF"/><g transform="translate(-45 -205)">',
        1,
    )
    blocks = iter(TEXT_BLOCKS)
    normalized, count = re.subn(r"<text\b.*?</text>", lambda _: next(blocks), source, flags=re.S)
    if count != len(TEXT_BLOCKS):
        raise RuntimeError(f"expected {len(TEXT_BLOCKS)} text blocks, found {count}")
    OUTPUT.write_text(normalized, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
