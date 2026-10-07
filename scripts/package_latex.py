#!/usr/bin/env python3
"""Package and independently compile the manuscript without editing its sources."""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'edbt.tex', 'acmart.cls', 'edbt-macros.tex', 'ACM-Reference-Format.bst',
    'sample-base.bib', 'uvr-refs.bib', 'edbt-paper-template.tex',
    'paper/body.tex', 'paper/fashion_table.tex', 'paper/ratio_table.tex',
    'paper/summary_table.tex', 'paper/sensitivity_table.tex',
    'paper/numbers.tex', 'paper/headline_table.tex', 'paper/digits_table.tex',
    'paper/alignment_table.tex', 'paper/event_cost_table.tex',
    'paper/paired_effects_table.tex', 'paper/protocol_table.tex',
    'paper/generated_manifest.json',
    'paper/semantic_section.tex', 'paper/semantic_table.tex',
    'paper/semantic_matched_table.tex', 'paper/semantic_numbers.tex',
    'paper/semantic_manifest.json',
    'paper/hotpot_semantic_table.tex',
    'paper/order_intensity_table.tex', 'paper/mechanism_table.tex',
    'paper/scale_table.tex',
    'paper/lifecycle_section.tex', 'paper/lifecycle_table.tex',
    'paper/claim_evidence_table.tex',
    'paper/large_scale_dynamic_table.tex',
    'paper/native_dynamic_table.tex',
    'paper/native_coverage_table.tex',
    'results/pareto.pdf', 'results/sensitivity.pdf', 'results/ratio_sensitivity.pdf',
    'results/alignment_diagnostics.pdf', 'results/semantic_rounds.pdf',
    'results/semantic_tradeoff.pdf', 'results/partition_mismatch.pdf',
    'results/uvrbench_protocol.pdf', 'results/uvrbench_protocol.svg',
    'results/uvrbench_protocol_source.svg', 'results/uvrbench_protocol_source.png',
    # The author-supplied final submission PDF is kept alongside the
    # reproducible source build for provenance; it is not part of the
    # manuscript-free artifact mirror.
    'edbt_submission.pdf',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_paper(directory):
    result = subprocess.run(['bash', 'build.sh'], cwd=directory, text=True,
                            errors='replace', stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout[-12000:])
    log = (directory / 'edbt.log').read_text(errors='replace')
    if re.search(r'undefined references|Citation .* undefined|Reference .* undefined', log):
        raise RuntimeError('Unresolved references in final LaTeX pass')
    info = subprocess.check_output(['pdfinfo', 'edbt.pdf'], cwd=directory, text=True)
    report = {
        'pages': int(re.search(r'Pages:\s+(\d+)', info).group(1)),
        'page_size': re.search(r'Page size:\s+(.+)', info).group(1),
        'existing_overfull_boxes': len(re.findall(r'Overfull \\[hv]box', log)),
        'overfull_horizontal_boxes': len(re.findall(r'Overfull \\hbox', log)),
        'overfull_vertical_points': [float(x) for x in re.findall(r'Overfull \\vbox \(([0-9.]+)pt', log)],
    }
    if report['overfull_horizontal_boxes']:
        raise RuntimeError(f'Horizontal content overflow remains: {report}')
    aux = (directory / 'edbt.aux').read_text()
    reference_page = int(re.search(r'\\newlabel\{sec:references\}\{\{[^}]*\}\{(\d+)\}', aux).group(1))
    # EDBT limits content (including Artifacts) to 12 pages; references are unlimited.
    # A bibliography beginning on page 13 is therefore valid.
    if report['pages'] < 11 or reference_page > 13 or 'A4' not in report['page_size']:
        raise RuntimeError(f'Expected 11–12 content pages on A4: {report}, references start page {reference_page}')
    float_pages = [int(p) for p in re.findall(r'\\newlabel\{(?:fig|tab|alg):[^}]+\}\{\{[^}]*\}\{(\d+)\}', aux)]
    if max(float_pages) > reference_page:
        raise RuntimeError('A content float appears after the references')
    report['references_start_page'] = reference_page
    return report


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    output = ROOT / 'dist'
    output.mkdir(exist_ok=True)
    archive = output / f'edbt2027_latex_{stamp}.zip'
    source_hashes = {name: digest(ROOT / name) for name in FILES}
    with tempfile.TemporaryDirectory(prefix='edbt-latex-') as temp:
        stage = Path(temp) / 'source'
        stage.mkdir()
        for name in FILES:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        shutil.copy2(ROOT / 'paper/LATEX_PACKAGE_README.md', stage / 'README.md')
        shutil.copy2(ROOT / 'scripts/build_paper.sh', stage / 'build.sh')
        first_build = compile_paper(stage)
        members = FILES + ['README.md', 'build.sh', 'edbt.pdf']
        sums = {name: digest(stage / name) for name in members}
        (stage / 'SHA256SUMS').write_text(''.join(
            f'{sums[name]}  {name}\n' for name in sorted(sums)), encoding='utf-8')
        with ZipFile(archive, 'x', ZIP_DEFLATED) as package:
            for name in members + ['SHA256SUMS']:
                package.write(stage / name, name)
        extracted = Path(temp) / 'extracted'
        with ZipFile(archive) as package:
            assert package.testzip() is None
            package.extractall(extracted)
        assert all(digest(extracted / name) == sums[name] for name in members)
        second_build = compile_paper(extracted)
        assert first_build == second_build
    assert all(digest(ROOT / name) == sha for name, sha in source_hashes.items())
    sha = digest(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{sha}  {archive.name}\n')
    report = {
        'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha,
        'files': members + ['SHA256SUMS'], 'validation': second_build,
        'compile_from_fresh_extraction': 'passed', 'source_files_unchanged': True,
        'created_utc': stamp,
    }
    archive.with_suffix('.validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
