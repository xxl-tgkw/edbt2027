# EDBT 2027 template and submission settings

Official CFP verified 2026-09-16:
https://edbticdt2027.github.io/contents/EDBT_CFP.html

It links the A4 template at:

- https://drive.google.com/file/d/1dtNEVPLPNTlKm1_agOxlHWuQK-Szd8Lo/view?usp=drive_link
- https://de.overleaf.com/read/khczkkygdrtt#aeaa28

The user-supplied original skeleton is edbt-paper-template.tex. Current manuscript
edbt.tex uses the supplied acmart.cls, edbt-macros.tex, ACM bibliography style and
A4 geometry with sigconf. These dependencies are present and compile locally.
Fonts, margins, inter-column spacing and supplied template macros were not changed.
The old compatible scaffold paper/edbt2027_a4.tex is not the current manuscript.

## Required settings from the official CFP

- The manuscript title begins with the required EA&B prefix `[EA&B]`.
- At most 12 pages for content including appendices; references may occupy more pages.
- Artifacts section immediately before references; that section is exempt from the limit.
- Single-anonymous review: include true author names and affiliations.
- All authors need ORCID registration in CMT; authorship and conflicts must be correct.
- Submit one PDF and supplemental artifact ZIP or an appropriate artifact link.

The current metadata is still a placeholder, and the supplied publication macros
contain pending fields. Those must be finalized for actual submission. The template
deliberately disables the ACM reference block, so acmart's associated warning alone
does not justify changing the EDBT macros.
The supplied class also balances the last page. In the tested TeX Live version this
can report a small vertical-box warning on the reference page; the package validation
records its exact size. It is not suppressed by altering the template geometry.

Third cycle: submission 7 October 2026; first decision 5 December 2026; revised
submission 4 January 2027; final decision 27 January 2027. CFP states 5pm PST.
