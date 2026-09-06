#!/usr/bin/env bash
# Export the manuscript as a single-column Word document.
# Requires pandoc >= 2.11 (for --citeproc). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"

# Convert PDF figures to PNG for Word; skip when PNG already exists.
for f in figures/*.pdf ../outputs/figures/*.pdf; do
  [ -f "$f" ] || continue
  png="${f%.pdf}.png"
  [ -f "$png" ] || pdftoppm -png -r 200 -singlefile "$f" "${f%.pdf}"
done

cat docx_shim.tex tables/macros.tex body.tex > .body_docx.tex
sed -i \
  -e 's/\\includegraphics\[[^]]*\]{\([^}]*\)}/\\includegraphics{\1.png}/' \
  -e 's/\\includegraphics{\([^}]*\)\.png\.png}/\\includegraphics{\1.png}/' \
  -e 's/\\resizebox{[^}]*}{!}{\(\\input{[^}]*}\)}/\1/' \
  -e 's/\\ifdefined\\balance.*\\fi\\fi//' \
  .body_docx.tex

pandoc .body_docx.tex \
  --from latex --to docx \
  --resource-path=.:figures:../outputs/figures \
  --citeproc --bibliography refs.bib --csl ieee.csl \
  --metadata title="Adaptive Intent-Preference Arbitration for Conversational Recommendation" \
  --metadata author="Dhayanidhi Rajasekaran (Research Scholar); Rajalakshmi N.R. (Prof. Dr.), Vel Tech Rangarajan Dr Sagunthala R&D Institute of Science and Technology, Chennai, India" \
  --number-sections \
  -o main_onecol.docx
rm -f .body_docx.tex
echo "wrote paper/main_onecol.docx"
