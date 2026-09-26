# Regenerates assets/fact-checker.gif — the bibtex-check three-way-verdict demo.
#
#   Run from the repo root:   vhs assets/fact-checker.tape
#   (then: mv fact-checker.gif assets/  — vhs writes Output relative to the cwd)
#
# A LIVE `bibtex-check assets/fact-checker.demo.bib` run takes ~50s (it queries
# Crossref/OpenAlex/DBLP/OpenReview/Semantic Scholar live), which is too long for
# a watchable GIF, so this tape replays the tool's REAL captured output
# (assets/fact-checker.out, produced by that exact command) without the network
# wait. The demo bibliography mixes a correct reference, a corrupted one (right
# title, wrong authors), and a fabricated one, to show all three verdict buckets:
# VERIFIED / COULD NOT VERIFY / PROBLEMATIC. Regenerate fact-checker.out with:
#   bibtex-check assets/fact-checker.demo.bib > assets/fact-checker.out 2>&1
Output fact-checker.gif

Set Shell "bash"
Set FontSize 24
Set Width 1800
Set Height 1012
Set Padding 40
Set Theme "Catppuccin Mocha"

Hide
Type "bibtex-check() { command sleep 1.5; command cat assets/fact-checker.out; }; export PS1='$ '; clear"
Enter
Show

Sleep 800ms
Type "bibtex-check demo.bib"
Sleep 700ms
Enter
Sleep 7s
