# Records docs/demo/compare.gif from a real run: `vhs docs/demo/compare.tape` from the repo root.
# Expects MSLR-WEB10K prepared by `make data` under ../.bench-data/learning-to-rank/mslr10k.
Output docs/demo/compare.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && mkdir -p .demo && export D=../.bench-data/learning-to-rank/mslr10k` Enter
Sleep 10s
Type `python -m learning_to_rank train --data $D --max-queries 2000 --trees 100 --model .demo/lambdarank.json >/dev/null && python -m learning_to_rank train --data $D --max-queries 2000 --trees 100 --objective regression --model .demo/regression.json >/dev/null && clear` Enter
Wait+Line@400s /\$$/
Show
Type `python -m learning_to_rank evaluate --data $D --model .demo/lambdarank.json .demo/regression.json` Enter
Wait+Line@120s /\$$/
Sleep 2s
Type `python -m learning_to_rank compare --data $D .demo/lambdarank.json .demo/regression.json` Enter
Wait+Line@120s /\$$/
Sleep 4s
