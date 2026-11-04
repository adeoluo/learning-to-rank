# Records docs/demo/train.gif from a real run: `vhs docs/demo/train.tape` from the repo root.
# Expects MSLR-WEB10K prepared by `make data` under ../.bench-data/learning-to-rank/mslr10k.
Output docs/demo/train.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 500
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && mkdir -p .demo && clear` Enter
Sleep 10s
Show
Type `python -m learning_to_rank info` Enter
Sleep 2s
Type `python -m learning_to_rank train --data ../.bench-data/learning-to-rank/mslr10k --max-queries 2000 --trees 100 --log-every 25 --model .demo/lambdamart.json` Enter
Wait+Screen@240s /err@10/
Sleep 4s
