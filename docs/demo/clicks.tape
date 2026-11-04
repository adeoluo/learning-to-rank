# Records docs/demo/clicks.gif from a real run: `vhs docs/demo/clicks.tape` from the repo root.
# Expects MSLR-WEB10K prepared by `make data` under ../.bench-data/learning-to-rank/mslr10k.
Output docs/demo/clicks.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m learning_to_rank clicks --data ../.bench-data/learning-to-rank/mslr10k --sessions 500000 --click-model pbm --eta 1` Enter
Wait+Line@240s /\$$/
Sleep 4s
