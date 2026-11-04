# Records docs/demo/serve.gif from a real run: `vhs docs/demo/serve.tape` from the repo root.
# Expects .demo/lambdamart.json written by docs/demo/train.tape.
Output docs/demo/serve.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 480
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 10s
Type `python -m learning_to_rank serve --model .demo/lambdamart.json --port 18390 >/dev/null 2>&1 &` Enter
Type `python scripts/demo_query.py ../.bench-data/learning-to-rank/mslr10k .demo/query.json` Enter
Sleep 10s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18390/health; echo` Enter
Sleep 1.5s
Type `curl -s localhost:18390/rank -d @.demo/query.json | jq -c` Enter
Sleep 3s
Type `curl -s localhost:18390/metrics | grep -v '^#' | head -n 8` Enter
Sleep 4s
Hide
Type `kill %1` Enter
Sleep 1s
