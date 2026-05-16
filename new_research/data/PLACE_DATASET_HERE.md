Put your dataset CSV in this folder.

Default expected path:
- `new_research/data/facebook-fact-check.csv`

If your CSV is elsewhere (or has a different name), run:
- `c:/Users/kriti/Downloads/QML-paper/.venv/Scripts/python.exe new_research/run_all.py --quick --data-csv "C:/path/to/your.csv"`

The code expects a column named `Rating` with values that include:
- `mostly true`
- `mostly false`
- `mixture of true and false`

and engagement/metadata feature columns (numeric + categorical). If your column names differ, we can adapt the loader.
