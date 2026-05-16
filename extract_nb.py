import json, glob

out = ''
for f in glob.glob('*.ipynb'):
    try:
        nb = json.load(open(f, encoding='utf-8'))
        out += f'\n\n--- {f} ---\n'
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                for o in cell.get('outputs', []):
                    if 'text' in o:
                        out += ''.join(o['text'])
                    elif 'data' in o and 'text/plain' in o['data']:
                        out += ''.join(o['data']['text/plain']) + '\n'
    except Exception as e:
        out += f'Error reading {f}: {e}\n'
open('notebook_outputs.txt', 'w', encoding='utf-8').write(out)
