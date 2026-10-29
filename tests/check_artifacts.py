"""Check evidence consistency without downloading data or retraining."""
from pathlib import Path
import json, csv, math
root=Path(__file__).resolve().parents[1]
r=json.loads((root/'dist/assets/results.json').read_text())
a=json.loads((root/'dist/assets/audit.json').read_text())
m=json.loads((root/'dist/assets/models.json').read_text())
f=json.loads((root/'artifacts/frozen_selection.json').read_text())
assert sum(v['n'] for v in a['splits'].values())==a['rows']
assert r['winner']==max(r['models'][:4],key=lambda v:v['validation']['ap'])['name']==f['winner']
for result,export in zip(r['models'],m):
    assert result['name']==export['name']
    t=result['test']; (tn,fp),(fn,tp)=t['confusion']
    assert tn+fp+fn+tp==r['test_n']
    assert tp+fn==r['test_positive']
    assert math.isclose(t['recall'],tp/(tp+fn))
    assert math.isclose(t['precision'],tp/(tp+fp))
    assert math.isclose(t['f1'],2*tp/(2*tp+fp+fn))
    assert t['threshold']==export['threshold']==f['thresholds'][result['name']]
    assert result['capacity'][-1]['captured']==r['test_positive']
    assert sum(s['n'] for s in result['subgroups'] if s['group']=='Sex')==r['test_n']
    assert sum(s['n'] for s in result['subgroups'] if s['group']=='Age')==r['test_n']
rows=list(csv.DictReader((root/'artifacts/split_assignments.csv').open()))
assert len(rows)==a['rows'] and len({v['ID'] for v in rows})==a['rows']
assert r['dummy']['recall']==0
print('PASS: split counts, frozen selection, thresholds, confusion metrics, capacity totals, subgroup coverage, baseline.')
