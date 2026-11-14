"""Quick look at a capsule log's replays: how many leakers each arm caught, how many it missed at 0.9 or more, and how
many clean pictures it flagged (Nov 5 morning, exploratory; the real scoring is benchmark.py score).

    python3 scripts/capsule_catch_rates.py output/capsules/sel01 output/capsules/sel01/blocks/n1
"""
import json, sys, collections
POS={'yes','leaker'}
for name in sys.argv[1:]:
    truths=[json.loads(l).get('truth') for l in open(name+'/decisions.jsonl')]  # the log's labels as they stand now, not as replayed
    rows=[json.loads(l) for l in open(name+'/arms.jsonl')]
    for r in rows:
        r['truth']=truths[r['index_global']]
    by=collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r.get('error') or r.get('truth') is None: continue
        q=r['question']
        if q not in ('leak_block','leak_capsule','tray_clear'): continue
        p=r['probs']; top=max(p,key=p.get); conf=p[top]
        pos = r['truth'] in POS if q!='tray_clear' else r['truth']=='no'
        k=by[r['arm']]
        if pos:
            k['pos']+=1; k['pos_hit']+= top==r['truth']; k['pos_missed_conf']+= (top!=r['truth'] and conf>=0.9)
        else:
            k['neg']+=1; k['false_alarm']+= top!=r['truth']
    print('==',name)
    for arm,k in sorted(by.items()):
        print(f"  {arm:22s} leakers shown {k['pos']:4d} caught {k['pos_hit']/max(k['pos'],1):.2f}  missed at >=0.9 {k['pos_missed_conf']:3d}  | clean {k['neg']:5d} false alarms {k['false_alarm']/max(k['neg'],1):.3f}")
