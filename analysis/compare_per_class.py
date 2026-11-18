import json, sys

old = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))

old_pc = {c['classe']: c for c in old['classes']}
new_pc = {c['classe']: c for c in new['classes']}

common = sorted(set(old_pc) & set(new_pc))
missing_old = set(new_pc) - set(old_pc)
missing_new = set(old_pc) - set(new_pc)
if missing_old:
    print(f"[Aviso] {len(missing_old)} classes só no novo (sem par no baseline)")
if missing_new:
    print(f"[Aviso] {len(missing_new)} classes só no baseline (sem par no novo)")

rows = []
for c in common:
    o, n = old_pc[c], new_pc[c]
    da = n['accuracy'] - o['accuracy']
    di = n['iou'] - o['iou']
    rows.append((c, o['accuracy'], n['accuracy'], da, o['iou'], n['iou'], di))

rows.sort(key=lambda r: r[6])  # pior delta de IoU primeiro

print(f"\n{'classe':38s} {'acc_old':>8s} {'acc_new':>8s} {'d_acc':>8s}  {'iou_old':>8s} {'iou_new':>8s} {'d_iou':>8s}")
print("-"*100)
n_worse_iou = n_worse_acc = 0
for c, ao, an, da, io, iN, di in rows:
    flag = ""
    if di < -0.02:
        flag += "  <-- IoU PIOROU"
        n_worse_iou += 1
    if da < -0.02:
        flag += "  <-- ACC PIOROU"
        n_worse_acc += 1
    print(f"{c[:37]:38s} {ao:8.4f} {an:8.4f} {da:+8.4f}  {io:8.4f} {iN:8.4f} {di:+8.4f}{flag}")

print(f"\nTotal classes comparadas: {len(rows)}")
print(f"Classes com IoU piorando >2pp: {n_worse_iou}")
print(f"Classes com accuracy piorando >2pp: {n_worse_acc}")
