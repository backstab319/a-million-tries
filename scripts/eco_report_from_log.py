"""Rebuild `report.jsonl` records from the per-world lines `eco_jax_run.py` prints (ep17 10l: a stale upload erased
days 1950-5280 of runs/eco-jax-2's report; the log kept every world's line). Food is kept as shares (%), not amounts,
and records carry `"from_log": true`.

    uv run python scripts/eco_report_from_log.py out/ecosystem/jax-2.log 1950 5280 > runs/eco-jax-2/report-from-log.jsonl
"""
import json
import re
import sys

NUM = r"([-0-9.e+]+)"
PAT = re.compile(
    rf"day +{NUM} world +(\d+): alive +(\d+) births +(\d+) deaths (\d+)/(\d+)/(\d+)/(\d+) \(killed/thirst/starved/old\);"
    rf" food fruit {NUM}% carcass {NUM}% green {NUM}% dead {NUM}% milk {NUM}%; drink {NUM}% of steps at thirst {NUM},"
    rf" thirst {NUM}, {NUM} m/s, calls {NUM}%, attacks {NUM}%, green {NUM}x; diet {NUM}, meat-eaters (\d+),"
    rf" gen (\d+), mass {NUM} kg \[crowded (\d+) full (\d+) capped (\d+) books {NUM}\]")


def main():
    path, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    seen = set()
    for ln in open(path):
        m = PAT.search(ln)
        if not m:
            continue
        g = m.groups()
        day, seed = float(g[0]), int(g[1])
        if not lo <= day <= hi or (day, seed) in seen:
            continue
        seen.add((day, seed))
        f = [float(x) for x in g[8:13]]
        print(json.dumps(dict(
            day=day, seed=seed, alive=int(g[2]), births=int(g[3]),
            dead=dict(killed=int(g[4]), thirst=int(g[5]), starved=int(g[6]), old=int(g[7])),
            food=dict(fruit=f[0], carcass=f[1], green=f[2], dead=f[3], milk=f[4]),
            drink=float(g[13]), drink_thirst=float(g[14]), thirst=float(g[15]), speed=float(g[16]),
            calls=float(g[17]), attacks=float(g[18]), green=float(g[19]), diet=float(g[20]),
            meat_eaters=int(g[21]), gen_max=int(g[22]), mass=float(g[23]), crowded=int(g[24]), full=int(g[25]),
            capped=int(g[26]), error=float(g[27]), engine="jax", from_log=True)))
    print(f"{len(seen)} records", file=sys.stderr)


if __name__ == "__main__":
    main()
