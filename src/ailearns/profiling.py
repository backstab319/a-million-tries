"""GPU time per kernel of a compiled JAX function, from the profiler's trace (ep16 step 9g): what's slow, measured."""

import jax


def profile(fn, *args, top=40, detail=0):
    """GPU time per kernel over one timed call, from the profiler's Perfetto trace, each kernel shown with its output
    shape and the JAX op path it came from (the compiled program's metadata; this JAX keeps source lines elsewhere).
    detail: for the top `detail` kernels, also the inputs of the fused computation (each with its shape)."""
    import collections, glob, gzip, json, re, tempfile  # noqa: E401
    text = fn.lower(*args).compile().as_text()
    where = {}  # instruction name -> "output shape  op path"
    calls = {m.group(1).replace(".", "_"): m.group(2) for m in re.finditer(r"%?([\w.\-]+) = [^\n]*?calls=%?([\w.\-]+)", text)}
    heads = {m.group(1): m.group(2) for m in re.finditer(r"\n%?([\w.\-]+) (\([^\n]*\)) -> [^\n]*\{", text)}
    for m in re.finditer(r"%?([\w.\-]+) = [^\n]*?metadata=\{([^}]*)\}", text):
        meta = dict(re.findall(r'(\w+)="?([^" ]*)"?', m.group(2)))
        src = meta.get("source_file", "").rsplit("/", 1)[-1]
        shape = re.match(r"\s*(\S+)", m.group(0).split(" = ", 1)[1]).group(1)
        where[m.group(1).replace(".", "_")] = f"{shape[:34]:34s} {meta.get('op_name', '')[-70:]}"
    jax.block_until_ready(fn(*args))  # compile
    out = tempfile.mkdtemp()
    with jax.profiler.trace(out, create_perfetto_trace=True):
        jax.block_until_ready(fn(*args))
    path = glob.glob(f"{out}/**/*perfetto_trace.json.gz", recursive=True)[0]
    ev = json.load(gzip.open(path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    gpu = {e["pid"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"
           and "GPU" in str(e.get("args", {}).get("name", ""))}
    if not gpu:
        print("no GPU process in the trace; processes:", sorted({str(e.get("args", {}).get("name")) for e in ev
                                                                  if e.get("name") == "process_name"}), flush=True)
    print(f"{len(where)} instructions with source lines in the compiled program", flush=True)
    tot, n = collections.Counter(), collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and e.get("pid") in gpu:
            tot[e["name"]] += e.get("dur", 0) / 1e6
            n[e["name"]] += 1
    allt = sum(tot.values())
    print(f"GPU kernels: {len(tot)} kinds, {sum(n.values())} launches, {allt:.2f} s busy", flush=True)
    for i, (name, t) in enumerate(tot.most_common(top)):
        key = re.sub(r"[.]", "_", name.split(":")[0])
        print(f"  {t:6.3f} s {t / allt:5.1%} x{n[name]:6d}  {name[:40]:40s} {where.get(key, '?')}", flush=True)
        if i < detail and key in calls:
            print(f"        inputs: {heads.get(calls[key], '?')[:600]}", flush=True)
