"""The checklist on the Hassler corpus at recording-session granularity.

The published adapter sets the capture of every record to its class block,
because the release carries no session field. Reviewer 2 (4) asked whether the
blocks are recording sessions. They are not: timestamp gaps inside each block
separate several recording sessions (rev_common.hassler_sessions). This script
runs the unchanged checklist on the same records with the capture field set to
the inferred session, so the two columns can be compared.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import hassler_sessions, load  # noqa: E402
from checklist import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hassler", required=True)
    ap.add_argument("--gap", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = load("hassler", a.hassler)
    # the cached records predate the corrected view description; refresh it
    from adapters import hassler as _fresh
    r.views = {**_fresh(a.hassler).views,
               "hassler_timestamps": r.views["hassler_timestamps"]}
    out = {}
    for label, cap in (("block", r.capture),
                       ("session", hassler_sessions(r, a.gap))):
        rr = copy.copy(r)
        rr.capture = cap
        res = run(rr)
        out[label] = [{"code": x.code, "status": x.status,
                       "value": None if x.value is None else str(x.value)}
                      for x in res]
        print(label, " ".join(f"{x.code}:{x.status}" for x in res))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
