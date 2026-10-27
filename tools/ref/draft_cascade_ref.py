# UNTRUSTED (2026-10-24): its own example() assertion fails on the trace — fix code or expectation before using it to cross-check C12.
"""Independent C12 cascade reference (stdlib only). Cross-check numbers, not a submission.

S1  Longest suffix n in 6..2. Every earlier hit yields a continuation of at most k tokens.
S2  margin = count(best next token) - count(runner-up) at that n (runner-up 0 if unique).
    Ties on (count, most-recent index) take that hit. margin >= tau -> draft k, else k_amb
    from the same hit. k_amb == 0 is an empty draft.
S3  On the first mismatched draft token, ban (suffix, bad token) for every n in 2..6
    at the context after the accepted prefix. Caller keeps `bans` for one request.
S4  Empty draft -> one DFlash draw. Cycle (4,4,4,4,5) is P(4)=0.8, P(5)=0.2, mean 4.2.
    m = min(draw, tokens left). A step is n-gram or DFlash, never both.
Accepted/step = (sum of draft hits m) / steps. The +1 bonus is not counted.
"""


def propose(ctx, bans, tau, k, k_amb, n_lo=2, n_hi=6):
    for n in range(min(n_hi, len(ctx) - 1), n_lo - 1, -1):
        suf = tuple(ctx[-n:])
        groups = {}
        for i in range(len(ctx) - n):
            if tuple(ctx[i:i + n]) != suf:
                continue
            cont = ctx[i + n:i + n + k]
            if cont and (suf, cont[0]) not in bans:
                groups.setdefault(cont[0], []).append((i, cont))
        if not groups:
            continue
        best = max(groups, key=lambda t: (len(groups[t]), max(p for p, _ in groups[t])))
        ms = groups[best]
        second = sorted(len(v) for v in groups.values())[-2] if len(groups) > 1 else 0
        margin = len(ms) - second
        take = k if margin >= tau else k_amb
        return list(max(ms)[1][:take]), n, margin
    return [], 0, 0


def simulate(context, target, bans=None, tau=2, k=4, k_amb=1,
             n_lo=2, n_hi=6, draws=(4, 4, 4, 4, 5), mode="cascade"):
    """mode: 'cascade' | 'ngram' | 'dflash'. Draws advance only on DFlash steps."""
    ctx, bans = list(context), (set() if bans is None else bans)
    idx = acc = steps = di = 0
    trace, L = [], len(target)
    while idx < L:
        rem = L - idx
        if mode == "dflash":
            draft, n, margin, kind = [], 0, 0, "dflash"
        else:
            draft, n, margin = propose(ctx, bans, tau, k, k_amb, n_lo, n_hi)
            kind = "ngram" if draft else ("ar" if mode == "ngram" else "dflash")
        if kind == "dflash":
            m = min(draws[di % len(draws)], rem)
            di += 1
        elif kind == "ar":
            m = 0
        else:
            m = 0
            while m < len(draft) and m < rem and draft[m] == target[idx + m]:
                m += 1
            if m < len(draft) and m < rem:
                seen = ctx + target[idx:idx + m]
                for w in range(n_lo, n_hi + 1):
                    if len(seen) >= w:
                        bans.add((tuple(seen[-w:]), draft[m]))
        adv = min(m + 1, rem)
        ctx.extend(target[idx:idx + adv])
        idx += adv
        acc += m
        steps += 1
        trace.append((kind, n, margin, m))
    return {"acc": acc, "steps": steps, "mean": (acc / steps if steps else 0.0),
            "trace": trace, "bans": bans}


def example():
    """tau=2, k=4, k_amb=1. E[accepted/step] = 61/20 = 3.05.

    ctx = [1,2,5,8,8,8]*4 + [1,2,6,4,9,9,1,7]*2 + [3,1,2]
    T   = [6, 4,9,9,1,2, 6,4,9,9,7, 10,11,12,13,14]
    s1 n=2 margin=4-2=2 draft[5,8,8,8] vs 6 -> m=0, ban token 5, bonus 6
    s2 n=3 margin=2 draft[4,9,9,1] matches, bonus 2 -> m=4
    s3 (1,2)->5 dropped; (1,2)->6 count 2, margin 2, draft[6,4,9,9] -> m=4, bonus 7
    s4 no hit, 5 gold left. draw 4 -> m=4 (p=0.8); draw 5 -> m=5 (p=0.2); E[m]=4.2
    E[acc] = 8 + 4.2 = 12.2 = 61/5 over 4 steps, so E[mean] = 61/20 = 3.05
    ngram-only (empty draft is m=0, advance 1): acc=8, steps=8, mean=1
    dflash-only, 6 gold tokens: draw 4 -> m=4 then m=1 (acc 5 / 2 steps = 2.5);
    draw 5 -> m=5 (acc 5 / 1 step = 5). E[ratio] = 0.8*2.5 + 0.2*5 = 3
    """
    block_a, block_b = [1, 2, 5, 8, 8, 8], [1, 2, 6, 4, 9, 9, 1, 7]
    ctx = block_a * 4 + block_b * 2 + [3, 1, 2]
    target = [6, 4, 9, 9, 1, 2, 6, 4, 9, 9, 7, 10, 11, 12, 13, 14]
    hi = simulate(ctx, target, draws=(4,))
    lo = simulate(ctx, target, draws=(5,))
    ngram = simulate(ctx, target, mode="ngram")
    assert hi["trace"] == [
        ("ngram", 2, 2, 0), ("ngram", 3, 2, 4),
        ("ngram", 2, 2, 4), ("dflash", 0, 0, 4),
    ]
    assert lo["trace"][3] == ("dflash", 0, 0, 5)
    assert (hi["acc"], lo["acc"], hi["steps"], lo["steps"]) == (12, 13, 4, 4)
    assert 4 * hi["acc"] + lo["acc"] == 61
    assert ((1, 2), 5) in hi["bans"] and len(hi["bans"]) == 5
    assert (ngram["acc"], ngram["steps"], ngram["mean"]) == (8, 8, 1.0)
    d4 = simulate([], [0] * 6, draws=(4, 4), mode="dflash")
    d5 = simulate([], [0] * 6, draws=(5,), mode="dflash")
    assert (d4["acc"], d4["steps"], d5["acc"], d5["steps"]) == (5, 2, 5, 1)
    return 61 / 20


if __name__ == "__main__":
    example()
