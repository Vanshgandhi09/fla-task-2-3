"""
21CSC301T - CLA-2, Task 3: CFG / PDA case studies
  Case 1  balanced brackets      (CFG G1 ambiguous, G1' unambiguous, DPDA)
  Case 2  palindromes            (CFG G2, nondeterministic PDA)
  Case 3  arithmetic expressions (CFG G3a ambiguous, G3 unambiguous, CFG->PDA)
Run:  python3 task3_models.py      -> prints derivations, stack traces and cross-checks
"""
import itertools
import random
import sys
import textwrap
from collections import deque, defaultdict

EPS = ""


# ----------------------------------------------------------------- grammars
class CFG:
    def __init__(self, name, start, rules):
        """rules: dict  NT -> list of bodies; a body is a tuple of symbols; () is epsilon."""
        self.name, self.start = name, start
        self.rules = {a: [tuple(b) for b in bs] for a, bs in rules.items()}
        self.nts = set(self.rules)

    def is_nt(self, x):
        return x in self.nts

    def productions(self):
        return [(a, b) for a, bs in self.rules.items() for b in bs]

    # --- Earley recogniser (handles epsilon rules and ambiguity)
    def recognises(self, tokens):
        tokens = list(tokens)
        n = len(tokens)
        S = [set() for _ in range(n + 1)]
        prods = [(a, b) for a, bs in self.rules.items() for b in bs]
        for b in self.rules[self.start]:
            S[0].add((self.start, b, 0, 0))
        for i in range(n + 1):
            work = list(S[i])
            while work:
                a, b, d, o = work.pop()
                if d < len(b):
                    x = b[d]
                    if self.is_nt(x):                                   # predict
                        for bb in self.rules[x]:
                            it = (x, bb, 0, i)
                            if it not in S[i]:
                                S[i].add(it); work.append(it)
                        # nullable completion handled below via re-completion
                        for (a2, b2, d2, o2) in list(S[i]):
                            if d2 == len(b2) and a2 == x and o2 == i:
                                it = (a, b, d + 1, o)
                                if it not in S[i]:
                                    S[i].add(it); work.append(it)
                    elif i < n and tokens[i] == x:                       # scan
                        S[i + 1].add((a, b, d + 1, o))
                else:                                                    # complete
                    for (a2, b2, d2, o2) in list(S[o]):
                        if d2 < len(b2) and b2[d2] == a:
                            it = (a2, b2, d2 + 1, o2)
                            if it not in S[i]:
                                S[i].add(it); work.append(it)
        return any(a == self.start and d == len(b) and o == 0 for (a, b, d, o) in S[n])

    # --- number of parse trees (epsilon-free, cycle-free grammars only)
    def count_trees(self, tokens):
        tokens = list(tokens)
        memo = {}

        def sym(x, i, j):
            if not self.is_nt(x):
                return 1 if (j == i + 1 and tokens[i] == x) else 0
            key = (x, i, j)
            if key in memo:
                return memo[key]
            memo[key] = sum(seq(b, 0, i, j) for b in self.rules[x])
            return memo[key]

        def seq(b, k, i, j):
            if k == len(b):
                return 1 if i == j else 0
            if len(b) - k > j - i:
                return 0
            # every remaining symbol must consume >= 1 token (epsilon-free), which also stops
            # left-recursive rules such as E -> E + E from calling themselves on the same span
            hi = j - (len(b) - k - 1)
            return sum(sym(b[k], i, m) * seq(b, k + 1, m, j) for m in range(i + 1, hi + 1))

        return sym(self.start, 0, len(tokens))


def check_derivation(g, steps, target):
    """steps: list of sentential forms (tuples). Verify each is a *leftmost* step of g."""
    for prev, cur in zip(steps, steps[1:]):
        k = next(i for i, x in enumerate(prev) if g.is_nt(x))
        if not any(prev[:k] + b + prev[k + 1:] == cur for b in g.rules[prev[k]]):
            return False
    return steps[0] == (g.start,) and steps[-1] == tuple(target)


def rightmost_check(g, steps, target):
    for prev, cur in zip(steps, steps[1:]):
        k = max(i for i, x in enumerate(prev) if g.is_nt(x))
        if not any(prev[:k] + b + prev[k + 1:] == cur for b in g.rules[prev[k]]):
            return False
    return steps[0] == (g.start,) and steps[-1] == tuple(target)


def leftmost_derivation_from_path(g, tokens):
    """Unique-ish leftmost derivation by BFS over sentential forms (terminal-count pruned)."""
    tokens = tuple(tokens)
    start = (g.start,)
    seen = {start: None}
    q = deque([start])
    while q:
        f = q.popleft()
        if f == tokens:
            out = []
            while f is not None:
                out.append(f); f = seen[f]
            return out[::-1]
        k = next((i for i, x in enumerate(f) if g.is_nt(x)), None)
        if k is None:
            continue
        if f[:k] != tokens[:k]:
            continue
        for b in g.rules[f[k]]:
            nf = f[:k] + b + f[k + 1:]
            if sum(1 for x in nf if not g.is_nt(x)) > len(tokens) or len(nf) > len(tokens) + 2:
                continue
            if nf not in seen:
                seen[nf] = f
                q.append(nf)
    return None


# --------------------------------------------------------------- the grammars
G1 = CFG("G1 (ambiguous Dyck)", "S", {"S": [("S", "S"), ("(", "S", ")"), ("[", "S", "]"), ()]})
G1R = CFG("G1' (unambiguous Dyck)", "S", {"S": [("(", "S", ")", "S"), ("[", "S", "]", "S"), ()]})
G2 = CFG("G2 (palindromes)", "P", {"P": [("a", "P", "a"), ("b", "P", "b"), ("a",), ("b",), ()]})
G3A = CFG("G3a (ambiguous expr)", "E", {"E": [("E", "+", "E"), ("E", "*", "E"), ("(", "E", ")"), ("id",)]})
G3 = CFG("G3 (unambiguous expr)", "E", {"E": [("E", "+", "T"), ("T",)],
                                        "T": [("T", "*", "F"), ("F",)],
                                        "F": [("(", "E", ")"), ("id",)]})
G3LL = CFG("G3' (LL(1) form)", "E", {"E": [("T", "X")], "X": [("+", "T", "X"), ()],
                                     "T": [("F", "Y")], "Y": [("*", "F", "Y"), ()],
                                     "F": [("(", "E", ")"), ("id",)]})


# ------------------------------------------------------ reference recognisers
def dyck_ref(s):
    st = []
    for c in s:
        if c in "([":
            st.append(c)
        else:
            if not st or (c == ")") != (st[-1] == "("):
                return False
            st.pop()
    return not st


def pal_ref(s):
    return s == s[::-1]


def expr_ref(tokens):
    """recursive-descent recogniser for E -> T {+ T}, T -> F {* F}, F -> id | ( E )"""
    toks, pos = list(tokens), [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def E():
        if not T():
            return False
        while peek() == "+":
            pos[0] += 1
            if not T():
                return False
        return True

    def T():
        if not F():
            return False
        while peek() == "*":
            pos[0] += 1
            if not F():
                return False
        return True

    def F():
        if peek() == "id":
            pos[0] += 1
            return True
        if peek() == "(":
            pos[0] += 1
            if not E() or peek() != ")":
                return False
            pos[0] += 1
            return True
        return False

    return E() and pos[0] == len(toks)


# ----------------------------------------------------------------------- PDA
class PDA:
    """delta: dict (state, input or '', top) -> set of (state', push tuple) ; push[0] becomes the new top."""

    def __init__(self, name, start, z0, finals, delta, accept_by="final", prune=None):
        self.name, self.start, self.z0, self.finals = name, start, z0, set(finals)
        self.delta, self.accept_by, self.prune = delta, accept_by, prune

    def moves(self, cfg, tokens):
        q, i, st = cfg
        out = []
        if st:
            top, rest = st[0], st[1:]
            for (q2, push) in sorted(self.delta.get((q, EPS, top), ()), key=repr, reverse=True):  # fixed order: same counts on every run
                out.append((("eps", q, top, q2, push), (q2, i, tuple(push) + rest)))
            if i < len(tokens):
                for (q2, push) in sorted(self.delta.get((q, tokens[i], top), ()), key=repr, reverse=True):
                    out.append(((tokens[i], q, top, q2, push), (q2, i + 1, tuple(push) + rest)))
        return out

    def is_accepting(self, cfg, tokens):
        q, i, st = cfg
        if i != len(tokens):
            return False
        return (q in self.finals) if self.accept_by == "final" else (len(st) == 0)

    def search(self, tokens, limit=200000):
        """BFS. Returns (accepted, path of configs, #configs explored).  Path = shortest accepting run,
        or the longest-progress dead end if rejected."""
        tokens = list(tokens)
        init = (self.start, 0, (self.z0,))
        parent = {init: None}
        todo = deque([init])
        best = init
        explored = 0
        while todo and explored < limit:
            c = todo.popleft()
            explored += 1
            if self.is_accepting(c, tokens):
                return True, self._path(parent, c), explored
            if c[1] > best[1] or (c[1] == best[1] and parent_chain(parent, c) > parent_chain(parent, best)):
                best = c
            for _, nc in self.moves(c, tokens):
                if self.prune and self.prune(nc, tokens):
                    continue
                if nc not in parent:
                    parent[nc] = c
                    todo.append(nc)
        return False, self._path(parent, best), explored

    @staticmethod
    def _path(parent, c):
        out = []
        while c is not None:
            out.append(c); c = parent[c]
        return out[::-1]

    def accepts(self, tokens):
        return self.search(tokens)[0]

    def is_deterministic(self):
        """At most one move per (state, input-or-eps, top), and no state/top pair that has both an
        epsilon-move and an input move."""
        if any(len(v) > 1 for v in self.delta.values()):
            return False
        eps = {(q, t) for (q, a, t) in self.delta if a == EPS}
        return not any((q, t) in eps for (q, a, t) in self.delta if a != EPS)


def parent_chain(parent, c):
    n = 0
    while c is not None:
        n += 1; c = parent[c]
    return n


# ---- Case 1: deterministic PDA for balanced brackets (acceptance by final state q0)
def make_dyck_dpda():
    d = {}
    d[("q0", "(", "Z")] = {("q1", ("P'", "Z"))}
    d[("q0", "[", "Z")] = {("q1", ("B'", "Z"))}
    for X in ("P", "B", "P'", "B'"):
        d[("q1", "(", X)] = {("q1", ("P", X))}
        d[("q1", "[", X)] = {("q1", ("B", X))}
    d[("q1", ")", "P")] = {("q1", ())}
    d[("q1", "]", "B")] = {("q1", ())}
    d[("q1", ")", "P'")] = {("q0", ())}
    d[("q1", "]", "B'")] = {("q0", ())}
    return PDA("DPDA for balanced brackets", "q0", "Z", {"q0"}, d)


# ---- Case 2: nondeterministic PDA for palindromes over {a,b}
def make_pal_pda():
    d = defaultdict(set)
    for x in "ab":
        for X in ("Z", "a", "b"):
            d[("q0", x, X)].add(("q0", (x, X)))      # push phase
            d[("q0", x, X)].add(("q1", (X,)))        # guess: x is the centre symbol (odd length)
        d[("q1", x, x)].add(("q1", ()))              # match phase
    for X in ("Z", "a", "b"):
        d[("q0", EPS, X)].add(("q1", (X,)))          # guess: centre lies between symbols (even length)
    d[("q1", EPS, "Z")].add(("q2", ("Z",)))
    return PDA("NPDA for palindromes", "q0", "Z", {"q2"}, dict(d))


# ---- Case 2b: deterministic PDA for the marked palindromes  w c w^R  (w over {a,b})
def make_marker_dpda():
    d = defaultdict(set)
    for x in "ab":
        for X in ("Z", "a", "b"):
            d[("q0", x, X)].add(("q0", (x, X)))      # push the first half
        d[("q1", x, x)].add(("q1", ()))              # pop and match the second half
    for X in ("Z", "a", "b"):
        d[("q0", "c", X)].add(("q1", (X,)))          # the marker c: switch phase
    d[("q1", EPS, "Z")].add(("q2", ("Z",)))
    return PDA("DPDA for w c w^R", "q0", "Z", {"q2"}, dict(d))


# ---- Case 3: CFG -> PDA conversion (one state, acceptance by empty stack)
def cfg_to_pda(g):
    d = defaultdict(set)
    for a, b in g.productions():
        d[("q", EPS, a)].add(("q", tuple(b)))
    terminals = {x for _, b in g.productions() for x in b if not g.is_nt(x)}
    for t in terminals:
        d[("q", t, t)].add(("q", ()))

    def prune(cfg, tokens):
        _, i, st = cfg
        rem = len(tokens) - i
        # terminals on the stack can never disappear without being matched
        if sum(1 for x in st if not g.is_nt(x)) > rem:
            return True
        # prefix of stack up to first non-terminal must match the input
        for k, x in enumerate(st):
            if g.is_nt(x):
                break
            if i + k >= len(tokens) or tokens[i + k] != x:
                return True
        return len(st) > rem + 4

    return PDA("PDA from " + g.name, "q", g.start, set(), dict(d), accept_by="empty", prune=prune)


def fmt_cfg(cfg, tokens, joiner=""):
    q, i, st = cfg
    rest = joiner.join(tokens[i:]) or "ε"
    stack = " ".join(st) if st else "ε"
    return "(%s, %s, %s)" % (q, rest, stack)


# ------------------------------------------------------------------- the demo
def main():
    out = []
    P = out.append
    P("=" * 76)
    P("TASK 3 DEMONSTRATION - CFG / PDA case studies")
    P("=" * 76)

    # ---- derivations
    P("\n[1] Ambiguity of G1 on '()()()' : two different leftmost derivations")
    A = [("S",), ("S", "S"), ("S", "S", "S"), ("(", "S", ")", "S", "S"), ("(", ")", "S", "S"),
         ("(", ")", "(", "S", ")", "S"), ("(", ")", "(", ")", "S"), ("(", ")", "(", ")", "(", "S", ")"),
         tuple("()()()")]
    B = [("S",), ("S", "S"), ("(", "S", ")", "S"), ("(", ")", "S"), ("(", ")", "S", "S"),
         ("(", ")", "(", "S", ")", "S"), ("(", ")", "(", ")", "S"), ("(", ")", "(", ")", "(", "S", ")"),
         tuple("()()()")]
    for nm, d in (("LMD-1", A), ("LMD-2", B)):
        P("  %s valid=%s : %s" % (nm, check_derivation(G1, d, "()()()"), " => ".join("".join(f) or "ε" for f in d)))
    P("  derivations differ: %s" % (A != B))
    C = [("S",), ("(", "S", ")", "S"), ("(", ")", "S"), ("(", ")", "[", "S", "]", "S"), ("(", ")", "[", "]", "S"), tuple("()[]")]
    P("  G1' LMD of '()[]' valid=%s : %s" % (check_derivation(G1R, C, "()[]"), " => ".join("".join(f) or "ε" for f in C)))

    P("\n[2] Parse-tree counts (ambiguity measure)")
    P("  %-18s %6s %6s" % ("input", "G3a", "G3"))
    for s in ["id+id*id", "id*id+id", "id+id+id", "id*id*id", "(id+id)*id", "id+id*id+id", "id+id+id+id"]:
        toks = s.replace("+", " + ").replace("*", " * ").replace("(", "( ").replace(")", " )").split()
        P("  %-18s %6d %6d" % (s, G3A.count_trees(toks), G3.count_trees(toks)))

    # ---- LMD / RMD for expressions
    t = ["id", "+", "id", "*", "id"]
    lm = leftmost_derivation_from_path(G3, t)
    P("\n[3] G3 leftmost derivation of id+id*id (valid=%s):" % check_derivation(G3, lm, t))
    P("  " + " => ".join(" ".join(f) for f in lm))
    rm = [("E",), ("E", "+", "T"), ("E", "+", "T", "*", "F"), ("E", "+", "T", "*", "id"), ("E", "+", "F", "*", "id"),
          ("E", "+", "id", "*", "id"), ("T", "+", "id", "*", "id"), ("F", "+", "id", "*", "id"), tuple(t)]
    P("  G3 rightmost derivation valid=%s:" % rightmost_check(G3, rm, t))
    P("  " + " => ".join(" ".join(f) for f in rm))
    pa1 = [("E",), ("E", "+", "E"), ("id", "+", "E"), ("id", "+", "E", "*", "E"), ("id", "+", "id", "*", "E"), tuple(t)]
    pa2 = [("E",), ("E", "*", "E"), ("E", "+", "E", "*", "E"), ("id", "+", "E", "*", "E"), ("id", "+", "id", "*", "E"), tuple(t)]
    P("  G3a LMD-1 valid=%s : %s" % (check_derivation(G3A, pa1, t), " => ".join(" ".join(f) for f in pa1)))
    P("  G3a LMD-2 valid=%s : %s" % (check_derivation(G3A, pa2, t), " => ".join(" ".join(f) for f in pa2)))
    pl = leftmost_derivation_from_path(G2, list("abbba"))
    P("  G2 LMD of abbba valid=%s : %s" % (check_derivation(G2, pl, "abbba"), " => ".join("".join(f) or "ε" for f in pl)))

    # ---- PDA traces
    dyck, pal = make_dyck_dpda(), make_pal_pda()
    P("\n[4] DPDA for balanced brackets - stack traces (top of stack on the left)")
    for s in ["([])()", ")(", "(()", "(]"]:
        ok, path, n = dyck.search(list(s))
        P("  input %r -> %s" % (s, "ACCEPT" if ok else "REJECT"))
        P("      " + " |- ".join(fmt_cfg(c, list(s)) for c in path))
    P("\n[5] NPDA for palindromes - accepting runs and an exhaustive rejection")
    for s in ["abba", "aba", "abb"]:
        ok, path, n = pal.search(list(s))
        P("  input %r -> %s   (%d configurations explored)" % (s, "ACCEPT" if ok else "REJECT", n))
        if ok:
            P("      " + " |- ".join(fmt_cfg(c, list(s)) for c in path))
    P("\n[6] CFG->PDA (from G3) on id+id*id, empty-stack acceptance")
    pda3 = cfg_to_pda(G3)
    ok, path, n = pda3.search(t)
    P("  ACCEPT=%s (%d configurations explored), %d configurations on the run" % (ok, n, len(path)))
    P("      " + " |- ".join(fmt_cfg(c, t, " ") for c in path))
    P("  CFG->PDA on 'id + * id' (reject):")
    ok2, path2, n2 = pda3.search(["id", "+", "*", "id"])
    P("  ACCEPT=%s (%d configurations explored)" % (ok2, n2))

    # ---- membership tables
    P("\n[7] Accepted / rejected strings")
    tests = {
        "balanced": (dyck_ref, G1R, ["", "()", "[]", "([])", "()[]", "(()[])()"],
                     ["(", ")(", "(]", "([)]", "(()", "())"]),
        "palin": (pal_ref, G2, ["", "a", "abba", "aba", "abbba", "baab"],
                  ["ab", "abb", "aab", "abab", "baa", "abca"[:3]]),
    }
    for nm, (ref, g, acc, rej) in tests.items():
        for s in acc:
            assert ref(s) and g.recognises(list(s)), (nm, s)
        for s in rej:
            assert (not ref(s)) and (not g.recognises(list(s))), (nm, s)
        P("  %s: %d accepted + %d rejected strings verified against CFG and reference" % (nm, len(acc), len(rej)))
    etests_acc = ["id", "id+id", "id*id", "id+id*id", "(id+id)*id", "((id))"]
    etests_rej = ["id+", "+id", "id id", "(id", "id)", "id**id", "()"]

    def tk(s):
        out, i = [], 0
        while i < len(s):
            if s.startswith("id", i):
                out.append("id"); i += 2
            else:
                out.append(s[i]); i += 1
        return out
    for s in etests_acc:
        assert expr_ref(tk(s)) and G3.recognises(tk(s)) and G3A.recognises(tk(s)) and G3LL.recognises(tk(s)) and pda3.accepts(tk(s)), s
    for s in etests_rej:
        sx = s.replace("id id", "idid") if False else s
        toks = ["id", "id"] if s == "id id" else tk(s)
        assert (not expr_ref(toks)) and (not G3.recognises(toks)) and (not pda3.accepts(toks)), s
    P("  expr: %d accepted + %d rejected strings verified (G3, G3a, G3', PDA, reference parser)" % (len(etests_acc), len(etests_rej)))

    # ---- exhaustive cross-checks
    P("\n[8] Exhaustive cross-checks")
    def run(name, strings, fns):
        n = bad = 0
        for s in strings:
            ref = fns[0](s)
            n += 1
            bad += sum(1 for f in fns[1:] if f(s) != ref)
        P("  %-52s %7d strings, %d disagreements" % (name, n, bad))

    def allstr(alpha, maxlen):
        for L in range(maxlen + 1):
            for t_ in itertools.product(alpha, repeat=L):
                yield list(t_)
    run("balanced: G1, G1', DPDA vs stack checker (len<=8)", allstr("()[]", 8),
        [lambda s: dyck_ref(s), lambda s: G1.recognises(s), lambda s: G1R.recognises(s), lambda s: dyck.accepts(s)])
    run("palindromes: G2, NPDA vs s==reverse(s) (len<=11)", allstr("ab", 11),
        [lambda s: pal_ref(s), lambda s: G2.recognises(s), lambda s: pal.accepts(s)])
    run("expr: G3a, G3, G3', CFG->PDA vs parser (len<=6)", allstr(["id", "+", "*", "(", ")"], 6),
        [expr_ref, G3A.recognises, G3.recognises, G3LL.recognises, pda3.accepts])
    mk = make_marker_dpda()

    def marker_ref(s):
        if s.count("c") != 1:
            return False
        w, v = "".join(s).split("c")
        return w == v[::-1] and set(w) <= set("ab")
    run("w c w^R: DPDA vs reference (all strings over {a,b,c}, len<=9)", allstr("abc", 9),
        [marker_ref, mk.accepts])
    P("  determinism: Dyck DPDA=%s, marker DPDA=%s, palindrome NPDA=%s, CFG->PDA(G3)=%s" %
      (dyck.is_deterministic(), mk.is_deterministic(), pal.is_deterministic(), pda3.is_deterministic()))
    rnd = random.Random(7)
    longer = [[rnd.choice(["id", "id", "+", "*", "(", ")"]) for _ in range(rnd.randint(7, 11))] for _ in range(1500)]
    run("expr: same five recognisers, random len 7..11", longer,
        [expr_ref, G3A.recognises, G3.recognises, G3LL.recognises, pda3.accepts])
    return "\n".join(out)


# ------------------------------------------------- compact "wide" console layout
def side_by_side(text, ncols=2, wrap=96):
    """Lay a long report out in `ncols` side-by-side columns so that it is wide instead of long
    (easy to capture in one screenshot).  Sections (separated by blank lines) are never split."""
    lines = []
    for l in text.rstrip().split("\n"):
        if len(l) <= wrap:
            lines.append(l)
        else:
            ind = len(l) - len(l.lstrip())
            lines += textwrap.wrap(l.strip(), width=wrap, initial_indent=" " * ind,
                                   subsequent_indent=" " * (ind + 4), break_long_words=False,
                                   break_on_hyphens=False)
    blocks, cur = [], []
    for l in lines:
        if l.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(l)
    if cur:
        blocks.append(cur)
    banner = blocks.pop(0) if blocks and set(blocks[0][0].strip()) == {"="} else None
    n, best = len(blocks), None
    for cuts in itertools.combinations(range(1, n), ncols - 1):
        groups = [blocks[a:b] for a, b in zip((0,) + cuts, cuts + (n,))]
        h = max(sum(len(b) for b in g) + len(g) - 1 for g in groups)
        if best is None or h < best[0]:
            best = (h, groups)
    cols = []
    for g in best[1]:
        col = []
        for i, b in enumerate(g):
            if i:
                col.append("")
            col += b
        cols.append(col)
    widths = [max(len(l) for l in c) for c in cols]
    rows = []
    for r in range(max(len(c) for c in cols)):
        rows.append(" | ".join((c[r] if r < len(c) else "").ljust(w) for c, w in zip(cols, widths)).rstrip())
    total = max(len(r) for r in rows)
    out = []
    if banner:
        out += ["=" * total, banner[1].strip() if len(banner) > 2 else banner[0].strip(), "=" * total]
    return "\n".join(out + rows)


def run_cli(report_text):
    """python3 script.py            -> compact, wide (2 columns)
       python3 script.py --cols=3   -> even wider and shorter
       python3 script.py --plain    -> original long, single-column output"""
    args = sys.argv[1:]
    if "--plain" in args:
        print(report_text)
        return
    ncols = 2
    for a in args:
        if a.startswith("--cols="):
            ncols = max(1, int(a.split("=")[1]))
    print(side_by_side(report_text, ncols))


if __name__ == "__main__":
    run_cli(main())
