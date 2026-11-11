"""
21CSC301T - CLA-2, Task 2: Finite-automata validators
Applications: (A) password policy, (B) e-mail format, (C) ATM PIN (+ retry-counter FA)

Every automaton is a DFA given as an explicit transition table.  A missing
transition means "go to the dead state" (reject forever).
Run:  python3 task2_validators.py     -> prints the demonstration / self-test report
"""
import contextlib
import io
import itertools
import random
import re
import sys
import textwrap
import time
from collections import deque

DEAD = "dead"
SPECIALS = "!@#$%^&*"


# ----------------------------------------------------------------- generic DFA
class DFA:
    def __init__(self, name, start, accepting, delta, classify):
        self.name, self.start, self.accepting = name, start, set(accepting)
        self.delta, self.classify = delta, classify

    def step(self, q, ch):
        if q == DEAD:
            return DEAD
        return self.delta.get((q, self.classify(ch)), DEAD)

    def run(self, s):
        """Return (accepted, [q0, q1, ..., qn]) -- path stops being interesting at dead."""
        q, path = self.start, [self.start]
        for ch in s:
            q = self.step(q, ch)
            path.append(q)
        return q in self.accepting, path

    def accepts(self, s):
        return self.run(s)[0]


# ------------------------------------------------------- (A) password policy
# Sigma = {U, L, D, S}; any other character is the invalid symbol X (-> dead)
def pw_class(ch):
    if "A" <= ch <= "Z":
        return "U"
    if "a" <= ch <= "z":
        return "L"
    if "0" <= ch <= "9":
        return "D"
    if ch in SPECIALS:
        return "S"
    return "X"


def make_len_dfa(n=8):
    delta = {(k, c): min(k + 1, n) for k in range(n + 1) for c in "ULDS"}
    return DFA("length>=%d" % n, 0, {n}, delta, pw_class)


CLASS_STATES = ["{}", "{U}", "{D}", "{S}", "{U,D}", "{U,S}", "{D,S}", "{U,D,S}"]


def _name(flags):
    return "{" + ",".join(f for f in "UDS" if f in flags) + "}"


def make_class_dfa():
    delta = {}
    for mask in range(8):
        flags = {f for i, f in enumerate("UDS") if mask >> i & 1}
        for c in "ULDS":
            new = flags | ({c} if c in "UDS" else set())
            delta[(_name(flags), c)] = _name(new)
    return DFA("has U,D,S", "{}", {"{U,D,S}"}, delta, pw_class)


LEN_DFA, CLS_DFA = make_len_dfa(), make_class_dfa()


def password_ok(s):
    """Factored design: two DFAs run in parallel; accept iff both accept."""
    return LEN_DFA.accepts(s) and CLS_DFA.accepts(s)


def password_reference(s):
    """Independent, non-automaton specification used to cross-check the DFAs."""
    cls = [pw_class(c) for c in s]
    return (len(s) >= 8 and "X" not in cls and "U" in cls and "D" in cls and "S" in cls)


# ------------------------------------------------------------ (B) e-mail format
# Sigma = {A(letter), D(digit), '.', '_', '-', '@'}; anything else -> dead
def mail_class(ch):
    if ch.isascii() and ch.isalpha():
        return "A"
    if ch.isascii() and ch.isdigit():
        return "D"
    return ch if ch in "._-@" else "X"


MAIL_DELTA_DRAFT = {   # first draft: 9 live states, NOT minimal (q0~q2 and q3~q5 are equivalent)
    ("q0", "A"): "q1", ("q0", "D"): "q1",
    ("q1", "A"): "q1", ("q1", "D"): "q1", ("q1", "."): "q2", ("q1", "_"): "q2",
    ("q1", "-"): "q2", ("q1", "@"): "q3",
    ("q2", "A"): "q1", ("q2", "D"): "q1",
    ("q3", "A"): "q4", ("q3", "D"): "q4",
    ("q4", "A"): "q4", ("q4", "D"): "q4", ("q4", "-"): "q5", ("q4", "."): "q6",
    ("q5", "A"): "q4", ("q5", "D"): "q4",
    ("q6", "A"): "q7", ("q6", "D"): "q4",
    ("q7", "A"): "q8", ("q7", "D"): "q4", ("q7", "."): "q6", ("q7", "-"): "q5",
    ("q8", "A"): "q8", ("q8", "D"): "q4", ("q8", "."): "q6", ("q8", "-"): "q5",
}
MAIL_DELTA = {   # minimal DFA: 7 live states + implicit dead state
    ("q0", "A"): "q1", ("q0", "D"): "q1",                      # start, or just after a local-part separator
    ("q1", "A"): "q1", ("q1", "D"): "q1", ("q1", "."): "q0", ("q1", "_"): "q0",
    ("q1", "-"): "q0", ("q1", "@"): "q2",
    ("q2", "A"): "q3", ("q2", "D"): "q3",                      # after '@' or after '-' in the domain
    ("q3", "A"): "q3", ("q3", "D"): "q3", ("q3", "-"): "q2", ("q3", "."): "q4",
    ("q4", "A"): "q5", ("q4", "D"): "q3",                      # just after a dot in the domain
    ("q5", "A"): "q6", ("q5", "D"): "q3", ("q5", "."): "q4", ("q5", "-"): "q2",
    ("q6", "A"): "q6", ("q6", "D"): "q3", ("q6", "."): "q4", ("q6", "-"): "q2",
}
MAIL = DFA("e-mail", "q0", {"q6"}, MAIL_DELTA, mail_class)
MAIL_DRAFT = DFA("e-mail draft", "q0", {"q8"}, MAIL_DELTA_DRAFT, mail_class)
MAIL_REGEX = re.compile(r"[a1]+([._-][a1]+)*@[a1]+(-[a1]+)*(\.[a1]+(-[a1]+)*)*\.[a]{2,}")


# ------------------------------------------------------------------ (C) ATM PIN
def pin_class(ch):
    return "D" if ("0" <= ch <= "9") else "X"


PIN_DELTA = {("p%d" % i, "D"): "p%d" % (i + 1) for i in range(6)}
PIN = DFA("PIN 4|6 digits", "p0", {"p4", "p6"}, PIN_DELTA, pin_class)

# Session automaton: events 'bad' / 'ok' (semantic result of the bank check)
ATT_DELTA = {("A0", "bad"): "A1", ("A1", "bad"): "A2", ("A2", "bad"): "LOCK",
             ("A0", "ok"): "OK", ("A1", "ok"): "OK", ("A2", "ok"): "OK"}
ATTEMPTS = DFA("retry counter", "A0", {"OK"}, ATT_DELTA, lambda e: e)


# ------------------------------------------------- minimisation / state counting
def reachable_and_minimal(start, alphabet, step, accepting):
    """step(q, a) -> q' on a *complete* DFA.  Returns (#reachable, #minimal)."""
    seen, todo = {start}, deque([start])
    while todo:
        q = todo.popleft()
        for a in alphabet:
            r = step(q, a)
            if r not in seen:
                seen.add(r)
                todo.append(r)
    states = list(seen)
    block = {q: (q in accepting) for q in states}
    while True:
        sig = {q: (block[q], tuple(block[step(q, a)] for a in alphabet)) for q in states}
        ids, new = {}, {}
        for q in states:
            new[q] = ids.setdefault(sig[q], len(ids))
        if len(set(new.values())) == len(set(block.values())):
            return len(states), len(set(new.values()))
        block = new


def password_product_counts():
    def step(q, a):
        if q == DEAD or a == "X":
            return DEAD
        k, mask = q
        return (min(k + 1, 8), mask | {"U": 1, "D": 2, "S": 4}.get(a, 0))
    acc = {(8, 7)}
    return reachable_and_minimal((0, 0), "ULDSX", step, acc)


def email_counts(draft=False):
    delta, acc = (MAIL_DELTA_DRAFT, {"q8"}) if draft else (MAIL_DELTA, {"q6"})
    def step(q, a):
        return delta.get((q, a), DEAD) if q != DEAD else DEAD
    return reachable_and_minimal("q0", ["A", "D", ".", "_", "-", "@", "X"], step, acc)


def pin_counts():
    def step(q, a):
        return PIN_DELTA.get((q, a), DEAD) if q != DEAD else DEAD
    return reachable_and_minimal("p0", "DX", step, {"p4", "p6"})


# ---------------------------------------------------------------- test material
PASSWORD_TESTS = [
    ("Passw0rd!", True, "U, D, S present; length 9"),
    ("Hello@123", True, "all three classes; length 9"),
    ("Abcdef1@", True, "exactly 8 characters (boundary)"),
    ("x9Y!zzzz", True, "mixed order of classes"),
    ("MyP@ssw0rd2026", True, "long password, all classes"),
    ("Pass1!", False, "length 6 < 8"),
    ("passw0rd!", False, "no uppercase letter"),
    ("Passw0rd", False, "no special character"),
    ("Password!", False, "no digit"),
    ("Pass word1!", False, "space is an invalid symbol"),
    ("", False, "empty string"),
]
EMAIL_TESTS = [
    ("user@example.com", True, "simple address"),
    ("first.last@mail.co.in", True, "dot separator in local part, multi-label domain"),
    ("a_b-c@sub.domain.org", True, "_ and - separators, sub-domain"),
    ("john99@site2.net", True, "digits allowed in local part and label"),
    ("x@y-z.io", True, "hyphen inside a domain label; 2-letter TLD"),
    ("user@example", False, "no dot / TLD after @"),
    (".user@example.com", False, "local part starts with a dot"),
    ("user..name@example.com", False, "two consecutive dots"),
    ("user@@example.com", False, "second @"),
    ("user@example.c", False, "TLD has only 1 letter"),
    ("user@-example.com", False, "label starts with a hyphen"),
    ("user@exam ple.com", False, "space is an invalid symbol"),
    ("user@example.c0m", False, "TLD contains a digit"),
]
PIN_TESTS = [
    ("1234", True, "4 digits"),
    ("0000", True, "leading zeros are fine: PIN is a string, not a number"),
    ("0007", True, "leading zeros"),
    ("987654", True, "6-digit PIN"),
    ("123456", True, "6-digit PIN"),
    ("123", False, "too short (3)"),
    ("12345", False, "length 5 is not allowed"),
    ("12a4", False, "non-digit character"),
    ("1234567", False, "too long (7)"),
    ("12 34", False, "space inside"),
    ("", False, "empty"),
]
SESSION_TESTS = [
    (["ok"], True, "correct on first try"),
    (["bad", "ok"], True, "correct on 2nd try"),
    (["bad", "bad", "ok"], True, "correct on 3rd try"),
    (["bad", "bad", "bad"], False, "3 failures -> card retained (LOCK)"),
    (["bad", "bad", "bad", "ok"], False, "LOCK is absorbing"),
    (["bad", "bad"], False, "session ended without success"),
    (["bad", "bad", "bad", "bad"], False, "nothing happens after LOCK"),
]


def fmt_path(path):
    return " > ".join(str(p) for p in path)


# ------------------------------------------------------------ self-tests & demo
def exhaustive_checks():
    out = []
    # password: abstract alphabet, exhaustive to length 8 + 100k random length 9..14
    import itertools
    bad = 0
    n = 0
    for L in range(0, 9):
        for t in itertools.product("ULDSX", repeat=L):
            s = "".join({"U": "A", "L": "a", "D": "5", "S": "!", "X": " "}[c] for c in t)
            n += 1
            bad += password_ok(s) != password_reference(s)
    rnd = random.Random(1)
    for _ in range(100000):
        L = rnd.randint(9, 14)
        s = "".join(rnd.choice("Aa5! ") for _ in range(L))
        n += 1
        bad += password_ok(s) != password_reference(s)
    out.append(("password DFA pair vs reference predicate", n, bad))
    # e-mail: DFA vs regular expression over abstract alphabet
    tr = {"a": "a", "1": "5", ".": ".", "_": "_", "-": "-", "@": "@"}
    n = bad = 0
    for L in range(0, 8):
        for t in itertools.product("a1._-@", repeat=L):
            s_abs = "".join(t)
            s = "".join("x" if c == "a" else tr[c] for c in s_abs)
            n += 1
            bad += MAIL.accepts(s) != bool(MAIL_REGEX.fullmatch(s_abs))
    for _ in range(100000):
        L = rnd.randint(8, 16)
        s_abs = "".join(rnd.choice("aaa11._-@") for _ in range(L))
        s = "".join("x" if c == "a" else tr[c] for c in s_abs)
        n += 1
        bad += MAIL.accepts(s) != bool(MAIL_REGEX.fullmatch(s_abs))
        bad += MAIL_DRAFT.accepts(s) != MAIL.accepts(s)
    out.append(("e-mail DFA (and draft) vs regular expression", n, bad))
    # PIN: D^4 | D^6
    n = bad = 0
    for L in range(0, 10):
        for t in itertools.product("0x", repeat=L):
            s = "".join(t)
            n += 1
            bad += PIN.accepts(s) != (L in (4, 6) and "x" not in s)
    out.append(("PIN DFA vs (D^4 | D^6)", n, bad))
    return out


def redos_demo():
    """Naive backtracking regex vs DFA on the near-miss input 'aaaa...a!'."""
    naive = re.compile(r"^([a-z0-9]+[._-]?)+@")
    rows = []
    for n in (16, 20, 22, 24):
        s = "a" * n + "!"
        t0 = time.perf_counter()
        naive.match(s)
        t_re = time.perf_counter() - t0
        t0 = time.perf_counter()
        MAIL.accepts(s)
        t_dfa = time.perf_counter() - t0
        rows.append((n + 1, t_re, t_dfa))
    return rows


def report():
    print("=" * 74)
    print("TASK 2 DEMONSTRATION - finite-automata validators")
    print("=" * 74)
    for title, tests, fn in (("A. PASSWORD", PASSWORD_TESTS, password_ok),
                             ("B. E-MAIL", EMAIL_TESTS, MAIL.accepts),
                             ("C. ATM PIN", PIN_TESTS, PIN.accepts)):
        print("\n" + title)
        print("%-24s %-9s %-9s %s" % ("input", "expected", "DFA", "reason"))
        for s, exp, why in tests:
            got = fn(s)
            flag = "" if got == exp else "   <-- MISMATCH"
            print("%-24r %-9s %-9s %s%s" % (s, "ACCEPT" if exp else "REJECT",
                                          "ACCEPT" if got else "REJECT", why, flag))
    print("\nD. ATM RETRY COUNTER")
    for ev, exp, why in SESSION_TESTS:
        ok, path = ATTEMPTS.run(ev)
        print("%-32s %-8s %s   [%s]" % (ev, "ACCEPT" if ok else "REJECT", why, fmt_path(path)))
        assert ok == exp
    print("\nState-transition traces")
    for fn_name, dfa, s in (("e-mail accept", MAIL, "x@y-z.io"), ("e-mail reject", MAIL, "a@b.c"),
                            ("PIN accept", PIN, "0007"), ("PIN reject", PIN, "12a4")):
        ok, path = dfa.run(s)
        print("  %-14s %-18r %s  -> %s" % (fn_name, s, fmt_path(path), "ACCEPT" if ok else "REJECT"))
    for s in ("Passw0rd!", "Password!"):
        _, p1 = LEN_DFA.run(s)
        _, p2 = CLS_DFA.run(s)
        print("  password %r -> %s" % (s, "ACCEPT" if password_ok(s) else "REJECT"))
        print("      length : %s" % fmt_path(p1))
        print("      classes: %s" % fmt_path(p2))
    print("\nExhaustive cross-checks (DFA vs independent specification)")
    for name, n, bad in exhaustive_checks():
        print("  %-42s %8d strings, %d disagreements" % (name, n, bad))
    print("\nState counts (reachable / minimal, including the dead state)")
    print("  password product DFA : %d / %d   (factored design: 9 + 8 = 17 states)" % password_product_counts())
    print("  e-mail first draft   : %d / %d" % email_counts(draft=True))
    print("  e-mail final DFA     : %d / %d" % email_counts())
    print("  PIN DFA              : %d / %d" % pin_counts())
    print("\nBacktracking regex  ^([a-z0-9]+[._-]?)+@  vs DFA on 'a'*n + '!'")
    for n, t_re, t_dfa in redos_demo():
        print("  length %2d: regex %9.4f s   DFA %9.6f s" % (n, t_re, t_dfa))


# ------------------------------------------------- compact "wide" console layout
def side_by_side(text, ncols=2, wrap=100):
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


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report()
    run_cli(buf.getvalue())


if __name__ == "__main__":
    main()
