n(prop(a)).
n(prop(c)).

a(prop(a), prop(b)).
a(prop(a), prop(c)).

i(prop(a), prop(b)).
i(prop(a), prop(c)).
i(prop(c), prop(b)).
i(prop(c), prop(c)).

rule_not([n(n(a))], X).
rule_not([n(n(a)), n(n(b)), n(c), p], X).

rule_contradiction([n(a), a], X).
rule_contradiction([n(a), n(b)], X).

rule_conjunction([a(p, q), b(e, r), a(r, s)], X).
rule_conjunction([o(p, q), b(e, r), o(r, s)], X).

rule_disjunction([n(a(p, q))], X).
rule_disjunction([n(a(p, q)), n(a(r, n(s)))], X).

rule_subset([n(p), a(p, q), i(r, s)], X).

get_necessary_set([a(r, s), box(a(r, t)), box(n(a)), n(box(a)), box(a)], X).

rule_not([n(n(a(box(p), a(box(n(a(p, n(q)))), n(box(q))))))], X).
rule_conjunction([a(box(p), a(box(n(a(p, n(q)))), n(box(q))))], X).
rule_conjunction([box(p), a(box(n(a(p, n(q)))), n(box(q)))], X).
rule_K([box(p), box(n(a(p, n(q)))), n(box(q))], X).
rule_K([n(p), n(box(n(a(n(p), p))))], X).
factor_necessity([n(p), n(box(n(a(n(p), p))))], X).
X = [n(p), n(box(n(a(n(p), p))))], subset(n(box(Y)), X, Z).
rule_disjunction([n(q), p, n(a(p, n(q)))], X).
rule_contradiction([n(p), n(q), p], X).
rule_contradiction([n(n(q)), n(q), p], X).

L = [n(n(a(box(p), a(box(n(a(p, n(q)))), n(box(q))))))], rule_not(L, X1), rule_conjunction(X1, X2), rule_conjunction(X2, X3), rule_K(X3, X4), rule_disjunction(X4, X5), rule_contradiction(X5, X6).

prove(n(a(box(p), a(box(n(a(p, n(q)))), n(box(q))))), 1).
prove(n(a(box(p), a(box(n(a(p, n(q)))), n(box(q))))), 6).
prove(n(a(box(p), a(box(n(a(p, n(q)))), n(box(q))))), 7).

prove(n(a(box(p), n(n(a(box(n(a(p, n(q)))), n(box(q))))))), 7).
prove(n(a(box(p), n(p))), 10).
prove(n(a(n(p), n(box(n(a(n(p), n(n(p)))))))), 7).

cd(sch(i(box(i(A4, B4)), i(box(A4), box(B4))), 1), sch(i(i(A2, i(B2, C2)), i(i(A2, B2), i(A2, C2))), 1), X).

derive(i(box(p), i(box(i(p, q)), box(q))), 4). - doesn't pass (not ok)
derive(i(box(p), i(box(i(p, q)), box(q))), 5). - passes (ok)
derive(i(box(q), i(box(p), box(p))), 4). - passes (ok)
derive(i(box(p), p), 4). - doesn't pass (ok)
derive(i(box(i(p, p)), i(q, q)), 4). - passes (ok)
derive(i(p, box(p)), 5). - doesn't pass (ok)

derive(i(i(box(i(a, b)), box(a)), i(box(i(a, b)), box(b))), 4). - pass (ok)
