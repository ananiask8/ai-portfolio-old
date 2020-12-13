:- dynamic schema/1.
schema(sch(i(A1, i(B1, A1)), 1)).
schema(sch(i(i(A2, i(B2, C2)), i(i(A2, B2), i(A2, C2))), 1)).
schema(sch(i(i(n(B3), n(A3)), i(A3, B3)), 1)).
schema(sch(i(box(i(A4, B4)), i(box(A4), box(B4))), 1)).

cd(X, Y, Z):-
	unify_with_occurs_check(sch(A, Count1), X),
	unify_with_occurs_check(sch(i(B, C), Count2), Y),
	unify_with_occurs_check(A, B),
	unify_with_occurs_check(Psi, C),
	NewCount is max(Count1, Count2) + 1,
	unify_with_occurs_check(Z, sch(Psi, NewCount)).

nec(X, Z):-
	unify_with_occurs_check(sch(A, Count), X),
	NewCount is Count + 1,
	unify_with_occurs_check(Z, sch(box(A), NewCount)).

%

derive(Formula, Depth, Predicate):-
	schema(S),
	cd(S, Predicate, Z),
	unify_with_occurs_check(Z, sch(Formula, Count)),
	Count < Depth + 1, !.
derive(Formula, Depth, Predicate):-
	schema(S),
	cd(S, Predicate, Z),
	unify_with_occurs_check(sch(_, Count), Z),
	Count < Depth,
	acyclic_term(Z),
	assertz(schema(Z)),
	derive(Formula, Depth, Z).

derive(Formula, Depth, Predicate):-
	schema(S),
	cd(Predicate, S, Z),
	unify_with_occurs_check(Z, sch(Formula, Count)),
	Count < Depth + 1, !.
derive(Formula, Depth, Predicate):-
	schema(S),
	cd(Predicate, S, Z),
	unify_with_occurs_check(sch(_, Count), Z),
	Count < Depth,
	acyclic_term(Z),
	assertz(schema(Z)),
	derive(Formula, Depth, Z).

%

derive(Formula, Depth, Predicate):-
	schema(S),
	nec(Predicate, Z),
	unify_with_occurs_check(Z, S),
	Count < Depth + 1, !.
derive(Formula, Depth, Predicate):-
	schema(S),
	nec(Predicate, Z),
	unify_with_occurs_check(S, Z),
	Count < Depth,
	acyclic_term(Z),
	assertz(schema(Z)),
	derive(Formula, Depth, Z).

%

derive(Formula, Depth):-
	schema(S),
	derive(Formula, Depth, S), !.

%
