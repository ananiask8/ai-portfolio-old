structured_clause(Clause, L):-
	Clause =.. [_|[]],
	L = [Clause].
structured_clause(Clause, L):-
	Clause =.. [Functor|[H|[]]],
	structured_clause(H, L1),
	L = [Functor|[L1]],!.
structured_clause(Clause, L):-
	Clause =.. [Functor|[H|T]],
	structured_clause(H, L1),
	structured_clauses(T, L2),
	append([Functor], [L1|L2], L).

structured_clauses([],[]).
structured_clauses([Clause|Rest], L):-
	structured_clause(Clause, L1),
	structured_clauses(Rest, L2),
	append([L1], L2, L).

build_clauses([], []).
build_clauses(Clauses, _):-
	not(is_list(Clauses)),!,fail.
build_clauses(Clauses, L):-
	Clauses = [Next|Rest],
	build_clause(Next, C1),
	build_clauses(Rest, L2),
	append([C1], L2, L).
	
build_clause([], []).
build_clause([], _).
build_clause([Functor|Terms], C):-
	Terms = [Next|Rest],
	not(is_list(Next)),
	iterate_over_terms(Rest, L2),
	append([Next], L2, L),
	C =.. [Functor|L].
build_clause([Functor|Terms], C):-
	Terms = [Next|Rest],
	is_list(Next),
	build_clause(Next, C1),
	iterate_over_terms(Rest, L2),
	append([C1], L2, L),
	C =.. [Functor|L].

iterate_over_terms([], []).
iterate_over_terms(Terms, L):-
	Terms = [Next|Rest],
	not(is_list(Next)),
	iterate_over_terms(Rest, L2),
	append([Next], L2, L).
iterate_over_terms(Terms, L):-
	Terms = [Next|Rest],
	is_list(Next),
	build_clause(Next, C1),
	iterate_over_terms(Rest, L2),
	append([C1], L2, L).

is_flat(L):-flatten(L, L).

replace_all(_, [], []).
replace_all(_, [L], L):-flatten(L, L1), L1 = [].
replace_all(Rule, [Atom|T1], L):-
	not(is_list(Atom)),
	replace(Rule, [Atom], NewClause),
	replace_all(Rule, T1, L2),
	append(NewClause, L2, L).
replace_all(Rule, [StructClause|T1], L):-
	is_list(StructClause), is_flat(StructClause), Clause =.. StructClause,
	replace(Rule, [Clause], NewClause),
	replace_all(Rule, T1, L2),
	append(NewClause, L2, L).
replace_all(Rule, [StructClause|T1], L):-
	is_list(StructClause), not(is_flat(StructClause)),
	StructClause = [Functor|Terms],
	replace_all(Rule, Terms, NewTerms),
	NewStructClause = [Functor|NewTerms],
	build_clause(NewStructClause, NewClause),
	replace(Rule, [NewClause], Replacement),
	replace_all(Rule, T1, L2),
	append(Replacement, L2, L).


fixpoint_old([], Clause, Clause).
fixpoint_old(Rules, Clause, Final):-
	Rules = [CurrentRule|Rest],
	structured_clause(Clause, StructClause),
	is_flat(StructClause),
	match(Rules, Clause, NewClause),
	fixpoint(Rest, NewClause, Final),
	Clause = Final.
fixpoint_old(Rules, Clause, Final):-
	Rules = [CurrentRule|Rest],
	structured_clause(Clause, StructClause),
	not(is_flat(StructClause)),
	StructClause = [Functor|Terms],
	replace_all(CurrentRule, Terms, NewTerms),
	NewStructClause = [Functor|NewTerms],
	build_clause(NewStructClause, NewClause),
	match(Rules, NewClause, NewNewClause),
	fixpoint(Rest, NewNewClause, Final).

search_2(Rules, Start, All):-
	findall(Goal, search_each(Rules, Start, Goal, _), List),
	unique(List, All).

unique([], []).
unique([H|T], L):-
	unique(T, L1),
	\+ member(H, L1), !,
	L = [H|L1].
unique([H|T], L):-
	unique(T, L).

concatenation([], L, L).
concatenation([X|L1], L2, [X|L3]):-
	concatenation(L1, L2, L3).
