replace(_, [], []).
replace(Rule, [X|T1], [Y|T2]):-
	replace_term(Rule, X, Y),
	replace(Rule, T1, T2).
replace(Rule, [H|T1], [H|T2]):-
	replace(Rule,T1,T2).

replace_term(Rule, T, R) :-
	T =.. [F|As],
	replace(Rule, As, Gs),
	R =.. [F|Gs].
replace_term(Rule, E, S) :-
	copy_term(Rule, RuleCopy),
	RuleCopy =.. [_, E, S].

match(Rules, Clause, Transformed):-
	Rules = [Rule|_],
	replace(Rule, [Clause], [Transformed]),
	Clause \= Transformed.
match(Rules, Clause, Transformed):-
	Rules = [Rule|Rest],
	replace(Rule, [Clause], [Transformed1]), !,
	match(Rest, Transformed1, Transformed).
match(_, Clause, Clause):-!.

fixpoint(Rules, Clause, Final):-
	match(Rules, Clause, Intermediate),
	Clause \= Intermediate, !,
	fixpoint(Rules, Intermediate, Final).
fixpoint(Rules, Clause, Final):-
	match(Rules, Clause, Final),
	Clause = Final.

search(Rules, Start, Goal):-
	distinct(Goal, search_each(Rules, Start, Goal, _)).
search_each(Rules, Start, Goal, PathToGoal):-
	search_each(Rules, Goal, [n(Start, [])], [], R),
	reverse(R, PathToGoal),
	R = [n(Goal, PathToGoal)|_].
search_each(Rules, Goal,[n(Goal, PathToGoal)|_], _, PathToGoal):-
	member(n(Goal, PathToGoal), PathToGoal).
search_each(Rules, Goal, [n(S, PathToS)|OpenNodes], Visited, PathToGoal):-
 	findall(n(Succ, [_|PathToS]),
 	(match(Rules, S, Succ), \+ member(Succ, Visited)), NewOpenNodes),
 	append(OpenNodes, NewOpenNodes, O),
 	search_each(Rules, Goal, O, [S|Visited], PathToGoal). 
