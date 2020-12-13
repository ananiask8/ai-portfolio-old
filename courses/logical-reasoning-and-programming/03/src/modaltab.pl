rule_not(G, Z):-
	select(n(n(Psi)), G, Phi),
	append([Psi], Phi, Z), !.

rule_contradiction(G, Z):-
	member(Psi, G),
	member(n(Psi), G),
	Z = [], !.

rule_conjunction(G, Z):-
	select(a(Psi, W), G, Phi),
	append([Psi, W], Phi, Z), !.

rule_disjunction(G, Z):-
	select(n(a(Psi, _)), G, Phi),
	append([n(Psi)], Phi, Z).
rule_disjunction(G, Z):-
	select(n(a(_, Psi)), G, Phi),
	append([n(Psi)], Phi, Z).

sub([], []).
sub([H|T], [H|NT]):-
  sub(T, NT).
sub([_|T], NT):-
  sub(T, NT).

rule_subset(G, Z):-
	sub(G, D),
	[] \= D,
	Z = D.

factor_necessity([], []).
factor_necessity(G, []):-
	not(select(box(_), G, _)).
factor_necessity(G, BG):-
	select(box(Psi), G, Phi),
	factor_necessity(Phi, BG2),
	\+ member(box(Psi), BG2),
	append([box(Psi)], BG2, BG), !.

remove_necessity([], []).
remove_necessity(G, Z):-
	\+ select(box(_), G, _),
	Z = G.
remove_necessity(G, Z):-
	select(box(Psi), G, Phi),
	remove_necessity(Phi, Z2),
	\+ member(Psi, Z2),
	append([Psi], Z2, Z), !.

rule_K(G, Z):-
	factor_necessity(G, BG),
	subset([n(box(Psi))|BG], G),
	subset(G, [n(box(Psi))|BG]),
	select(n(box(Psi)), G, Phi),
	remove_necessity(Phi, W),
	append([n(Psi)], W, Z), !.

prove([], Depth, Depth):-!.
prove(G, Depth, Depth):-
	G \= [], false, !.
prove([], Depth, Counter):-
	Depth > Counter, !.
prove(G, Depth, Counter):-
	Depth > Counter,
	rule_contradiction(G, Z),
	Increment is Counter + 1,
	prove(Z, Depth, Increment).
prove(G, Depth, Counter):-
	Depth > Counter,
	rule_not(G, Z),
	Increment is Counter + 1,
	prove(Z, Depth, Increment).
prove(G, Depth, Counter):-
	Depth > Counter,
	rule_conjunction(G, Z),
	Increment is Counter + 1,
	prove(Z, Depth, Increment).
prove(G, Depth, Counter):-
	Depth > Counter,
	rule_disjunction(G, Z),
	Increment is Counter + 1,
	prove(Z, Depth, Increment).
prove(G, Depth, Counter):-
	Depth > Counter,
	rule_subset(G, Z),
	Increment is Counter + 1,
	prove(Z, Depth, Increment).
prove(G, Depth, Counter):-
	Depth > Counter,
	rule_K(G, Z),
	Increment is Counter + 1,
	prove(Z, Depth, Increment).
prove(Formula, Depth):-
	prove([n(Formula)], Depth, 0), !.