#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pycosat
import sys
import math
import itertools

def print_board(board):
	out = ""
	for j in range(len(board) - 1, -1, -1):
		for i in range(0, len(board)):
			out += board[i][j]
		out += "\r\n"
	print(out)

def build_black_exact_constraints(i, j, k, n):
	pos = []
	neg = []
	if i - 1 >= 1:
		pos.append(n*j + i - 1 + 2*n**2)
		neg.append(-(n*j + i - 1 + 2*n**2))
	if i + 1 <= n:
		pos.append(n*j + i + 1 + 2*n**2)
		neg.append(-(n*j + i + 1 + 2*n**2))
	if j - 1 >= 0:
		pos.append(n*(j - 1) + i + 2*n**2)
		neg.append(-(n*(j - 1) + i + 2*n**2))
	if j + 1 <= n - 1:
		pos.append(n*(j + 1) + i + 2*n**2)
		neg.append(-(n*(j + 1) + i + 2*n**2))

	if len(neg) < k: return [[]]

	cnf = []
	cnf += [list(c) for c in itertools.combinations(pos, len(neg) - k + 1)] # length of disjunction: available spaces minus spaces to occupy plus one
	cnf += [list(c) for c in itertools.combinations(neg, k + 1)] # combine k literals which are true with one extra that will be false and make the prop pass
	# print(cnf)
	return cnf

def if_conditions_then_lighted_constraints(i, j, n):
	cnf = []
	for k in range(1, n + 1):
		if k != i:
			clause = []
			if k > i: clause += [-(n*j + kk) for kk in range(i + 1, k + 1)]
			else: clause += [-(n*j + kk) for kk in range(k, i)]
			clause.append(-(n*j + i)) # -Wij
			clause.append(-(n*j + i + 2*n**2)) # -Lij
			cnf.append(clause + [-(n*j + k + n**2), n*j + k + n**2]) # Xkj : place lighted if all spaces are white
			cnf.append(clause + [-(n*j + k + 2*n**2)]) # -Lkj : no lightbulb there if all spaces are white
		if (k - 1) != j:
			clause = []
			if k > j + 1: clause += [-(n*(kk - 1) + i) for kk in range(j + 2, k + 1)]
			else: clause += [-(n*(kk - 1) + i) for kk in range(k, j + 1)]
			clause.append(-(n*j + i)) # -Wij
			clause.append(-(n*j + i + 2*n**2)) # -Lij
			cnf.append(clause + [-(n*(k - 1) + i + n**2), n*(k - 1) + i + n**2]) # Xik : place lighted if all spaces are white
			cnf.append(clause + [-(n*(k - 1) + i + 2*n**2)]) # -Lik : no lightbulb there if all spaces are white
	return cnf

def if_lighted_then_conditions_constraints(i, j, n):
	clause = set([-(n*j + i), n*j + i + 2*n**2]) # Wij -> Some Lxy in the cross that is not blocked by black cell
	for k in range(i + 1, n + 1):
		if grid[k - 1][j] == "W": clause.add(n*j + k + 2*n**2)
		else: break
	for k in range(i - 1, 0, -1):
		if grid[k - 1][j] == "W": clause.add(n*j + k + 2*n**2)
		else: break
	for k in range(j + 1, n):
		if grid[i - 1][k] == "W": clause.add(n*k + i + 2*n**2)
		else: break
	for k in range(j - 1, -1, -1):
		if grid[i - 1][k] == "W": clause.add(n*k + i + 2*n**2)
		else: break
	return [list(clause)]
if __name__=="__main__":
	for line in sys.stdin:
		n = int(math.sqrt(len(line)))
		grid = [[str(line[n*y + x]) for y in range(n)] for x in range(n)]
		cnf = []
		# print_board(grid)
		# Vars defined for CNF:
		# Wij goes from 0 to n**2
		# Xij goes from n**2 to 2*n**2
		# Lij goes from 2*n**2 to 3*n**2
		for j in range(0, n):
			for i in range(1, n + 1):
				# print((i, j), grid[i - 1][j], n*j + i, n*j + i + n**2, n*j + i + 2*n**2)
				# From init (givens)
				if grid[i - 1][j] == "W":
					# All white spaces (Wij) are True
					cnf.append([n*j + i])
				else:
					# All black spaces (Wij) are False
					cnf.append([-(n*j + i)])

				# Constraints:
				# 1. Exact constraints around black cells
				if grid[i - 1][j] != "W" and grid[i - 1][j] != "B":
					cnf += build_black_exact_constraints(i, j, int(grid[i - 1][j]), n)
				# 2. The light of a lightbulb reaches as far as there are only white spaces in between
				# Lij and Wij and ... and Wik -> Xik
				# Lij and Wij and ... and Wkj -> Xkj
				cnf += if_conditions_then_lighted_constraints(i, j, n)
				# 3. If (i,j) is a white cell, there must exist a lightbulb in the cross (+) of which it is the center position
				# and which is delimited by the black spaces that contain it, and the borders of the board
				# Wij -> Lij or L(i-1)j or L(i+1)j or Li(j-1) or Li(j+1) or ... (expanded in all directions and stopping in each when a black space is met in it)
				cnf += if_lighted_then_conditions_constraints(i, j, n)
				# 4. I can only put lights into white spaces: Lij -> Wij
				cnf.append([-(n*j + i + 2*n**2), n*j + i])

		solution = pycosat.solve(cnf)
		if solution == "UNSAT":
			print("0")
			exit()

		for p in solution:
			l = abs(p)
			if p > 0 and l > 2*n**2 and l <= 3*n**2: grid[(l - 1)%n][(l - 2*n**2 - 1)//n] = "L"

		# print_board(grid)
		out = ""
		for j in range(n):
			for i in range(n):
				out += grid[i][j]

		sys.stdout.write(out + "\r\n")
		exit()
