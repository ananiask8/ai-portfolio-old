#!/usr/bin/env python3

import random
from math import factorial, log
from itertools import permutations, chain, combinations

from Template import Node, DirEdge, Graph, Agent

class Game:
  def banzhaf(self):
    pass

  def shapley(self):
    pass

  def shapley_approx(self, m):
    pass

class FlowGame(Graph, Game):
  def __init__(self, A=None, G=None):
    super().__init__(A=A, G=G)
    self.A = A
    self.G = G

  def get_worth_of_coalition(self, A=None):
    pass

  def compute_values_of_coalitions(self):
    pass

def prepare(args):
  input_file = open(args.input, "r")
  size_of_V, size_of_E, size_of_N = map(int, input_file.readline().split(" "))
  G = Graph()
  V = {id: Node(id) for id in range(size_of_V)}
  A = {Agent(id): set() for id in range(1, size_of_N + 1)}
  for i in range(size_of_E):
    tail_id, head_id, cap_e, agent_id = map(int, input_file.readline().split(" "))
    a = Agent(agent_id)
    u, v = V[tail_id], V[head_id]
    e = DirEdge(tail=u, head=v, l=0, u=cap_e)
    G.add_edge(e)
    A[a] |= {e}
  return A, G

def run(FG):
  pass

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.io", type=str, help="Name of file to read.")
    parser.add_argument("--output", default="output.io", type=str, help="Name of file to write to.")
    args = parser.parse_args()

    A, G = prepare(args)
    FG = FlowGame(A, G)
    run(FG)
