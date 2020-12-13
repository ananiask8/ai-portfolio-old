#!/usr/bin/env python2

import time
import os 
import lmcut

test_set_dir = "../pui-test-set"
for problem_dir in os.listdir(test_set_dir):
    for problem_instance in filter(lambda i: i != "domain.pddl" and ".pddl" in i, os.listdir("%s/%s" % (test_set_dir, problem_dir))):
        # if problem_dir == 'depot' and problem_instance == 'pfile1.pddl':
        if problem_dir == 'woodworking':
            problem_id = "%s/%s/%s" % (test_set_dir, problem_dir, problem_instance.split(".")[0])
            print(problem_id)
            start = time.time()
            os.system("python lmcut.py %s.strips %s.fdr" % (problem_id, problem_id))
            end = time.time()
            print(end - start)
            print()
        # break
        # os.system("./validate -v domain.pddl %s.pddl %s.plan" % (problem_id, problem_id))
    # break
