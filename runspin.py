from spinup.utils.test_policy import load_policy, run_policy
import mycartCont
import sys

dirname = sys.argv[1:]

_, get_action = load_policy(dirname[0])
env = mycartCont.MyCartContEnv()
run_policy(env, get_action)
