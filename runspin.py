from spinup.utils.test_policy import load_policy, run_policy
import mycart
_, get_action = load_policy('./logs/')
env = mycart.MyCartPoleEnv()
run_policy(env, get_action)