import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        type=str,
        default='ObsBot2D',
        help='Name of a RL environment')
    parser.add_argument(
        '--metfield_path',
        type=str,
        help=' The directory path containing meteorological data (in .h5 format)')
    parser.add_argument(
        '--result_path',
        type=str,
        help='The path to store results')
    parser.add_argument(
        '--start_timesteps',
        default=10000,
        type=int,
        help='Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network')
    parser.add_argument(
        '--eval_freq',
        default=5000,
        type=int,
        help='How often the evaluation step is performed (after how many timesteps)')
    parser.add_argument(
        '--max_timesteps',
        default=500000,
        type=int,
        help='Total number of iterations/timesteps')
    parser.add_argument(
        '--save_models',
        action='store_true',
        help='Whether or not to save the trained model')
    parser.set_defaults(save_models=False)
    parser.add_argument(
        '--save_env_vid',
        action='store_true',
        help='Whether or not to save videos')
    parser.set_defaults(save_env_vid=False)
    parser.add_argument(
        '--expl_noise',
        default=0.1,
        type=float,
        help='Exploration noise - STD value of exploration Gaussian noise')
    parser.add_argument(
        '--batch_size',
        default=100,
        type=int,
        help='Batch size')
    parser.add_argument(
        '--discount',
        default=0.99,
        type=float,
        help='Discount factor gamma, used in the calculation of the total discounted reward')
    parser.add_argument(
        '--tau',
        default=0.005,
        type=float,
        help='Target network update rate')
    parser.add_argument(
        '--policy_noise',
        default=0.2,
        type=float,
        help='STD of Gaussian noise added to the actions for the exploration purposes')
    parser.add_argument(
        '--noise_clip',
        default=0.5,
        type=float,
        help='Maximum value of the Gaussian noise added to the actions (policy)')
    parser.add_argument(
        '--policy_freq',
        default=2,
        type=int,
        help='Number of iterations to wait before the policy network (Actor model) is updated')

    args = parser.parse_args()

    return args
