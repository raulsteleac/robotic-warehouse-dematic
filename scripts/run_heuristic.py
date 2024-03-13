
from dataclasses import dataclass
from rware.heuristic import heuristic_episode
from rware.warehouse import RewardType, Warehouse

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description="Run tests with vector environments on WarehouseEnv", formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The seed to run with"
    )

args = parser.parse_args()
# https://github.com/raulsteleac/robotic-warehouse-dematic

def info_statistics(infos, global_episode_return, episode_returns):
    _total_deliveries = 0
    _total_clashes = 0
    _total_stuck = 0
    for info in infos:
        _total_deliveries += info["shelf_deliveries"]
        _total_clashes += info["clashes"]
        _total_stuck += info["stucks"]
        info["total_deliveries"] = _total_deliveries
        info["total_clashes"] = _total_clashes
        info["total_stuck"] = _total_stuck
    last_info = infos[-1]
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = global_episode_return
    last_info["episode_returns"] = episode_returns
    return last_info

@dataclass
class TARWAREConfig:
    num_managers =  1
    num_agvs = 12
    num_pickers = 7
    shelf_columns = 5
    column_height = 8
    num_goals = 8
    shelf_rows = 3
    msg_bits = 0
    sensor_range = 1
    request_queue_size = 28
    max_steps = 500
    global_observations = True
    num_episodes= 10000

if __name__ == "__main__":
    env_conf = TARWAREConfig()
    env = Warehouse(
        n_agvs = env_conf.num_agvs,
        n_pickers = env_conf.num_pickers,
        global_observations = env_conf.global_observations,
        n_goals = env_conf.num_goals,
        shelf_columns = env_conf.shelf_columns,
        column_height = env_conf.column_height,
        shelf_rows = env_conf.shelf_rows,
        msg_bits = env_conf.msg_bits,
        sensor_range = env_conf.sensor_range,
        request_queue_size = env_conf.request_queue_size,
        max_steps = env_conf.max_steps,
        max_inactivity_steps = None,
        reward_type = RewardType.INDIVIDUAL,
    )
    seed = args.seed
    env.seed(seed)
    
    completed_episodes = 0
    for _ in range(env_conf.num_episodes):
        infos, global_episode_return, episode_returns = heuristic_episode(env.unwrapped)
        last_info = info_statistics(infos, global_episode_return, episode_returns)
        last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
        print(f"Completed Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]| [Global return={last_info.get('global_episode_return'):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]| [Total clashes={last_info.get('total_clashes'):.2f}]| [Total stuck={last_info.get('total_stuck'):.2f}]")
        completed_episodes += 1