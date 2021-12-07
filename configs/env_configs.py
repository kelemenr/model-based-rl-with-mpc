env_config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15
    },
    "vehicles_density": 1.2,
    "reward_speed_range": [25, 30],
    "collision_reward": -2,
}

speed_range_env_config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15
    },
    "vehicles_density": 1.2,
    "reward_speed_range": [25, 30],
    "collision_reward": -2,
}

high_speed_range_env_config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15
    },
    "vehicles_density": 1.4,
    "reward_speed_range": [26, 30],
    "collision_reward": -2,
}

big_state_env_config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    },
    "vehicles_density": 1.2,
    "reward_speed_range": [25, 30],
    "collision_reward": -2,
}

dense_env_config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15
    },
    "vehicles_density": 1.5,
    "reward_speed_range": [25, 30],
    "collision_reward": -2,
}

intersection_env_config = {
    "observation": {
        "type": "OccupancyGrid",
        "absolute": False}
}
