from models import (DynamicsModel, DynamicsModelLSTM, RewardModel,
                    RewardModelLSTM)
from toolbox.factory import (loss_function_factory, model_factory,
                             optimizer_factory)


def calculate_state_size(state_shape, state_type):
    if state_type == "OccupancyGrid":
        return state_shape[0] * state_shape[1] * state_shape[2]
    elif state_type == "Kinematics":
        return state_shape[0] * state_shape[1]


def create_dynamics_model(env, config):
    state_shape = env.observation_space.shape
    state_size = calculate_state_size(state_shape, config.state_type)

    dynamics_model = model_factory(config.model.dynamics_model)
    model = dynamics_model(state_size=state_size, action_size=1,
                           hidden_size=config.model.dynamics_hidden_size,
                           dt=1 / env.unwrapped.config["policy_frequency"])

    optimizer_function = optimizer_factory(config.model.optimizer)
    optimizer = optimizer_function(model.parameters(
    ), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

    loss_function = loss_function_factory(config.model.loss_type)
    return model, optimizer, loss_function


def create_reward_model(env, config):
    state_shape = env.observation_space.shape
    state_size = calculate_state_size(state_shape, config.state_type)

    reward_model = model_factory(config.model.reward_model)
    model = reward_model(state_size=state_size, action_size=5,
                         hidden_size=config.model.reward_hidden_size)

    optimizer_function = optimizer_factory(config.model.optimizer)
    optimizer = optimizer_function(model.parameters(
    ), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

    loss_function = loss_function_factory(config.model.loss_type)
    return model, optimizer, loss_function


def setup_model(model_type):
    if model_type in ["dynamics", "dynamics-lstm"]:
        return create_dynamics_model
    elif model_type in ["reward", "reward-lstm"]:
        return create_reward_model
    else:
        raise ValueError("Unknown model type: {}".format(model_type))
