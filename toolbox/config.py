import os


class PathConfig:
    def __init__(self, model_to_train, model_dir, checkpoint_dir, checkpoint_file, dynamics_file, reward_file, data_dir, base_data_file, controlled_data_file, logger_dir):
        self.model_path = os.path.join(model_dir, f"{model_to_train}.pt")
        self.checkpoint_path = os.path.join(
            checkpoint_dir, f"{model_to_train}.pt")
        self.base_data_path = os.path.join(data_dir, base_data_file)
        self.controlled_data_path = os.path.join(
            data_dir, controlled_data_file)
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.dynamics_file = dynamics_file
        self.reward_file = reward_file
        self.data_dir = data_dir
        self.base_data_file = base_data_file
        self.controlled_data_file = controlled_data_file
        self.logger_dir = logger_dir


class ModelConfig:
    def __init__(self, model_to_train, dynamics_model, reward_model, learning_rate, weight_decay, loss_type, optimizer, reward_hidden_size, dynamics_hidden_size, num_epochs, scheduler_gamma, train_split_ratio, batch_size):
        self.model_to_train = model_to_train
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.reward_hidden_size = reward_hidden_size
        self.dynamics_hidden_size = dynamics_hidden_size
        self.num_epochs = num_epochs
        self.scheduler_gamma = scheduler_gamma
        self.train_split_ratio = train_split_ratio
        self.batch_size = batch_size


class Config:
    def __init__(self, env_name, num_episodes, device, state_type, controlled, num_iterations, planning_horizon, seed, paths, model_config):
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.device = device
        self.state_type = state_type
        self.controlled = controlled
        self.num_iterations = num_iterations
        self.planning_horizon = planning_horizon
        self.seed = seed
        self.paths = PathConfig(
            model_to_train=model_config["model_to_train"], **paths)
        self.model = ModelConfig(**model_config)
