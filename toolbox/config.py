import os


class PathConfig:
    def __init__(self, model_type, model_dir, checkpoint_dir, checkpoint_file,
                 dynamics_file, reward_file, data_dir, base_data_file, controlled_data_file, logger_dir):
        self.model_path = os.path.join(model_dir, f"{model_type}.pt")
        self.checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}.pt")
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
    def __init__(self, model_type, learning_rate, weight_decay, loss_type, optimizer, reward_hidden_size, dynamics_hidden_size, num_epochs, scheduler_gamma, train_split_ratio, batch_size, shuffle):
        self.model_type = model_type
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
        self.shuffle = shuffle


class Config:
    def __init__(self, env_name, num_episodes, device, state_type, controlled, num_iterations, paths, model_config):
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.device = device
        self.state_type = state_type
        self.controlled = controlled
        self.num_iterations = num_iterations
        self.paths = PathConfig(model_type=model_config["model_type"], **paths)
        self.model = ModelConfig(**model_config)
