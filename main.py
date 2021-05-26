import gym
# wrappers from https://github.com/higgsfield/RL-Adventure/blob/master/common/wrappers.py
from wrappers import make_atari, wrap_deepmind, wrap_pytorch
import torch
from model import ConvDuelingDQN
from experienceReplayBuffer import ReplayBuffer
from train import train_model

if __name__ == "__main__":
    # Load atari game
    env_id = "Atlantis-v0"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    # Initialize both Convolutional Dueling Q Netowrks
    current_model = ConvDuelingDQN(env.observation_space.shape, env.action_space.n,
                                   env)
    target_model = ConvDuelingDQN(env.observation_space.shape, env.action_space.n,
                                   env)
    gpu = False
    if torch.cuda.is_available():
        # Use GPU in both networks
        current_model = current_model.cuda()
        target_model = target_model.cuda()
        gpu = True


    # Define hyperparameters
    num_frames = 500000
    batch_size = 32
    gamma = 0.99
    lr = 0.00025

    replay_buffer = ReplayBuffer(10000)

    train_model(num_frames, env, current_model, target_model, replay_buffer, batch_size, lr, gamma, gpu)