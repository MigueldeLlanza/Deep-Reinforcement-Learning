import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def compute_loss(batch, current_model, target_model, lr, gamma, replay_buffer, gpu):
    """
    Compute loss for a given batch
    """
    # Optimizer and learning rate
    optimizer = optim.Adam(current_model.parameters(), lr=lr)
    # Huber loss
    criterion = torch.nn.SmoothL1Loss()

    # Sample from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample_past_exp(batch)
    # Check if gpu is available
    if gpu:
        device = "gpu"
    else:
        device = "cpu"
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Q-values of current network
    current_Q_vals = current_model.forward(states).gather(1, actions.unsqueeze(1))
    current_Q_vals = current_Q_vals.squeeze(1)

    # Next Q-values of target network
    next_Q_vals = target_model.forward(next_states)
    # Max Q-values
    max_next_Q_vals = next_Q_vals.max(1)[0]

    # Expected Q values
    expected_Q_values = rewards + gamma * max_next_Q_vals

    # Compute the loss using the Huber loss function
    loss = criterion(current_Q_vals, expected_Q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def epsilon_by_frame(frame_idx):
    """
    Adapt epsilon depending on the frame
    """
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


def train_model(num_frames, env, current_model, target_model, replay_buffer, batch_size, lr, gamma, gpu):
    losses = []
    all_rewards = []
    episode_reward = 0
    state = env.reset()
    print('Training episode {}...'.format(len(all_rewards)))
    for frame in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame)

        action = current_model.get_action(state, epsilon)

        next_state, reward, done, _ = env.step(action)

        replay_buffer.add_past_exp(state, action, reward, next_state, done)

        state = next_state

        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            print('Training episode {}...'.format(len(all_rewards)))

        if len(replay_buffer.buffer) > batch_size:
            loss = compute_loss(batch_size, current_model, target_model, lr, gamma, replay_buffer, gpu)
            losses.append(loss.item())

        if frame % 1000 == 0:
            plt.figure(figsize=(20, 5))
            plt.subplot(131)
            plt.title('Frame: {} | Reward: {}'.format(frame, np.mean(all_rewards[-10:])))
            plt.plot(all_rewards)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.subplot(132)
            plt.title('Loss per frame')
            plt.xlabel('frames')
            plt.ylabel('loss')
            plt.plot(losses)
            plt.show()

        if frame % 500 == 0:
            target_model.load_state_dict(current_model.state_dict())