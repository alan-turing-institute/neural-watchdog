import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")


def getActivation(name):
# the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if __name__ == "__main__":
    args = parser.parse_args()
    activation = {}
    activation_tensors=[]
    m=0
    i= 0
    store_activation_means=[]
    store_activation_var=[]
    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    print(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        env = utils.make_env(args.env, args.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text)
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)
        
        

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

        layers=[0]
        for k in layers:    
            agent.acmodel.actor[k].register_forward_hook(getActivation(str(k)))
            m+=1
            if m>1:
                activation_tensors.append(activation['0'].flatten())
                stacked_inputs= torch.stack(activation_tensors,dim=0)
                print("these are activation tensors", activation_tensors)
                print("this is the overall activation tensor length",len(activation_tensors))
                print("this is a stacked input",stacked_inputs)
                print("stacked_inputs_overall_length", len(stacked_inputs))
                overall_input= len(stacked_inputs)
                if(overall_input % 10 == 0):
                    i+=1
                    stacked_input_var=  torch.var(stacked_inputs, dim=0)
                    stacked_input_mean= torch.mean(stacked_inputs, dim=0)
                    print("Mean",stacked_input_mean, "Var", stacked_input_var)    
                    store_activation_means.append(stacked_input_mean)
                    store_activation_var.append(stacked_input_var)

    end_time = time.time()

    # Print logs
    
    if args.show_logs==True:
        num_frames = sum(logs["num_frames_per_episode"])
        fps = num_frames / (end_time - start_time)
        duration = int(end_time - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
        .format(num_frames, fps, duration,
                *return_per_episode.values(),
                *num_frames_per_episode.values()))
    

    # Print worst episodes

    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
