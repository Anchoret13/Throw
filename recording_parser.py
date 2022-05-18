import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# matplotlib.rc('font', **font)

ci = lambda x, z=2: (np.mean(x) - z*np.std(x)/len(x)**.5, np.mean(x) + z*np.std(x)/len(x)**.5 )
err_bar = lambda x, z=2: z*np.std(x)/len(x)**.5

loc = "logging/recordings/"

def format_method(inpt):
	if inpt == "q-learning": return "DDPG"
	if inpt == "delta-ddpg": return "delta-DDPG"
	if inpt == "her": return "HER"
	if inpt == "usher": return "USHER"
	return inpt

def format_metric(inpt): 	
	mapping = {'sr': "Success Fraction", }
	if inpt in mapping.keys(): 
		return mapping[inpt]
	else: 
		return ' '.join([s.capitalize() for s in inpt.split("_")])

def format_title(inpt): 	
	if "Torus" in inpt: inpt += "D"
	return (inpt.replace("RandomGridworld", " Random Obstacles")).replace("Gridworld", " Continuous Long/Short Path Environment")

def get_stats(dic):
	out_dict = {}
	for epoch in dic.keys():
		out_dict[epoch] = { "mean": np.mean(dic[epoch]),
				"ci": ci(dic[epoch]),
				"err_bar": err_bar(dic[epoch])
			}
	return out_dict

def match_pattern(env, string):
	prefix = "name_" + env
	l = len(prefix)
	if string[:l] == prefix:
		return True
	else: 
		return False

def parse_recording(env):
	files = sorted(os.listdir(loc))
	relevant_files = [f for f in files if match_pattern(env, f)]
	agents = {}
	for name in relevant_files:
		parses = name.split("__")
		agent_name = parses[-1].split("_")[-1][:-4]
		with open(loc + name, "r") as f: 
			srs = {}
			rewards = {}
			values = {}
			biases = {}
			for line in f: 
				line_split = tuple([t.strip() for t in line.split(",")])
				if len(line_split) <= 1: continue
				epoch = int(line_split[0])
				sr = float(line_split[1])
				reward = float(line_split[2])
				value = float(line_split[3])
				bias = value - reward
				if epoch not in srs.keys(): srs[epoch] = []
				if epoch not in rewards.keys(): rewards[epoch] = []
				if epoch not in values.keys(): values[epoch] = []
				if epoch not in biases.keys(): biases[epoch] = []
				srs[epoch].append(sr)
				rewards[epoch].append(reward)
				values[epoch].append(value)
				biases[epoch].append(bias)

		agents[agent_name] = {'sr': srs, 'rewards': rewards, 'values': values, 'biases': biases}
		agents[agent_name] = {key: get_stats(agents[agent_name][key]) for key in agents[agent_name].keys()}
	return agents

def line_plot(experiment_dict, name="Environment"):
	color_list = ["red", "green", "blue", "purple", "brown"]
	def color_map(x):
		if "usher" in x or "USHER" in x: 
			return "green"
		elif "her" in x or "HER" in x: 
			return "red"
		elif "q-learning" in x or "DDPG" == x:
			return "blue"
		else: 
			return "purple"
	# def sort_key(x): 
	# 	if "her" in x or "HER" in x: 
	# 		return "AAAAA" + x
	# 	else: 
	# 		return x
	methods = [m for m in experiment_dict.keys()]
	# methods = sorted(methods, key=sort_key)
	for metric in experiment_dict[methods[0]].keys():
		for i, method in enumerate(methods):
			epochs = list(experiment_dict[method][metric].keys())
			mean_vals = [experiment_dict[method][metric][epoch]["mean"]     for epoch in epochs]
			lower_ci_list = [experiment_dict[method][metric][epoch]["ci"][0]  for epoch in epochs]
			upper_ci_list = [experiment_dict[method][metric][epoch]["ci"][1]  for epoch in epochs]

			if method == "delta-ddpg" and  (metric == "values" or metric == "biases"): 
				continue
			color=color_map(method)
			plt.plot(epochs, mean_vals, color=color,label=format_method(method))
			plt.fill_between(epochs, lower_ci_list, upper_ci_list, color=color, alpha=.1)

		plt.xlabel("Epoch")
		plt.ylabel(format_metric(metric))
		env_title = format_title(name)
		plt.title(f"{env_title} Performance")
		plt.legend()
		plt.savefig(f"logging/images/time_plots/{name}__{metric}.png")
		plt.show()


envs = sys.argv
if len(envs) <= 1: 
	print("Required arguments: environments")
else: 
	[line_plot(parse_recording(env), name=env) for env in envs[1:]]
