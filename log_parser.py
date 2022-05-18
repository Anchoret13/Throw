import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

ci = lambda x, z=2: (np.mean(x) - z*np.std(x)/len(x)**.5, np.mean(x) + z*np.std(x)/len(x)**.5 )
err_bar = lambda x, z=2: z*np.std(x)/len(x)**.5
keylist = ["success_rate", "average_reward", "average_initial_value"]
#loc = "train_her_mod_logging"
# loc = ""

def format_name(inpt): 	
	return ' '.join([s.capitalize() for s in inpt.split("_")])

def format_title(inpt): 	
	return (inpt.replace("RandomGridworld", " Random Obstacles")).replace("Gridworld", " Continuous Long/Short Path Environment")

def format_method(inpt):
	if "1-goal" in inpt: 
		return "HER"
	elif "2-goal ratio" in inpt:
		return "USHER (proposed)"
		if "delta-probability" in inpt: 
			return "USHER (delta probability model) (proposed)"
		else: 
			return "USHER (bellman probability model)(proposed)"
	elif "2-goal" in inpt:
		return "2-goal HER"

	elif "Q-learning" in inpt:
		return "Q-learning"
	assert False

def parse_log(env_name):
	filename = f"{loc}/{env_name}.txt"
	print(f"Filename: {filename}")
	with open(filename, "r") as f: 
		file = f.read()

	file_list = file.split("\n")
	experiment_dict = {}
	# current_list = []
	current_dict = {key: [] for key in keylist}
	for line in reversed(file_list):
		if line == '': 
			continue
		elif "run" in line: 
			env, noise_string, method = tuple(line.split(","))
			noise = float(noise_string[:-len("noise")])
			if method not in experiment_dict.keys():
				experiment_dict[method] = {key: [] for key in keylist + ["bias"]}
			for key in keylist:
				# if not len(current_dict[key]) > 0:
				values = current_dict[key]
				# if key == "average_reward": values = [v/(1-.9**20) for v in values]
				# if key == "average_initial_value": values = [v*(1-.9**20) for v in values]
				mean = np.mean(values)
				list_ci = ci(values)
				# experiment_dict[method].append((noise, current_list))
				# if "2-goal ratio" in line:
				experiment_dict[method][key].append((noise, mean, list_ci))

			r = np.array(current_dict["average_reward"])
			v = np.array(current_dict["average_initial_value"])
			bias = v-r
			mean_bias = np.mean(bias)
			ci_bias = ci(bias)
			experiment_dict[method]["bias"].append((noise, mean_bias, ci_bias))

			current_dict = {key: [] for key in keylist}
		else: 
			# data = line.split(",")
			# num = 0#float(line.split(":")[-1])
			# for item in data: 
				# if "success_rate" in item: 
				# 	num = float(line.split(":")[-1])
			for key in keylist:
				if key in line: 
					current_dict[key].append(float(line.split(":")[-1]))
			# current_list.append(num)

	return experiment_dict

def plot_log(experiment_dict, name="Environment"):
	method = [i for i in experiment_dict.keys()][0]
	if len(experiment_dict[method][keylist[0]]) > 1:
		line_plot(experiment_dict, name=name)
	else: 
		bar_plot(experiment_dict, name=name)

def line_plot(experiment_dict, name="Environment"):
	color_list = ["red", "green", "blue", "brown", "pink"]
	methods = [m for m in experiment_dict.keys()]
	for key in experiment_dict[methods[0]].keys():
		i=0
		for method in experiment_dict.keys():
			experiment_dict[method][key] = sorted(experiment_dict[method][key], key=lambda x: x[0])
			noise_list = [elem[0] for elem in experiment_dict[method][key]]
			mean_list = [elem[1] for elem in experiment_dict[method][key]]
			upper_ci_list = [elem[2][0] for elem in experiment_dict[method][key]]
			lower_ci_list = [elem[2][1] for elem in experiment_dict[method][key]]

			plt.plot(noise_list, mean_list, color=color_list[i],label=format_method(method))
			plt.fill_between(noise_list, lower_ci_list, upper_ci_list, color=color_list[i], alpha=.1)

			i += 1

		plt.xlabel("Noise (fraction of maximum action)")
		plt.ylabel(format_name(key))
		env_title = format_title(name)
		plt.title(f"{env_title} Performance")
		plt.legend()
		plt.savefig(f"{loc}/images/{name}__{key}.png")
		plt.show()


	# font = {'family' : 'normal',
 #        # 'weight' : 'bold',
 #        'size'   : 1}

	for key in ["average_reward", "average_initial_value"]:
		i=0
		for method in experiment_dict.keys():
			noise_list = [elem[0] for elem in experiment_dict[method][key]]
			mean_list = [elem[1] for elem in experiment_dict[method][key]]
			upper_ci_list = [elem[2][0] for elem in experiment_dict[method][key]]
			lower_ci_list = [elem[2][1] for elem in experiment_dict[method][key]]

			linestyle = "-" if key == "average_reward" else "+"
			label = "Reward" if key == "average_reward" else "Value"

			plt.plot(noise_list, mean_list, linestyle, color=color_list[i],label=format_method(method) + " -- " + label)
			plt.fill_between(noise_list, lower_ci_list, upper_ci_list, color=color_list[i], alpha=.1)
			i += 1

	plt.xlabel("Noise (fraction of maximum action)")
	plt.ylabel("Average reward and value")
	env_title = format_title(name)
	plt.title(f"{env_title} Performance")
	plt.legend()
	plt.savefig(f"{loc}/images/{name}__reward_and_value.png")
	plt.show()



	for key in ["average_reward", "average_initial_value"]:
		i=0
		for method in experiment_dict.keys():
			noise_list = [elem[0] for elem in experiment_dict[method][key]]
			mean_list = [elem[1] for elem in experiment_dict[method][key]]
			upper_ci_list = [elem[2][0] for elem in experiment_dict[method][key]]
			lower_ci_list = [elem[2][1] for elem in experiment_dict[method][key]]

			linestyle = "-" if key == "average_reward" else "+"
			label = "Reward" if key == "average_reward" else "Value"

			plt.plot(noise_list, mean_list, linestyle, color=color_list[i],label=format_method(method) + " -- " + label)
			plt.fill_between(noise_list, lower_ci_list, upper_ci_list, color=color_list[i], alpha=.1)
			i += 1

	plt.xlabel("Noise (fraction of maximum action)")
	plt.ylabel("Average reward and value")
	env_title = format_title(name)
	plt.title(f"{env_title} Performance")
	plt.legend()
	plt.savefig(f"{loc}/images/{name}__error.png")
	plt.show()


	# pdb.set_trace()

def bar_plot(experiment_dict, name="Environment"):
	color_list = ["red", "green", "blue", "brown", "pink"]
	methods = [m for m in experiment_dict.keys()]
	ticks = list(range(len(methods)))
	# for key in keylist:
	for key in experiment_dict[methods[0]].keys():
		i=0
		# for method in experiment_dict.keys():
		# 	mean_list = [elem[1] for elem in experiment_dict[method][key]]
		# 	upper_ci_list = [elem[2][0] for elem in experiment_dict[method][key]]
		# 	lower_ci_list = [elem[2][1] for elem in experiment_dict[method][key]]

		mean_list = [experiment_dict[method][key][0][1] for method in methods]
		var_list = [experiment_dict[method][key][0][2][0] - experiment_dict[method][key][0][2][1] for method in methods]

		plt.bar(ticks, mean_list, yerr=var_list)
		# plt.xticks(ticks, experiment_dict.keys())
		plt.xticks(ticks, [format_method(method) for method in experiment_dict.keys()])


		# plt.xlabel("Noise (fraction of maximum action)")
		plt.ylabel(format_name(key))
		# env_title = format_title(name)
		# plt.title(f"{env_title} Performance")
		plt.legend()
		plt.savefig(f"{loc}/images/{name}__{key}.png")
		plt.show()

envs = sys.argv
if len(envs) <= 2: 
	print("Required arguments: file location, environments")
else: 
	loc = envs[1]
	[plot_log(parse_log(env), name=env) for env in envs[2:]]
