import matplotlib.pyplot as plt

# In the log file, we have the following format:
# Iteration 1
#   test model
#   test top 1 acc 0.001  test ownership MSE 0.0192  learning rate 1.0e-02  backbone multiplier 0.0  time 13:28:52
#   time 13:28:55
#   train model
# Iteration 2
#   time 13:29:55
#   train model
# We need to extract metrics from iterations which have them and plot them.


def parse_logs(log_file):
    with open(log_file, 'r') as f:
        log_lines = f.readlines()
    iterations = []
    accs = []
    ownership_mses = []
    learning_rates = []
    backbone_multipliers = []
    ownership_losses = [None]
    policy_losses = [None]
    for l in log_lines:
        line = l.strip()
        if line.startswith('Iteration'):
            cur_it = int(line.split()[1])
        elif line.startswith('test top 1 acc'):
            iterations.append(cur_it)
            metrics = line.split()
            accs.append(float(metrics[4]))
            ownership_mses.append(float(metrics[8]))
            learning_rates.append(float(metrics[11]))
            backbone_multipliers.append(float(metrics[14]))
        elif line.startswith("ownership loss"):
            # other lines with metrics have format:   ownership loss 1.274  policy loss 2.658  test top 1 acc 0.329  test ownership MSE 0.0026  learning rate 1.0e-02  backbone multiplier 0.0  time 13:34:13
            iterations.append(cur_it)
            metrics = line.split()
            accs.append(float(metrics[10]))
            ownership_mses.append(float(metrics[14]))
            learning_rates.append(float(metrics[17]))
            backbone_multipliers.append(float(metrics[20]))
            ownership_losses.append(float(metrics[2]))
            policy_losses.append(float(metrics[5]))
    return iterations, accs, ownership_mses, learning_rates, backbone_multipliers, ownership_losses, policy_losses


if __name__ == '__main__':
    log_files = "log-19x19-frozen", "log-19x19-unfrozen-15k", "log-19x19-barehead"
    iteration_list = []
    acc_list = []
    ownership_mse_list = []
    learning_rate_list = []
    backbone_multiplier_list = []
    ownership_loss_list = []
    policy_loss_list = []
    for log_file in log_files:
        iterations, accs, ownership_mses, learning_rates, backbone_multipliers, ownership_losses, policy_losses = parse_logs(log_file)
        iteration_list.append(iterations)
        acc_list.append(accs)
        ownership_mse_list.append([m * 1000 for m in ownership_mses])
        learning_rate_list.append(learning_rates)
        backbone_multiplier_list.append(backbone_multipliers)
        ownership_loss_list.append(ownership_losses)
        policy_loss_list.append(policy_losses)
    plt.figure()
    # plt.plot(iteration_list[0], ownership_loss_list[0], label='frozen')
    # plt.plot(iteration_list[1], ownership_loss_list[1], label='unfrozen')
    network_names = ['frozen', 'unfrozen', 'no backbone']
    metric_lists = [acc_list, ownership_mse_list, ownership_loss_list, policy_loss_list]
    titles = ['Policy Top-1 Accuracy', 'Local Ownership MSE', 'Ownership L2 Loss', 'Policy Cross-Entropy Loss']
    ylables = ['Accuracy', 'MSE (*1000)', 'Loss', 'Loss']
    for metric_list, title, ylabel in zip(metric_lists, titles, ylables):
        plt.figure()
        for i, metric in enumerate(metric_list):
            plt.plot(iteration_list[i], metric, label=network_names[i])
        plt.legend(fontsize=20)
        plt.title(title, fontsize=30)
        plt.xlabel('Iteration', fontsize=30)
        plt.ylabel(ylabel, fontsize=30)
        # Set font size in ticks to be bigger
        plt.tick_params(axis='both', which='major', labelsize=22)
        # Set the canvas a bit bigger on the bottom and on the left so that the big labels fit
        if title == 'Local Ownership MSE':
            plt.ylim(top=7)
        plt.subplots_adjust(bottom=0.18, left=0.17)
        plt.savefig(title.replace(' ', '_') + '.png')
        plt.show()
    # plt.plot(iteration_list[0], ownership_mse_list[0], label='frozen')
    # plt.plot(iteration_list[1], ownership_mse_list[1], label='unfrozen')
    # plt.plot(iteration_list[2], ownership_mse_list[2], label='no backbone')
    # # Artificially restrict the y-axis to be at most 0.01
    # plt.ylim(top=7)
    # # Set font to be bigger
    # # plt.rcParams.update({'font.size': 30})
    # plt.legend()
    # plt.title('Local ownership MSE', fontsize=30)
    # plt.xlabel('Iteration', fontsize=30)
    # plt.ylabel('MSE (*1000)', fontsize=30)
    # # Set font size in ticks to be bigger
    # plt.tick_params(axis='both', which='major', labelsize=22)
    # # plt.rcParams.update({'font.size': 22})
    # # Set the canvas a bit bigger on the bottom and on the left so that the big labels fit
    # plt.subplots_adjust(bottom=0.18, left=0.18)
    # plt.savefig('ownership_mse.png')
    # plt.show()
