from icecream import ic
import yaml
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})  # Sets the global font size to 12

x_query_predicted_errors = []
y_query_predicted_errors = []
z_query_predicted_errors = []
yaw_query_predicted_errors = []
pitch_query_predicted_errors = []
roll_query_predicted_errors = []

x_query_guess_errors = []
y_query_guess_errors = []
z_query_guess_errors = []
yaw_query_guess_errors = []
pitch_query_guess_errors = []
roll_query_guess_errors = []

x_query_predicted_mae = []
y_query_predicted_mae = []
z_query_predicted_mae = []
yaw_query_predicted_mae = []
pitch_query_predicted_mae = []
roll_query_predicted_mae = []

x_guess_predicted_mae = []
y_guess_predicted_mae = []
z_guess_predicted_mae = []
yaw_guess_predicted_mae = []
pitch_guess_predicted_mae = []
roll_guess_predicted_mae = []

x_guess_query_mae = []
y_guess_query_mae = []
z_guess_query_mae = []
yaw_guess_query_mae = []
pitch_guess_query_mae = []
roll_guess_query_mae = []

x_query = []
y_query = []
z_query = []
yaw_query = []
pitch_query = []
roll_query = []

x_pred = []
y_pred = []
z_pred = []
yaw_pred = []
pitch_pred = []
roll_pred = []

execution_times = []

dir = 'urban_localization/results_field_exp/depth_any'

#list of yaml files in results directory
files = os.listdir(dir)
for file in files:
    path = os.path.join(dir, file)
    with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            config['query_position'][0] = 6600
            config['query_position'][1] = 4761
            config['query_position'][2] = 61
            config['query_orientation'][0] = 140
            config['query_orientation'][1] = -25
            config['query_orientation'][2] = 0

            #position: [6600.        , 4761.        ,   61]
            #orientation: [140, -25, 0]

            x_query.append(config['query_position'][0])
            y_query.append(config['query_position'][1])
            z_query.append(config['query_position'][2])
            yaw_query.append(config['query_orientation'][0])
            pitch_query.append(config['query_orientation'][1])
            roll_query.append(config['query_orientation'][2])

            x_pred.append(config['predicted_position'][0])
            y_pred.append(config['predicted_position'][1])
            z_pred.append(config['predicted_position'][2])
            yaw_pred.append(config['predicted_orientation'][0])
            pitch_pred.append(config['predicted_orientation'][1])
            roll_pred.append(config['predicted_orientation'][2])

            execution_times.append(config['execution_time'])
            
            # store errors query predicted (not absolute)
            x_query_predicted_errors.append(config['predicted_position'][0] - config['query_position'][0])
            y_query_predicted_errors.append(config['predicted_position'][1] - config['query_position'][1])
            z_query_predicted_errors.append(config['predicted_position'][2] - config['query_position'][2])
            yaw_query_predicted_errors.append(config['predicted_orientation'][0] - config['query_orientation'][0])
            pitch_query_predicted_errors.append(config['predicted_orientation'][1] - config['query_orientation'][1])
            roll_query_predicted_errors.append(config['predicted_orientation'][2] - config['query_orientation'][2])

            # store errors query guessed (not absolute)
            x_query_guess_errors.append(config['guessed_position'][0] - config['query_position'][0])
            y_query_guess_errors.append(config['guessed_position'][1] - config['query_position'][1])
            z_query_guess_errors.append(config['guessed_position'][2] - config['query_position'][2])
            yaw_query_guess_errors.append(config['guessed_orientation'][0] - config['query_orientation'][0])
            pitch_query_guess_errors.append(config['guessed_orientation'][1] - config['query_orientation'][1])
            roll_query_guess_errors.append(config['guessed_orientation'][2] - config['query_orientation'][2])


            #store absolute errors query predicted
            x_query_predicted_mae.append(abs(config['predicted_position'][0] - config['query_position'][0]))
            y_query_predicted_mae.append(abs(config['predicted_position'][1] - config['query_position'][1]))
            z_query_predicted_mae.append(abs(config['predicted_position'][2] - config['query_position'][2]))

            yaw_query_predicted_mae.append(abs(config['predicted_orientation'][0] - config['query_orientation'][0]))
            pitch_query_predicted_mae.append(abs(config['predicted_orientation'][1] - config['query_orientation'][1]))
            roll_query_predicted_mae.append(abs(config['predicted_orientation'][2] - config['query_orientation'][2]))

            # store absolute errors guessed predicted
            x_guess_predicted_mae.append(abs(config['predicted_position'][0] - config['guessed_position'][0]))
            y_guess_predicted_mae.append(abs(config['predicted_position'][1] - config['guessed_position'][1]))
            z_guess_predicted_mae.append(abs(config['predicted_position'][2] - config['guessed_position'][2]))
            
            yaw_guess_predicted_mae.append(abs(config['predicted_orientation'][0] - config['guessed_orientation'][0]))
            pitch_guess_predicted_mae.append(abs(config['predicted_orientation'][1] - config['guessed_orientation'][1]))
            roll_guess_predicted_mae.append(abs(config['predicted_orientation'][2] - config['guessed_orientation'][2]))

            # store absolute errors guessed query
            x_guess_query_mae.append(abs(config['guessed_position'][0] - config['query_position'][0]))
            y_guess_query_mae.append(abs(config['guessed_position'][1] - config['query_position'][1]))
            z_guess_query_mae.append(abs(config['guessed_position'][2] - config['query_position'][2]))

            yaw_guess_query_mae.append(abs(config['guessed_orientation'][0] - config['query_orientation'][0]))
            pitch_guess_query_mae.append(abs(config['guessed_orientation'][1] - config['query_orientation'][1]))
            roll_guess_query_mae.append(abs(config['guessed_orientation'][2] - config['query_orientation'][2]))
            
print("Mean absolute error query prediction")
ic(np.average(x_query_predicted_mae))
ic(np.average(y_query_predicted_mae))
ic(np.average(z_query_predicted_mae))
ic(np.average(yaw_query_predicted_mae))
ic(np.average(pitch_query_predicted_mae))
ic(np.average(roll_query_predicted_mae))

print("Standard deviation query prediction")
ic(np.std(x_query_predicted_errors))
ic(np.std(y_query_predicted_errors))
ic(np.std(z_query_predicted_errors))
ic(np.std(yaw_query_predicted_errors))
ic(np.std(pitch_query_predicted_errors))
ic(np.std(roll_query_predicted_errors))

print("Mean absolute error guess query")
ic(np.average(x_guess_query_mae))
ic(np.average(y_guess_query_mae))
ic(np.average(z_guess_query_mae))
ic(np.average(yaw_guess_query_mae))
ic(np.average(pitch_guess_query_mae))
ic(np.average(roll_guess_query_mae))

print('Standard deviation query guess')
ic(np.std(x_query_guess_errors))
ic(np.std(y_query_guess_errors))
ic(np.std(z_query_guess_errors))
ic(np.std(yaw_query_guess_errors))
ic(np.std(pitch_query_guess_errors))
ic(np.std(roll_query_guess_errors))

print('mean execution time')
ic(np.average(execution_times))

# compute mse
x_mse =np.square(np.subtract(x_query,x_pred)).mean()
y_mse =np.square(np.subtract(y_query,y_pred)).mean()
z_mse =np.square(np.subtract(z_query,z_pred)).mean()
yaw_mse =np.square(np.subtract(yaw_query,yaw_pred)).mean()
pitch_mse =np.square(np.subtract(pitch_query,pitch_pred)).mean()
roll_mse =np.square(np.subtract(roll_query,roll_pred)).mean()


errors = {
    'x': x_query_predicted_errors,  # X-coordinate errors
    'y': y_query_predicted_errors,  # Y-coordinate errors
    'z': z_query_predicted_errors,  # Z-coordinate errors
    r'$\psi$': yaw_query_predicted_errors,  # Yaw errors
    r'$\theta$': pitch_query_predicted_errors,  # Pitch errors
    r'$\phi$': roll_query_predicted_errors  # Roll errors
}

# Splitting the errors dictionary into two for plotting
position_errors = {k: errors[k] for k in ['x', 'y', 'z']}
rotation_errors = {k: errors[k] for k in [r'$\psi$', r'$\theta$', r'$\phi$']}

# Creating subplots side by side
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # figsize is adjustable
# set title
#fig.suptitle("Experiment 6")
# Plotting position errors
bplot1 = axs[0].boxplot(position_errors.values(), labels=position_errors.keys(), patch_artist=True, showmeans=True)
axs[0].set_title('Position Errors')
axs[0].set_ylabel('Observed values [m]')
axs[0].yaxis.grid(True)
axs[0].set_ylim(bottom=-60, top=60)
#axs[0].set_ylim(bottom=-200, top=200)


# Plotting rotation errors
bplot2 = axs[1].boxplot(rotation_errors.values(), labels=rotation_errors.keys(), patch_artist=True, showmeans=True)
axs[1].set_title('Rotation Errors')
axs[1].set_ylabel('Observed values [degrees]')
axs[1].yaxis.grid(True)
axs[1].set_ylim(bottom=-70, top=70)
axs[1].set_ylim(bottom=-80, top=120)

# Filling both plots with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.tight_layout()  # Adjust layout to not overlap
plt.show()


'''

fig, ax = plt.subplots()
bplot = ax.boxplot(errors.values(), labels=errors.keys(), patch_artist=True, showmeans=True)
ax.set_title(dir)
ax.set_ylabel('Error')
#plt.xticks(rotation=45)

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# adding horizontal grid lines

ax.yaxis.grid(True)
ax.set_ylabel('Observed values')
ax.set_ylim(bottom=0, top=120)

plt.show()
'''
'''

execution_time = {
    'Execution time': execution_times
}

plt.plot(execution_times, np.zeros_like(execution_times), 'x')
plt.title(dir)
plt.show()


############################################################
# only keep execution times under 100
execution_times_clipped = [x for x in execution_times if x < 100]
# avg
ic(np.average(execution_times_clipped))
'''