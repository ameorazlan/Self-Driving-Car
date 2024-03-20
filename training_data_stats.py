import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('keyboard_data\driving_log.csv')  # Replace 'your_file.csv' with the name of your file

data.columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

steerings = data['steering']

throttles = data['throttle']

reverse = data['reverse']



plt.hist(steerings, bins=20, range=(-1,1))  # Adjust 'bins' as needed for finer/coarser granularity
plt.title('Keyboard Input: Steering Angles')
plt.xlabel('Steering Angle')
plt.ylabel('Occurences')
plt.show()

plt.hist(throttles, bins=20, range=(0,1))  # Adjust 'bins' as needed for finer/coarser granularity
plt.title('Keyboard Input: Throttle')
plt.xlabel('Throttle Amount')
plt.ylabel('Occurences')
plt.show()

plt.hist(reverse, bins=20, range=(0,1))  # Adjust 'bins' as needed for finer/coarser granularity
plt.title('Keyboard Input: Brake')
plt.xlabel('Brake Amount')
plt.ylabel('Occurences')
plt.show()
