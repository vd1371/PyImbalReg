# A simple example of using gaussian noise based on histogram frequency
import matplotlib.pyplot as plt
from seaborn import load_dataset
import PyImbalReg as pir


data = load_dataset('dots')

# Plotting the histogram of target values before oversampling
plt.hist(data.iloc[:, -1].values)
plt.title('Before')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

print (len(data), "-----------------")

gnhf = pir.GNHF(df = data,
				perm_amp = 0.01,
				bins = 50,
				should_log_transform = False)

new_data = gnhf.get()


print (len(new_data), "-----------------")
# Plotting the histogram of target values after oversampling
plt.hist(new_data.iloc[:, -1].values)
plt.title('After')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()


