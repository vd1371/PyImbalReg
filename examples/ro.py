# A simple example of oversampling

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


# The default relevance funtion will be used
ro = pir.RandomOversampling(df = data,
                            rel_func = 'default',
                            threshold = 0.7,
                            o_percentage = 5)
new_data = ro.get()


# Plotting the histogram of target values after oversampling
plt.hist(new_data.iloc[:, -1].values)
plt.title('After')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()
