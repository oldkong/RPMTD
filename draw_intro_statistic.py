import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

statisticjta = plt.figure(figsize=(13, 10))

x = range(1999, 2021, 1)
y = [6.3, 6.7, 6.7, 6.9, 6.9, 7.5, 8.1, 8.6, 9.1, 9.3, 9.0, 9.6, 10.0, 10.5, 11.0, 11.5, 12.1, 12.5, 13.4, 14.1, 14.7, 3.9]

plt.plot(x, y, "o-", data=y)

for i in x[0: -2]:
    plt.annotate(y[i-1999], xy=(i-0.4, y[i-1999]+0.2), fontsize=15)

plt.annotate(y[x[-1]-1999], xy=(x[-1], y[x[-1]-1999]+0.2), fontsize=15)
plt.show()


plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(x, ["99", "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"])
plt.xlabel("year", fontsize=15)
plt.ylabel("billion people", fontsize=15)
plt.margins(x=0.05, y=0.1)
plt.show()
pp = PdfPages("statisticjta.pdf")
pp.savefig(statisticjta, bbox_inches = 'tight', pad_inches = 0)
pp.close()