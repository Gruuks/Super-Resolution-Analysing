import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def gauss2(x, a_1, b_1, c_1, a_2, b_2, c_2):
    return a_1 * np.exp(-(x - b_1) ** 2 / (2 * c_1 ** 2)) + a_2 * np.exp(-(x - b_2) ** 2 / (2 * c_2 ** 2))


def auto(dataset_input):
    dataset = str(dataset_input)
    dataset = "data/data " + dataset
    dataset_fitted = dataset + " (fitted).csv"
    dataset_processed = dataset_fitted[:22] + " (processed).png"

    x = np.loadtxt(dataset, skiprows=1)[:, 0] * 1000
    y = np.loadtxt(dataset, skiprows=1)[:, 1]

    plt.figure(dpi=100)
    plt.plot(x, y)
    plt.grid()
    plt.xlabel("Position (nm)")
    plt.ylabel("Counts")
    plt.savefig(dataset + ".png")
    # plt.show()
    plt.close()

    close_peaks = find_peaks(y, height=30)
    close_peaks = close_peaks[0]
    x_close_peak_1, x_close_peak_2 = x[close_peaks[0]], (x[close_peaks[1]])

    p0 = [100, x_close_peak_1, 30, 100, x_close_peak_2, 30]

    popt, pcov = curve_fit(gauss2, x, y, p0, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    xf = np.linspace(0, x[(len(x) - 1)], 10000)
    yf = gauss2(xf, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])

    plt.figure(dpi=100, figsize=(10, 5))
    plt.plot(xf, yf)
    plt.grid()
    plt.xlabel("Position (nm)")
    plt.ylabel("Counts")
    plt.title("Peak$_1$ = " + str(popt[1]) + "+/- " + str(perr[1]) + " (nm)" + "\n" +
              "Peak$_2$ = " + str(popt[4]) + "+/- " + str(perr[4]) + " (nm)")
    # plt.show()
    plt.savefig(dataset + "(fitted + errors).png")
    plt.close()

    data_fit = np.column_stack((xf, yf))
    np.savetxt(dataset_fitted, data_fit)

    dataset_fit = dataset_fitted
    x = np.loadtxt(dataset_fit, skiprows=1)[:, 0]
    y = np.loadtxt(dataset_fit, skiprows=1)[:, 1]

    peaks = find_peaks(y)
    peaks = peaks[0]
    y_peaks = y[peaks]
    x_peaks = x[peaks]
    plt.plot(x_peaks, y_peaks, marker="x", linestyle='none')
    plt.xlabel("Position (nm)")
    plt.ylabel("Counts")
    plt.plot(x, y)
    delta_x = x[peaks[1]] - x[peaks[0]]
    u_delta_x = np.sqrt(perr[1] ** 2 + perr[4] ** 2)
    plt.title("Distance between fitted peaks: " + str(round(delta_x, 2)) + " +/- " + "{:.2f}".format(u_delta_x) + " nm")
    plt.grid()
    savename = dataset_processed
    plt.savefig(savename)
    # plt.show()
    plt.close()
    return delta_x, u_delta_x


datasets_1 = np.array([])  # semi wrong
datasets_1 = 1 + np.arange(1, 10, 1) / 10
datasets_1 = np.append(datasets_1, 1.11)

datasets_2 = np.array([])  # semi wrong
datasets_2 = 2 + np.arange(1, 8, 1) / 10
datasets_2 = np.delete(datasets_2, 2)

datasets_3 = np.array([])
datasets_3 = 3 + np.arange(1, 6, 1) / 10

datasets_4 = np.array([])
datasets_4 = 4 + np.arange(1, 6, 1) / 10

datasets_5 = np.array([])  # very yes
datasets_5 = 5 + np.arange(1, 6, 1) / 10

datasets = np.concatenate((datasets_1, datasets_2, datasets_3, datasets_4, datasets_5))

delta_x_total = np.array([])
u_delta_x_total = np.array([])
j = 0
while j <= (len(datasets) - 1):
    delta_x_calc, u_delta_x_calc = auto(datasets[j])
    delta_x_total = np.append(delta_x_total, delta_x_calc)
    u_delta_x_total = np.append(u_delta_x_total, u_delta_x_calc)
    j += 1

delta_x_avg = np.average(delta_x_total)
delta_x_stddev = np.std(delta_x_total)
print("Average = ", delta_x_avg, " (nm)")
print("Standard Deviation = ", delta_x_stddev, " (nm)")

plt.figure(figsize=(10, 5))
hist, bin_edges = np.histogram(delta_x_total)
bin_edges = np.round(bin_edges, 0)
plt.bar(bin_edges[:-1], hist, width=3, color='#0504aa', alpha=0.7)
plt.xlim(min(bin_edges), max(bin_edges))
plt.grid(axis='y', alpha=0.75)
plt.xlabel("Resolved distance (nm)")
plt.ylabel("Counts")

plt.title("Average Resolved Distance = " + "{:.2f}".format(delta_x_avg) +
          " (nm)" + ", Standard Deviation = " + "{:.2f}".format(delta_x_stddev) + " (nm)")

plt.savefig("data/processed histogram.png")
plt.show()
