from deephaven.TableTools import readCsv
from deephaven import Plot

from deephaven import tableToDataFrame
from deephaven import dataFrameToTable
from sklearn.neighbors import KDTree as kdtree
import pandas as pd
import numpy as np

# Read external data, remove unwanted parts, and split into train/test
creditcard = readCsv("/data/examples/CreditCardFraud/csv/creditcard.csv")
creditcard = creditcard.select("Time", "V4", "V12", "V14", "Amount", "Class")
train_data = creditcard.where("Time >= 43200 && Time < 57600")
test_data = tableToDataFrame(creditcard.where("Time >= 129600 && Time < 144000"))

# Turn the training data into a Pandas DataFrame
data = tableToDataFrame(train_data.select("V4", "V12", "V14")).values

# Get nearest neighbor distances using a K-d tree
tree = kdtree(data)
dists, inds = tree.query(data, k = 2)

# Sort the nearest neighbor distances in ascending order
neighbor_dists = np.sort(dists[:, 1])

x = np.array(range(len(neighbor_dists)))

# Turn our x and y (sorted neighbor distances) into a Deephaven table
nn_dists = pd.DataFrame({"X": x, "Y": neighbor_dists})
nn_dists = dataFrameToTable(nn_dists)

# Plot the last few hundred points so we can see the "elbow"
neighbor_dists = Plot.plot("Nearest neighbor distance", nn_dists.where("X > 30000"), "X", "Y").show()
