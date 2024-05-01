import pandas as pd
import pickle
from surprise import SVD
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate

# IMPORT DATA
rating = pd.read_csv('./training.txt', names = ['u', 'i', 'r'], sep = ',', header = None)
reader = Reader(rating_scale = (0, 5))
training = Dataset.load_from_df(rating, reader)
#data = Dataset.load_builtin(rating, reader)

# CROSS VALIDATION 
# CHECK VALUE OF RMSE
clf = SVD(n_epochs = 5, n_factors = 10, verbose = True)
cross_validate(clf, training, measures = ['RMSE'], cv=10, n_jobs = -1, verbose = True)

training = training.build_full_trainset()
clf.fit(training)