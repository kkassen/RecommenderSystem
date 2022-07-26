# Author: Kyle Arick Kassen

# Import statements
import numpy as np
import pandas as pd
from scipy import stats

class DataRetrieval():
    # Books Data
    # Retrieving data and storing in dataframe
    books = pd.read_csv(r'C:\Users\kykas\Desktop\pyPrograms\book-names.csv', header=None)
    books.rename(columns={0: 'Book Id', 1: 'Book Title'}, inplace=True)

    # Ratings Data
    # Retrieving data and storing in dataframe
    ratings = pd.read_csv(r'C:\Users\kykas\Desktop\pyPrograms\book-ratings-40.csv')
    ratings = ratings.pivot_table(values='rating', index=ratings['user_id'], columns=ratings['book_id'])

    # Filling missing values with zeros
    ratings = ratings.fillna(0)


class RecommenderSystem():

    # Identifies the K nearest neighbors of u_t among all other users...
    # ...by computing the Pearson Correlation Coefficient between u_t...
    # ...and all other users. Ranks them in decreasing order of similarity
    def similarityMeasure(self, dataset, u_t, K):
        # we use training data here; users 0 - 29
        d_correlation = pd.DataFrame(dataset, copy=True)
        correlation = []
        for i in range(len(d_correlation)):
            x0 = d_correlation.iloc[i]
            x1 = []
            y0 = u_t
            y1 = []
            for j in range(len(u_t)):
                # Ensuring items have been rated by both users
                if x0[j] != 0 and y0[j] != 0:
                    x1.append(x0[j])
                    y1.append(y0[j])
            # Must have more than two values and values can't all be the same
            if len(x1) < 2 or np.all(x1 == x1[0]) or np.all(y1 == y1[0]):
                correlation.append(0)
            else:
                # Using scipy built-in method to calculate Pearson Correlation Coefficient
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
                correlation.append(stats.pearsonr(x1, y1)[0])
        # Adding a column for the Pearson Correlation Coefficient to our dataframe
        d_correlation.insert(column='correlation', value=correlation, loc=len(u_t))
        # Ranking in decreasing order of similarity
        d_correlation.sort_values(by='correlation', ascending=False, inplace=True)
        # Removing negative values
        d_correlation = d_correlation[d_correlation['correlation'] >= 0]
        # Returning specified K nearest neighbors
        return d_correlation[0:K]

    # Returns a series of rating predictions for u_t on i_t that is based
    # ...on the results returned from our similarityMeasure() function
    def prediction(self, cor, u_t):
        targetRow = pd.Series(u_t, copy=True)
        books = cor.columns.drop(labels='correlation')
        pSeries = pd.Series(targetRow, copy=True)
        for b in books:
            r_sim = 0
            sim = 0
            for i, j in cor.iterrows():
                if j.loc[b] != 0:
                    # r(NU, i_t) = sum (r(u_i) * sim(NU, u_i) / sum(sim(NU, u_l))
                    r_sim += j.loc[b]*j['correlation']
                    sim += j['correlation']
            if r_sim == 0:
                # if no rating exists, we compute the mean rating on the item across all users
                pSeries.update(pd.Series([training_data[b].sum()/len(training_data[b])], [b]))
            else:
                # r(NU, i_t) = sum (r(u_i) * sim(NU, u_i) / sum(sim(NU, u_l))
                pSeries.update(pd.Series([r_sim/sim], [b]))
        return pSeries

    # Given a user (u_t) and the number of desired recommendations (N)...
    # ...this function generates the top N recommended items for u_t
    def topNrecommendations(self, N, cor, u_t):
        targetRecommendations = pd.Series(u_t, copy=True)
        recommendations = self.prediction(cor, u_t)
        # So we can see the name of the book in the output terminal
        recommendations.rename({0: '0 TRUE BELIEVER', 1: '1 THE DA VINCI CODE', 2: '2 THE WORLD IS FLAT',
                                3: '3 MY LIFE SO FAR', 4: '4 THE TAKING', 5: '5 THE KITE RUNNER', 6: '6 RUNNY BABBIT',
                                7: '7 HARRY POTTER'}, inplace=True)
        for i, j in targetRecommendations.iteritems():
            # Identify and eliminate items that have already been rated
            if j != 0:
                recommendations[i] = np.nan
        recommendations.dropna(inplace=True)
        if N < len(recommendations):
            stop = N
        else:
            stop = len(recommendations)
        # Returns top N recommended items; ranked in decreasing order
        return recommendations[0:stop].sort_values(ascending=False)


# The Interface: a simple text interface for interacting with the...
# ...Simple Recommender System using K-Nearest-Neighbor Collaborative Filtering
while True:
    userInput = input('Menu Options:\n'
                      '1) Display Top N Recommendations\n'
                      '*) Exit\n'
                      'Please type the corresponding menu option number: ')

    # Creating a Recommender System Object
    R = RecommenderSystem()

    # Creating a Data Retrieval Object
    data = DataRetrieval()

    # Partitioning the training and testing data
    training_data = data.ratings.iloc[data.ratings.index <= 29]
    testing_data = data.ratings.iloc[data.ratings.index >= 30]

    # Customizing the decimal precision of data tables displayed in output terminal
    pd.set_option("display.precision", 16)

    if userInput == '1':
        print()
        user = input('Please type a value for the user id: ')
        user = int(user)
        K = input('Please type a value for K: ')
        K = int(K)
        N = input('Please type a value for N: ')
        N = int(N)
        u_t = data.ratings.iloc[user - 1]
        z = R.similarityMeasure(training_data, u_t, K=K)
        print()
        print('Top ' + str(N) + ' Recommendations: ' + 'User=' + str(user) + ', N=' + str(N) + ', K=' + str(K) + ' -->')
        print(R.topNrecommendations(N, cor=z, u_t=u_t).to_string(header=None))
        print()

    # Exit the program gracefully
    elif userInput == '*':
        quit()
