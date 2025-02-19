from pyclassify import utils
class kNN(): 

    def __init__(self, k: int):

        if not isinstance(k, int):
            raise TypeError
        self.k = k
    
    def _get_k_nearest_neighbours(self, X: list[list], y:list , x: list):
        tmp  = sorted(zip(X,y), key=lambda x_: utils.distance(x_[0], x))
        _, y_sorted = zip(*tmp)

        return y_sorted[:self.k]
    
    def __call__( self, data, new_points):
        '''
        data: tuple with X,y
        new_points: point
        '''
        self.predicted_classes = []
        for point in new_points:
            knearestneighbours = self._get_k_nearest_neighbours(data[0], data[1], point)
            majority_vote = utils.majority_vote(knearestneighbours)
            self.predicted_classes.append(majority_vote)

        return self.predicted_classes
    