from sklearn import base, metrics
import numpy as np
import torch

class ClusterMultitree(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__( self, attn_module, aggregator, cluster_model, classificators, random_state=None):
        super(ClusterMultitree, self).__init__()

       
        if random_state is not None:
            if not isinstance(random_state, int):
                raise TypeError("random_state must be int. Got {}".format(type(random_state)))
        
        self.attn_module = attn_module
        self.aggregator = aggregator
        self.cluster_model = cluster_model
        self.classificators = classificators
        self.random_state = random_state                
        self._clust = None
        self._trees = None
        
    def get_attn(self, X, classifier, aggregator):

        attention = []

        for _, attn in classifier.forward_iter(X):
            attention.append(attn)

        attention = torch.cat(attention, dim=1)
        attention = aggregator(attention).detach().cpu().numpy()
        return attention        
        
    def get_params(self, deep=True):
        return {
            'attn_module': self.attn_module,
            'aggregator': self.aggregator,
            'cluster_model': self.cluster_model, 
            'classificator': self.classificators,
            'random_state': self.random_state
        }
            
    
    def fit(self, X, y=None):
        attn_matrices = self.get_attn(X, self.attn_module, self.aggregator)
        original_examples = X

        clust_labels = self.cluster_model.fit_predict(attn_matrices)
        
        y_pred = np.zeros((X.shape[0],))

        for label in np.unique(clust_labels):
            indices = np.where(clust_labels == label)[0]
            self.classificators[label] = self.classificators[label].fit(original_examples[indices], y[indices])
            y_pred[indices] = self.classificators[label].predict(original_examples[indices])
                    
        return self
    
    def predict(self, X):

        attn_matrices = self.get_attn(X, self.attn_module, self.aggregator)
        original_examples = X

        predictions = np.zeros((original_examples.shape[0],))
        clust_labels = self.cluster_model.predict(attn_matrices)
        
        for label in np.unique(clust_labels):
            indices = np.where(clust_labels == label)[0]
            predictions[indices] = self.classificators[label].predict(original_examples[indices])    
            
        return predictions

    def score(self, X, y):
        preds = self.predict(X)
        score_val = metrics.accuracy_score(y, preds)
        return score_val

