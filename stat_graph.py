from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
zscore = lambda x: (x-x.mean())/x.std()

class regviz(object):
    """
    Parameters
    -----------
    sm_results: statsmodels linear regression results object
    
    Returns
    -------
    None
    """
        
    def __init__(self,sm_results):
        # Check that sm_results is a fitted stats models regression object
        if not str(type(sm_results)) == "<class 'statsmodels.regression.linear_model.RegressionResultsWrapper'>":
            raise ValueError('Must Pass the results from statsmodels.OLS.fit')
        
        # Assign class variables
        self.sm_results = sm_results
        self.multipliers = None
        self.fitted = False
        
        # Build dataframe with data from statsmodels object, set fitted as true
        res = self.fit()
        if res == 0:
            self.fitted = True
        
        return None
        
    def set_multipliers(self,multipliers):
        """
        Set's units that feature should be featured in. Multiplies each feature by it's multiplier.
        Parameters
        ----------
        multipliers: dictionary of feature_name : multiplier
        
        Returns
        -------
        None
        """
        
        self._check_fitted()
        
        # Check that multipliers is a dict
        if not type(multipliers) == dict:
            raise ValueError("multipliers must be a dict of feature_name:multiplier")
        
        
        # Check that all keys in multipliers dict are features in the model
        keys = set(multipliers.keys())
        features = set(self.feature_data.index)
        
        # Take set diff to check for multipliers not in dict keys
        keys_not_in_features = list(keys.difference(features))

        key_len = len(keys_not_in_features)
        
        # if there are differences, raise exception with keys not found in feature list
        if key_len > 0:
            if key_len == 1:
                key_str = "'"+keys_not_in_features[0]+ "' is not in the feature set. Please check your multipliers keys and try again"
            else:
                for i,key in enumerate(keys_not_in_features,1):
                    if i == 1:
                        key_str = "'"+key+"', "
                    elif i < key_len:
                        key_str += "'"+key+"', "
                    else:
                        key_str += "and '"+key+"' "
                key_str += 'are not in the feature set. Please check your multipliers keys and try again'
            raise Exception(key_str)
        
        
        # TO DO
        ## Check if selected multipliers have already been set, if yes, reset to original and then update self multiplier dict
        
            
        # set regvizs multipliers dict from multipliers and update feature_data
        self.multipliers = multipliers
        
        if not self.multipliers == None:
                for key,val in zip(self.multipliers.keys(),self.multipliers.values()):
                    self.feature_data.loc[key] = self.feature_data.loc[key]*val
        
        self.multipliers_set = True
        
        return None
    
    def undo_multipliers(self):
        """
        Set's feature units back to original units
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        self._check_fitted()
        
        if self.multipliers_set == True:
            for key,val in zip(self.multipliers.keys(),self.multipliers.values()):
                    self.feature_data.loc[key] = self.feature_data.loc[key]/val
                    
        return None
        
    def fit(self):
        '''
        Fit statsmodels results object to a pandas dataframe of features with their
        confidence intervals and p-values
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        try:
            feature_data = pd.DataFrame(self.sm_results.conf_int(alpha=0.05))
            feature_data99 = pd.DataFrame(self.sm_results.conf_int(alpha=0.01))
            feature_data99.columns=['l_ci_99','h_ci_99']
            feature_data.columns=['l_ci_95','h_ci_95']
            feature_data['coefs'] = self.sm_results.params
            feature_data = feature_data.join(feature_data99)
            feature_data['pval'] = self.sm_results.pvalues
            

            self.feature_data = feature_data
            return 0
        except:
            raise Exception('Error building data Frame')
            
    
    def sort_features(self,sort_by='coefs',ascending=True):
        """
        Sort the features in dataframe and plot by various statistics
        
        Paramaters
        ----------
        sort_by: string containing either 'coefs' to sort by coeffcients or 'pval' to sort by p values. 'coefs' by default.
        
        ascending: boolean specifying if the sort should be ascending if true or descending if false. True by default.
        
        Returns
        -------
        None
        
        """
        
        if not sort_by in ('coefs','pval'):
            raise ValueError("sort_by must be either 'coefs' (sort by coefficient) or 'pval' (sort by pvalue)")
            
        if not ascending in (True, False):
            raise ValueError('ascending must be either a True or False Value')
        
        self._check_fitted()
            
        self.feature_data.sort_values(sort_by,ascending=ascending,inplace=True)
        return None
    
    def _check_fitted(self):
        """
        Checks if the dataframe that feeds the plot has been fitted
        """
        if self.fitted == False:
            raise Exception('Please fit the vizualization using regiz.fit()')
        return None
    
    def set_feature_names(self, ftr_names):
        """
        Updates the names of the features to names set by user
        
        Parameters
        ----------
        ftr_names: dictionary of old features names as keys mapped with the new keys to use
        
        Returns
        -------
        None
        
        """
        self._check_fitted()
        
        if not type(ftr_names) == dict:
            raise ValueError("feature_names must be a dict of old_feature_name:new_feature_name")
        
        #Create list of new feature names (include old feature names if we don't wnat to update them)
        new_feature_name_index = []
        for idx in self.feature_data.index:
            if idx in ftr_names.keys():
                new_feature_name_index.append(ftr_names[idx])
            else:
                new_feature_name_index.append(idx)
            
        #Set the new index to the dataframe
        self.feature_data.index = new_feature_name_index
        
        # Check to see if multipliers exist. If they do, update the names of those we changed
        if not self.multipliers == None:
            for key in ftr_names.keys():
                if key in self.multipliers.keys():
                    self.multipliers[ftr_names[key]] = self.multipliers.pop(key)
                    
        return None
    
    def plot(self,ax=None,hide_features=None):
        '''
        Plotting the regression vizualization 
        Parameters
        ----------
        ax: the matplotlib axis that will hold this image
        
        hide_features: features that should not be visualized on the plot
        
        Returns
        --------
        
        '''
        
        ## Check if the data frame has been fit with the results file
        if not hasattr(self, 'feature_data'):
            raise Exception('Please fit the object with regviz.fit() before plotting it')
        
        if not ax == None:
            
            if not type(ax) == matplotlib.axes._axes.Axes:
                raise Exception('AX Must be an axes object.')
            
        def pval_color(x):
            if x > .05:
                return 'blue'
            elif (x <=.05) and (x >.01):
                return 'orange'
            elif (x <=.01):
                return 'red'
        
        feature_data_plot = self.feature_data.copy()

        if not hide_features == None:
            feature_data_plot.drop(hide_features,inplace = True)
    
        ## Check if an axes was passed, if not create a figure with an axes else use what was passed
        if ax == None:
            fig=plt.figure()
            ax = fig.add_axes([0,0,1,1])
        ax.grid(axis='y',linewidth=.33)
        ax.scatter(y=feature_data_plot.index,x=feature_data_plot['coefs'],c=list(map(pval_color,feature_data_plot['pval'])))
        ax.scatter(y=feature_data_plot.index,x=feature_data_plot['l_ci_99'],marker='|',s=100,c='r')
        ax.scatter(y=feature_data_plot.index,x=feature_data_plot['h_ci_99'],marker='|',s=100,c='r')
        ax.scatter(y=feature_data_plot.index,x=feature_data_plot['l_ci_95'],marker='|',s=25,c='orange')
        ax.scatter(y=feature_data_plot.index,x=feature_data_plot['h_ci_95'],marker='|',s=25,c='orange')
        ax.set_xlabel('Expected Return From Feature',size=14)
        ax.set_title('Range of Returns By Feature',size=16)
        return ax
    
    def get_feature_data(self):
        return self.feature_data


