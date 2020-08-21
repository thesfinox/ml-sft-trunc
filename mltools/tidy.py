import pandas as pd
import numpy as np
import sys

class TidySet:
    '''
    Produce a tidy dataset, ready for the analysis.
    '''
    
    def __init__(self, path, format='csv', **kwargs):
        '''
        Arguments:
        
            path:   the path to the dataset,
            format: the format of the dataset,
            kwargs: additional arguments passed to Pandas.
        '''
        if type(path) is not str:
            sys.stderr.write('Path should be a string.')
        if format not in ['csv', 'json']:
            sys.stderr.write('Unknown format.')
            
        if format == 'csv':
            self.__df__ = pd.read_csv(path, **kwargs)
        if format == 'json':
            self.__df__ = pd.read_json(path, **kwargs)
            
    def get_dataframe(self):
        '''
        Get the dataframe.
        
        Returns:
            
            the Pandas dataframe.
        '''
        return self.__df__
    
    def colrename(self, func):
        '''
        Rename the columns according to a function or map.
        
        Arguments:
        
            func: the function or dictionary mapping old to new.
        '''
        self.__df__ = self.__df__.rename(columns=func)
        
    def addlabel(self, name='label'):
        '''
        Add a column labelling each entry of a vector by an integer.
        
        Arguments:
        
            name: the name of the new column to add.
        '''
        if type(name) is not str:
            sys.stderr.write('The name of the column must be a string.')
            name='label'
            
        # compute the shape
        shapes = self.__df__.applymap(len)
        shapes = shapes.apply(lambda x: np.max(x), axis=1)
        cols   = self.__df__.columns.tolist()

        # add the label
        label = []
        for n in range(shapes.shape[0]):
            label.append(np.full(shapes.iloc[n], n, dtype=np.int32))
        self.__df__[name] = label
        
        # reorder the labels
        cols = [name] + cols
        self.__df__ = self.__df__[cols]
        
    def rowexplode(self):
        '''
        Stack each vector entry on top of each other.
        '''
        self.__df__ = pd.concat([pd.DataFrame({f: self.__df__[f].iloc[n] for f in self.__df__})
                                 for n in range(self.__df__.shape[0]) 
                                ],
                                axis=0,
                                ignore_index=True
                               )
        
    def dupremove(self):
        '''
        Remove the duplicates.
        
        Return:
        
            a Pandas dataframe containing the duplicates.
        '''
        duplicate_id = self.__df__.duplicated()
        
        # drop the duplicates
        duplicates  = self.__df__.loc[duplicate_id]
        self.__df__ = self.__df__.loc[~duplicate_id]
        
        return duplicates
    
    def coldrop(self, columns):
        '''
        Drop columns from the dataset.
        
        Arguments:
        
            columns: the columns to drop.
        '''
        if type(columns) is str:
            columns = [columns]
            
        self.__df__ = self.__df__.drop(columns=columns)
    
    def save(self, path, format='csv', **kwargs):
        '''
        Save the dataset to file.
        
        Arguments:
        
            path:   the path to the saved file,
            format: the format of the new file,
            kwargs: additional arguments passed to Pandas.
        '''
        if type(path) is not str:
            sys.stderr.write('Path should be a string.')
        if format not in ['csv', 'json']:
            sys.stderr.write('Unknown format.')
            
        if format == 'csv':
            self.__df__.to_csv(path, **kwargs)
        if format == 'json':
            self.__df__.to_json(path, **kwargs)