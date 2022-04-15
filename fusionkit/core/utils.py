"""
Utilities
A collection of general numerical or Python utilities useful across the framework

"""

import numpy as np
import os

def number(x):
    """Check if x is actually a (real) number type (int,float).

    Args:
        `x` (any): the value to be checked for being a number.

    Returns:
        int,float: returns `x` typed as either int or float.
    """
    try:
        return int(x)
    except ValueError:
        return float(x)

def find(val, arr,n=1):
    """Find the n closest values in an array.

    Args:
        `val` (int,float,str): the sought after value.
        `arr` (ndarray,list): the array to search in for val.
        `n` (int, optional): the number of closest values in arr to be returned. Defaults to 1.

    Returns:
        int,list: depending on `n` either an int for the index of the closest found value, or a list of the `n` indexes of the closest found indexes is returned.
    """
    if isinstance(arr,list):
        arr_ = np.array(arr)
    else:
        arr_ = arr
    if n == 1:
        return np.argsort(np.abs(arr_-val))[0]
    else:
        return list(np.argsort(np.abs(arr_-val)))[:n]

def calcz(x,y):
    """Calculate the inverse scale length z.

    Args:
        `x` (ndarray): the radial coordinate vector.
        `y` (ndarray): the value vector.

    Returns:
        ndarray: the inverse scale length.
    """
    z = (1/y)*np.gradient(y,x,edge_order=2)
    return -z

def read_file(path='./',file=None,mode='r'):
    """Read the contents of a file to a list of lines, with automatic path validity checks.

    Args:
        `path` (str): path to the file. Defaults to './'.
        `file` (str): filename. Defaults to None.
        `mode` (str, optional): reading mode of open(). Defaults to 'r', 'rb' also possible.

    Raises:
        `ValueError`: if `path` is not a valid path
        `ValueError`: if `path+file` is not a valid path

    Returns:
        `list`: a list of the lines read from the file
    """
    # check if the provided path exists
    if os.path.isdir(path):
        # check if the provided file name exists in the output path
        if os.path.isfile(path+file):
            # read the file contents into a list of strings of the lines
            with open(path+file,mode) as f:
                lines = f.readlines()
        else:
            raise ValueError('The file {}{} does not exist!'.format(path,file))
    else:
        raise ValueError('{} is not a valid path to a directory!'.format(path))

    return lines

def autotype(value):
    """Automatically types any string value input to bool, int, float or str.
    Useful when reading mixed type values from a text file.

    Args:
        `value` (str): a value that needs to be re-typed

    Returns:
        bool,int,float,str: typed value
    """
    # first check if value is a bool to prevent int(False)=0 or int(True)=1
    if not isinstance(value,bool):
        # then try int
        try:
            value = int(value)
        except:
            # then try float
            try:
                value = float(value)
            # if not float then perhaps a bool string (from Fortran)
            except:
                if value in ['.T.','.t.','.true.','T','t','True','true','y','Y','yes','Yes']:
                    value = True
                elif value in ['.F.','.f.','.false.','F','f','False','false','n','N','no','No']:
                    value = False
                # no other likely candidates, just strip whitespace and return string
                else:
                    value = str(value.strip("'"))
    # return the bool
    else:
        value = bool(value)
    return value

def list_to_array(object):
    """Convert any list in the object to a ndarray.
    Includes recursive check for dict as input to convert any list in the dict to ndarray, assuming fully unconnected dict!

    Args:
        `object` (list,dict): the object containing one or more list that needs to be converted to an array

    Returns:
        ndarray,dict: `object` containing the converted arrays
    """
    if isinstance(object,dict):
        #print('found dict instead of list, rerouting...')
        for key in object.keys():
            object[key] = list_to_array(object[key])
    elif isinstance(object,list):
        # check if any value in the list is a str
        str_check = [isinstance(value,str) for value in object]
        # if not any strings in the list convert to ndarray
        if not any(str_check):
            #print('converting list to array...')
            object = np.array(object)

    return object

def array_to_list(object):
    """Convert any ndarray into a (list of) list(s).
    Includes recursive check for dict as input to convert any ndarray in the dict to list, assuming fully unconnected dict!

    Args:
        object (ndarray, dict): the object containing one or more arrays that needs to be converted to a list

    Returns:
        list,dict: the object containing the converted lists
    """
    if isinstance(object,dict):
        #print('found dict instead of array, rerouting...')
        for key in object.keys():
            object[key] = array_to_list(object[key])
    elif isinstance(object,np.ndarray):
        #print('converting array to list...')
        object = list(object)
    
    return object