'''
The DataSpine class serves as a tree object for storing fusionkit objects 
that are part of a project or as preparation for writing them to disk.
'''

# general imports
import os
import json
import copy
import codecs
# import methods from
from datetime import datetime
# framework imports
from ..core.utils import *

class DataSpine:
    def __init__(self,tree=False):
        self.dataspine = {}
        author = self.set_author()
        date_created = self.set_timestamp()
        framework_version = self.set_version()
        self.metadata = {'author':author,'date_created':date_created,'framework_version':framework_version}
        if tree:
            self.dataspine = {}

    def set_author(self):
        if os.environ.get('USER'):
            author = os.environ.get('USER')
        elif os.environ.get('USERNAME'):
            author = os.environ.get('USERNAME')
        else:
            author = 'unknown'
        
        return author
    
    def set_timestamp(self):
        today = datetime.now()
        timestamp = "{}/{}/{} @ {}:{}".format(today.day,today.month,today.year,today.strftime("%H"),today.strftime("%M"))

        return timestamp
    
    def set_version(self):
        version = 'v1.2022.0'

        return version

    def add_metadata(self,projectname=None):
        if 'metadata' not in self.dataspine:
            self.dataspine['metadata'] = {}
        today = datetime.now()
        created = "{}/{}/{} @ {}:{}".format(today.day,today.month,today.year,today.strftime("%H"),today.strftime("%M"))
        self.dataspine['metadata'].update({'created':created})
        if self.author is not None:
            self.dataspine['metadata'].update({'author':self.author})
        else:
            self.dataspine['metadata'].update({'author':'Unknown'})

        return self
    
    def read_json(self,path=None,f_name=None,tree=None,verbose=False):
        # check for JSON extension, append if needed
        if f_name.strip().split('.')[-1] not in ['json','JSON']:
            f_name += '.json'
        # read the JSON tree
        with open(path+f_name,'r') as file:
            tree = json.load(file)
        # get the tree name
        tree_name = list(tree.keys())[0]
        # print if verbose
        if verbose:
            print("Reading fusionkit.{}".format(tree_name))
        public_self = copy.deepcopy(self.__dict__)
        for key in self.__dict__:
            if key not in public_attributes(self):
                del public_self[key]
        for key in tree[tree_name].keys():
            if key in public_self:
                self.__dict__[key] = copy.deepcopy(list_to_array(tree[tree_name][key]))

        '''var_list = public_attributes(self)
        if not tree:
            tree = {}
        tree_name = str(self.__class__.__name__)
        if tree_name not in tree:
            tree[tree_name] = {}

        for key in var_list:
            value = array_to_list(vars(self)[key])
            tree[tree_name].update({key:value})'''

        return self

    def write_json(self,path=None,f_name=None,tree=None,verbose=False):
        var_list = public_attributes(self)
        if not tree:
            tree = {}
        tree_name = str(self.__class__.__name__)
        if tree_name not in tree:
            tree[tree_name] = {}

        for key in var_list:
            value = array_to_list(vars(self)[key])
            tree[tree_name].update({key:value})

        if f_name.strip().split('.')[-1] not in ['json','JSON']:
            f_name += '.json'
        
        json.dump(tree, codecs.open(path+f_name, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)

        if verbose:
            print('Generated fusionkit.{} file at: {}'.format(tree_name,path+f_name))

        return self

    def print_tree(self,tree):
        preamb = ''
        for key in tree.keys():
            if isinstance(tree[key],dict):
                print('{}{}:{}'.format(preamb,key,type(tree[key])))
                preamb = ''
                self.print_tree(tree[key])
            else:
                preamb= '\t'
                #print(type(tree[key]))
                if isinstance(tree[key],list) and all([isinstance(value,float) for value in tree[key]]):
                    nan = np.isnan(np.array(tree[key])).any()
                else:
                    nan = False
                print('{}{}:{}'.format(preamb,key,type(tree[key])))
                
    def remove_nan_tree(self,tree):
        _tree = copy.deepcopy(tree)
        for key in _tree.keys():
            if isinstance(tree[key],dict):
                self.remove_nan_tree(tree[key])
            else:
                if isinstance(tree[key],list) and all([isinstance(value,float) for value in tree[key]]):
                    nan = np.isnan(np.array(tree[key])).any()
                    if nan:
                        del tree[key]
        return
    
    def create(self,author=None):
        self.author = author
        self.add_metadata()
        return self