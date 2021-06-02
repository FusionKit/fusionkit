'''
The DataSpine class serves as a tree object for storing fusionkit objects 
that are part of a project or as preparation for writing them to disk.
'''

from datetime import datetime

class DataSpine:
    def __init__(self,dataspine=None):
        if dataspine is None:
            self.dataspine = {}

    def create(self,author=None):
        self.author = author
        self.add_metadata()
        return self

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