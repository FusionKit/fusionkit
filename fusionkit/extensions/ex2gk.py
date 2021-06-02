'''
The EX2GK extension class serves as an object to interact with experimental and fitting data
to add/edit/filter data through the custom filter option in EX2GK

Requires ptools!
'''

import os
import pandas as pd
import numpy as np
import copy

## EX2GK
class EX2GK:
    def __init__(self):
        self.gpr_data = {}
        self.gpr_data['metadata'] = {}
    
    # I/O functions
    def read_file(self, data_loc=None, fname=None, quantities=None):
        gpr_data = self.gpr_data

        empty_lines = []
        table_lines = {}
        line_count = 0
        if data_loc!=None and os.path.isdir(data_loc):
            with open(data_loc+fname, 'r') as file:
                for line in file.readlines():
                    line_count += 1
                    # If the line_count = 1 note the data type
                    if line_count == 1:
                        gpr_data['metadata']['type'] = line.split()[3]
                    # Add the header contents to the metadata section of gpr_data
                    if 'Shot' in line:
                        gpr_data['metadata']['shot'] = str([int(s) for s in line.split() if s.isdigit()][0])
                    elif 'Radial' in line:
                        gpr_data['metadata']['x'] = line.split()[-1]
                    elif 'Time' in line:
                        if 'time' not in gpr_data['metadata']:
                            gpr_data['metadata']['time'] = []
                        gpr_data['metadata']['time'].append(np.round(float(line.split()[-2]),2))
                    # If the line is empty add it to the empty_lines list
                    if not line.strip():
                        empty_lines.append(line_count)
                    # If there have been any empty lines, check if the table header on the next line contains one of the requested quantities 
                    if len(empty_lines)>=1 and (line_count)-1 == empty_lines[-1]:
                        for quantity in quantities:
                            # If the quantity was not already found and the table header contains it, record the line counter
                            if quantity not in table_lines:
                                if gpr_data['metadata']['type'] == 'Processed':
                                    qstring = 'QLK_'+quantity+' Proc.'
                                else:
                                    qstring = quantity+' '+gpr_data['metadata']['type']
                                if qstring in line:
                                    table_lines[quantity] = line_count-1
                                    gpr_data[quantity] = {}
                                    #print(line)
            #print(empty_lines)
            #print(table_lines)

            for quantity in quantities:
                if quantity in table_lines:
                    line = table_lines[quantity]
                    line_index = empty_lines.index(line)
                    df = pd.read_csv(data_loc+fname, delimiter='\\s{2,}', skiprows=line, nrows=(empty_lines[line_index+1]-empty_lines[line_index])-1, engine='python')
                    if gpr_data['metadata']['type'] == 'Raw':
                        df = df.iloc[:,0:5]
                        df.columns = ['x', 'y', 'y_sigma', 'x_err', 'diagnostic']
                        #print(df)
                        for source in list(sorted(set(df['diagnostic']))):
                            if 'BC' not in source:
                                if source not in gpr_data[quantity]:
                                    #print(source)
                                    gpr_data[quantity][source] = []
                                gpr_data[quantity][source] = df[df['diagnostic']==source].sort_values('x').reset_index(drop=True)
                    elif gpr_data['metadata']['type'] == 'Fit':
                        df = df.iloc[:,0:5]
                        df.columns = ['x', 'y', 'y_sigma', 'dydx', 'dydx_err']
                        #print(df)
                        gpr_data[quantity] = df
                    elif gpr_data['metadata']['type'] == 'Processed':
                        df.columns = ['x', 'qlk_x', 'y', 'y_sigma']
                        #print(df)
                        gpr_data[quantity] = df
            #print(gpr_data)
            if 'TIMP' in gpr_data:
                gpr_data['TI'] = gpr_data.pop('TIMP')
            return gpr_data
        
        else:
            print('No valid data location was specified!')
    
    # Filter functions
    def timeavg_filter(input_data=None, quantity_filter=None, source_filter=None):
        '''
        This function returns a dataframe containing the time averaged data for all the sources specified in the source_filter list

        :param input_data: a dict of the form raw_data[quantity][source][timeslice]

        :param source_filter: a list of strings indicating which sources to include in the returned time averaged data
        '''
        filtered_data = {}
        # Check that a data structure is provided to be time averaged
        if input_data != None and isinstance(input_data,dict):
            raw_data = input_data
            # If no quantity_filter list is specified, copy all the sources listed in the data structure
            if quantity_filter == None:
                quantity_filter = list(raw_data.keys())
                #print(quantity_filter)
            # If no source_filter list is specified, copy all the sources listed in the data structure
            if source_filter == None:
                source_filter = []
                for quantity in quantity_filter:
                    source_list = list(raw_data[quantity].keys())
                    for source in source_list:
                        if source not in source_filter:
                            source_filter.append(source)
                #print(source_filter)
                
            # Assuming the data structure has the raw_data[quantity][source][timeslice]= pandas.dataFrame(columns=['x', 'y', 'y_sigma', 'x_err', 'diagnostic']) structure
            for quantity in quantity_filter:
                df_dict = {}
                #print(raw_data)
                # Sequence through all the data source by source
                for source in raw_data[quantity].keys():
                    if source in source_filter:
                        # Specify the minimum number of samples in time to use scaling by number of data points in average input variance, does not affect population variance
                        if len(raw_data[quantity][source]) < 2:
                            use_n = False
                        else:
                            use_n = True

                        source_df = pd.concat(raw_data[quantity][source]).sort_values(by=['x']).reset_index(drop=True)
                        #print(source_df.to_string())
                        source_concat = pd.DataFrame(columns=raw_data[quantity][source][0].columns)

                        x_index = 0
                        x_ref = source_df.iloc[x_index]['x']
                        while x_ref < source_df.iloc[-1]['x']:
                            temp_concat = pd.DataFrame(columns=raw_data[quantity][source][0].columns)
                            #print('first x_ref: '+str(x_ref))
                            if x_ref < 0.01:
                                threshold = 50#5
                            elif x_ref < 0.06:
                                threshold = 25#3.25
                            elif x_ref < 0.1:
                                threshold = 18#2.5
                            elif x_ref < 0.2:
                                threshold = 7#2.5
                            else:
                                if source == "KG10":
                                    if x_ref > 1.02:
                                        threshold = 0.3
                                    elif x_ref > 1.005:
                                        threshold = 0.6
                                    elif x_ref > 0.9:
                                        threshold = 0.825
                                    else:
                                        threshold =3.5#1.5
                                else:
                                    if x_ref < 0.27:
                                        threshold = 1.25
                                    else:
                                        threshold = 1.1
                            while 100*((source_df.iloc[x_index]['x']/x_ref)-1) < threshold and x_index < source_df.shape[0]-1:
                                temp_concat.loc[temp_concat.shape[0]+1] = source_df.iloc[x_index]
                                x_index+=1
                            if temp_concat.shape[0] > 1:
                                temp_y, temp_yerr, temp_ystdm = ptools.calc_eb_error(list(temp_concat['y']),list(temp_concat['y_sigma']),use_n=use_n)
                                temp_concat = temp_concat.mean()
                                temp_concat['y'] = temp_y
                                temp_concat['y_sigma'] = temp_yerr
                                temp_concat['diagnostic'] = source
                            source_concat = source_concat.append(temp_concat,ignore_index=True).reset_index(drop=True)
                            #print(source_concat)
                            x_ref = source_df.iloc[x_index]['x']

                        df_dict[source] = source_concat
                        #if source == 'KG10':
                        #    print(source_concat.to_string())
                    else:
                        print('\t\t'+source+' is not in the source filter list')

                if len(df_dict) > 0:
                    filtered_data[quantity] = pd.concat(df_dict).sort_values('x')
                    filtered_data[quantity].reset_index(drop=True,inplace=True)
                else:
                    print('\t\tNo sources were selected for '+quantity+'!')
                #print(str(quantity)+": \n"+pm_data[quantity].to_string())
                print("\tFiltered custom data for: "+quantity)
            return filtered_data
        else:
            print('No valid input data structure was provided!')

    def copyandpaste_filter(input_data=None, copy_quantity=None, paste_quantity=None, source_filter=None, radial_filter=None):
        output_data = copy.deepcopy(input_data)
        if output_data != None and isinstance(output_data,dict):
            if copy_quantity != None and isinstance(copy_quantity,str) and paste_quantity != None and isinstance(paste_quantity,str):
                # If no source_filter list is specified, copy all the sources listed in the data structure
                if source_filter == None:
                    source_filter = set(output_data[copy_quantity]['diagnostic'])
                    #print(source_filter)
                if radial_filter == None:
                    radial_filter = [0, np.max(pd.Series(output_data[copy_quantity]['x']))]
                    #print(radial_filter)
                if output_data[copy_quantity]['diagnostic'].isin(source_filter).any():
                    #print('Selected diagnostic data is available for copy')
                    if output_data[copy_quantity]['x'].between(radial_filter[0],radial_filter[1]).any():
                        #print('Data is available for copy in the selected radial range')
                        output_data[paste_quantity] = output_data[paste_quantity].append(output_data[copy_quantity][output_data[copy_quantity]['diagnostic'].isin(source_filter) & output_data[copy_quantity]['x'].between(radial_filter[0],radial_filter[1])],ignore_index=True)
                        output_data[paste_quantity].sort_values(by=['x']).reset_index(drop=True,inplace=True)
                        #print(output_data[paste_quantity])

                        return output_data
                    else:
                        print('No data is available for copy in the selected radial range')
                else:
                    print('No data is available for copy for the selected diagnostic')

            else:
                print('Check proper definition of quantities to be copied and pasted from!')
        else:
            print('No valid input data structure was provided!')

    def delete_filter(input_data=None, delete_quantity=None, source_filter=None, radial_filter=None, value_filter=None):
        output_data = copy.deepcopy(input_data)
        if output_data != None and isinstance(output_data,dict):
            if delete_quantity != None and isinstance(delete_quantity,str):
                # If no source_filter list is specified, copy all the sources listed in the data structure
                if source_filter == None:
                    print('No sources were selected to delete data for in the input data structure')
                if radial_filter == None and value_filter == None:
                    print('Both no radial range and value range were selected to delete data for in the input data structure')
                if output_data[delete_quantity]['diagnostic'].isin(source_filter).any():
                    if radial_filter is not None:
                        if output_data[delete_quantity]['x'].between(radial_filter[0],radial_filter[1]).any():
                            output_data[delete_quantity] = output_data[delete_quantity].drop(output_data[delete_quantity][output_data[delete_quantity]['diagnostic'].isin(source_filter) & output_data[delete_quantity]['x'].between(radial_filter[0],radial_filter[1])].index)
                            output_data[delete_quantity].sort_values(by=['x']).reset_index(drop=True,inplace=True)
                            return output_data

                    elif value_filter is not None:
                        if output_data[delete_quantity]['y'].between(value_filter[0],value_filter[1]).any():
                            output_data[delete_quantity] = output_data[delete_quantity].drop(output_data[delete_quantity][output_data[delete_quantity]['diagnostic'].isin(source_filter) & output_data[delete_quantity]['y'].between(value_filter[0],value_filter[1])].index)
                            output_data[delete_quantity].sort_values(by=['x']).reset_index(drop=True,inplace=True)
                            return output_data
                    else:
                        print('No data is available for deletion in the selected radial range')
                else:
                    print('No data is available for deletion for the selected diagnostics for quantity: '+delete_quantity)

            else:
                print('Check proper definition of quantities to be deleted!')
        else:
            print('No valid input data structure was provided!')