# Utilities
# general numerical or pythonics utilities

import re
import copy
import numpy as np
from pathlib import Path
from numpy.lib import gradient

# Common numerical data types, for ease of type-checking
np_itypes = (np.int8, np.int16, np.int32, np.int64)
np_utypes = (np.uint8, np.uint16, np.uint32, np.uint64)
np_ftypes = (np.float16, np.float32, np.float64)

number_types = (float, int, np_itypes, np_utypes, np_ftypes)
array_types = (list, tuple, np.ndarray)

def number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def find(val, arr,n=1):
    if isinstance(arr,list):
        arr_ = np.array(arr)
    else:
        arr_ = arr
    if n == 1:
        return np.argsort(np.abs(arr_-val))[0]
    else:
        return list(np.argsort(np.abs(arr_-val)))[:n]

def calcz(x,y):
    z = (1/y)*np.gradient(y,x,edge_order=2)
    return -z

def read_eqdsk(filepath):

    # specify the eqdsk file format, based on 'G EQDSK FORMAT - L Lao 2/7/97'
    eqdsk_format = {
        0:{'vars':['code','case','idum','nw','nh'],'size':[5]},
        1:{'vars':['rdim', 'zdim', 'rcentr', 'rleft', 'zmid'],'size':[5]},
        2:{'vars':['rmaxis', 'zmaxis', 'simag', 'sibry', 'bcentr'],'size':[5]},
        3:{'vars':['current', 'simag2', 'xdum', 'rmaxis2', 'xdum'],'size':[5]},
        4:{'vars':['zmaxis2', 'xdum', 'sibry2', 'xdum', 'xdum'],'size':[5]},
        5:{'vars':['fpol'],'size':['nw']},
        6:{'vars':['pres'],'size':['nw']},
        7:{'vars':['ffprim'],'size':['nw']},
        8:{'vars':['pprime'],'size':['nw']},
        9:{'vars':['psirz'],'size':['nw','nh']},
        10:{'vars':['qpsi'],'size':['nw']},
        11:{'vars':['nbbbs','limitr'],'size':[2]},
        12:{'vars':['rbbbs','zbbbs'],'size':['nbbbs']},
        13:{'vars':['rlim','zlim'],'size':['limitr']},
    }
    max_values = 5 # maximum number of values per line

    eqdata = {}  # Empty container for eqdsk data

    # check if eqdsk file path is provided and if it exists
    eqpath = Path(filepath) if isinstance(filepath, str) else None
    if eqpath is not None and eqpath.is_file():
        
        # read the g-file
        with open(str(eqpath),'r') as file:
            lines = file.readlines()
        
        # convert the line strings in the values list to lists of numerical values, while retaining potential character strings at the start of the file
        for i,line in enumerate(lines):
            # split the line string into separate values by ' ' as delimiter, adding a space before a minus sign if it is the delimiter
            values = list(filter(None,re.sub(r'(?<![Ee])-',' -',line).rstrip('\n').split(' ')))
            #print('values: '+str(values))
            # select all the numerical values in the list of sub-strings of the current line, but keep them as strings so the fortran formatting remains
            numbers = [j for i in [number for number in (re.findall(r'^(?![A-Z]).*-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', value) for value in values)] for j in i]
            #print('numbers: '+str(numbers))
            # select all the remaining sub-strings and store them in a separate list
            strings = [value for value in values if value not in numbers]
            # if there is a list of strings, this means it is the first line of the eqdsk file, so split it in the code and case strings
            if len(strings) > 0:
                strings = [strings[0],' '.join([string for string in strings[1:]])]
            #print('strings: '+str(strings))
            # handle the exception of the first line where in the case description numbers and strings can be mixed
            if i == 0:
                numbers = numbers[-3:]
                case = [string for string in list(line.rstrip('\n').split())[1:] if string not in numbers] 
                strings = [strings[0],' '.join(case)]
            # convert the list of numerical sub-strings to their actual int or float value
            numbers = [number(value) for value in numbers]
            #print('numbers: '+str(numbers))
            lines[i] = strings+numbers
            #print('line after: '+str(lines[i]))

        # start at the top of the file
        current_row = 0
        # go through the eqdsk format line by line and collect all the values for the vars in each format line
        for key in eqdsk_format:
            if current_row < len(lines):
                # check if the var size is a string refering to a value to be read from the eqdsk file and backfill it, for loop for multidimensional vars
                for i,size in enumerate(eqdsk_format[key]['size']):
                    if isinstance(size,str):
                        eqdsk_format[key]['size'][i] = eqdata[size]

                # compute the row the current eqdsk format line ends
                if len(eqdsk_format[key]['vars']) != np.prod(eqdsk_format[key]['size']):
                    end_row = current_row + int(np.ceil(len(eqdsk_format[key]['vars'])*np.prod(eqdsk_format[key]['size'])/max_values))
                else:
                    end_row = current_row + int(np.ceil(np.prod(eqdsk_format[key]['size'])/max_values))

                # check if there are values to be collected
                if end_row > current_row:
                    # collect all the values between current_row and end_row in the eqdsk file and flatten the resulting list of lists to a list
                    values = [j for i in lines[current_row:end_row] for j in i]
                    # handle the exception of len(eqdsk_format[key]['vars']) > 1 and the data being stored in value pairs 
                    if len(eqdsk_format[key]['vars']) > 1 and len(eqdsk_format[key]['vars']) != eqdsk_format[key]['size'][0]:
                        # make a shadow copy of values
                        values_ = copy.deepcopy(values)
                        # empty the values list
                        values = []
                        # collect all the values belonging to the n-th variable in the format list and remove them from the shadow value list until empty
                        for j in range(len(eqdsk_format[key]['vars']),0,-1):
                            values.append(np.array(values_[0::j]))
                            values_ = [value for value in values_ if value not in values[-1]]
                    # store and reshape the values in a np.array() in case eqdsk_format[key]['size'] > max_values
                    elif eqdsk_format[key]['size'][0] > max_values:
                        values = [np.array(values).reshape(eqdsk_format[key]['size'])]
                    # store the var value pairs in the eqdsk dict
                    eqdata.update({var:values[k] for k,var in enumerate(eqdsk_format[key]['vars'])})
                # update the current position in the 
                current_row = end_row

    else:
        print('Invalid file or path provided!')

    return eqdata

def write_eqdsk(filepath, nw, nh, rdim, zdim, rcentr, rleft, zmid, rmaxis, zmaxis, simag, sibry, bcentr, \
                current, fpol, pres, ffprim, pprime, psirz, qpsi, nbbbs, limitr, \
                rbbbs=None, zbbbs=None, rlim=None, zlim=None, case='', idum=0, **kwargs):
    """
    Writes provided equilibrium data into the EQDSK format, requires passing explicit arguments. Provided from EX2GK from Aaron Ho.

    :arg filepath: str. Name of the EQDSK file to be generated.

    :arg nw: int. Number of radial points in 2D grid and in 1D profiles, assumed equal to each other.

    :arg nh: int. Number of vertical points in 2D grid.

    :arg rdim: float. Width of 2D grid box, in the radial direction.

    :arg zdim: float. Height of 2D grid box, in the vertical direction.

    :arg rcentr: float. Location of the geometric center of the machine in the radial direction, does not necessarily need to be mid-point in radial direction of 2D grid.

    :arg rleft: float. Location of the left-most point in radial direction (lowest radial value) of 2D grid, needed for grid reconstruction.

    :arg zmid: float. Location of the mid-point in vertical direction of 2D grid, needed for grid reconstruction.

    :arg rmaxis: float. Location of the magnetic center of the equilibrium in the radial direction.

    :arg zmaxis: float. Location of the magnetic center of the equilibrium in the vertical direction.

    :arg simag: float. Value of the poloidal flux at the magnetic center of the equilibrium.

    :arg sibry: float. Value of the poloidal flux at the boundary of the equilibrium, defined as the last closed flux surface and not necessarily corresponding to any edge of the 2D grid.

    :arg bcentr: float. Value of the toroidal magnetic field at the geometric center of the machine, typically provided as the value in vacuum.

    :arg current: float. Value of the total current in the plasma, typically provided as the total current in the toroidal direction.

    :arg fpol: array. Absolute unnormalized poloidal flux as a function of radius.

    :arg pres: array. Total plasma pressure as a function of radius.

    :arg ffprime: array. F * derivative of F with respect to normalized poloidal flux as a function of radius.

    :arg pprime: array. Derivative of plasma pressure with respect to normalized poloidal flux as a function of radius.

    :arg psirz: array. 2D poloidal flux map as a function of radial coordinate and vertical coordinate.

    :arg qpsi: array. Safety factor as a function of radius.

    :arg nbbbs: int. Number of points in description of plasma boundary contour, can be zero.

    :arg limitr: int. Number of points in description of plasma limiter contour, can be zero.

    :arg rbbbs: array. Ordered list of radial values corresponding to points in the plasma boundary contour description.

    :arg zbbbs: array. Ordered list of vertical values corresponding to points in the plasma boundary contour description.

    :arg rlim: array. Ordered list of radial values corresponding to points in the plasma boundary contour description.

    :arg zlim: array. Ordered list of vertical values corresponding to points in the plasma boundary contour description.

    :kwarg case: str. String to identify file, non-essential and written into the 48 character space at the start of the file.

    :kwarg idum: int. Dummy integer value to identify file origin, non-essential and is sometimes used to identify the FORTRAN output number.

    :returns: none.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath field must be a string. EQDSK file write aborted.")
    if not isinstance(nw, int):
        raise TypeError("nw field must be an integer. EQDSK file write aborted.")
    if not isinstance(nh, int):
        raise TypeError("nh field must be an integer. EQDSK file write aborted.")
    if not isinstance(rdim, float):
        raise TypeError("rdim field must be a real number. EQDSK file write aborted.")
    if not isinstance(zdim, float):
        raise TypeError("zdim field must be a real number. EQDSK file write aborted.")
    if not isinstance(rcentr, float):
        raise TypeError("rcentr field must be a real number. EQDSK file write aborted.")
    if not isinstance(rleft, float):
        raise TypeError("rleft field must be a real number. EQDSK file write aborted.")
    if not isinstance(zmid, float):
        raise TypeError("zmid field must be a real number. EQDSK file write aborted.")
    if not isinstance(rmaxis, float):
        raise TypeError("rmaxis field must be a real number. EQDSK file write aborted.")
    if not isinstance(zmaxis, float):
        raise TypeError("zmaxis field must be a real number. EQDSK file write aborted.")
    if not isinstance(simag, float):
        raise TypeError("simag field must be a real number. EQDSK file write aborted.")
    if not isinstance(sibry, float):
        raise TypeError("sibry field must be a real number. EQDSK file write aborted.")
    if not isinstance(bcentr, float):
        raise TypeError("bcentr field must be a real number. EQDSK file write aborted.")
    if not isinstance(current, float):
        raise TypeError("current field must be a real number. EQDSK file write aborted.")
    if not isinstance(fpol, array_types):
        raise TypeError("fpol field must be an integer. EQDSK file write aborted.")
    if not isinstance(pres, array_types):
        raise TypeError("pres field must be an integer. EQDSK file write aborted.")
    if not isinstance(ffprim, array_types):
        raise TypeError("ffprim field must be an integer. EQDSK file write aborted.")
    if not isinstance(pprime, array_types):
        raise TypeError("pprime field must be an integer. EQDSK file write aborted.")
    if not isinstance(psirz, array_types):
        raise TypeError("psirz field must be an integer. EQDSK file write aborted.")
    if not isinstance(qpsi, array_types):
        raise TypeError("qpsi field must be an integer. EQDSK file write aborted.")
    if nbbbs is not None and not isinstance(nbbbs, int):
        raise TypeError("nbbbs field must be an integer or set to None. EQDSK file write aborted.")
    if limitr is not None and not isinstance(limitr, int):
        raise TypeError("limitr field must be an integer or set to None. EQDSK file write aborted.")
    if rbbbs is not None and not isinstance(rbbbs, array_types):
        raise TypeError("rbbbs field must be an integer. EQDSK file write aborted.")
    if zbbbs is not None and not isinstance(zbbbs, array_types):
        raise TypeError("zbbbs field must be an integer. EQDSK file write aborted.")
    if rlim is not None and not isinstance(rlim, array_types):
        raise TypeError("rlim field must be an integer. EQDSK file write aborted.")
    if zlim is not None and not isinstance(zlim, array_types):
        raise TypeError("zlim field must be an integer. EQDSK file write aborted.")
    if not isinstance(case, str):
        raise TypeError("case field must be a string. EQDSK file write aborted.")
    if not isinstance(idum, int):
        raise TypeError("idum field must be an integer. EQDSK file write aborted.")
    eqpath = Path(filepath)
    if eqpath.is_file():
        print("%s exists, overwriting file with EQDSK file!" % (str(eqpath)))
    if nbbbs is None or rbbbs is None or zbbbs is None:
        nbbbs = 0
        rbbbs = []
        zbbbs = []
    if limitr is None or rlim is None or zlim is None:
        limitr = 0
        rlim = []
        zlim = []
    with open(str(eqpath), 'w') as ff:
        gcase = ""
        if "code" in kwargs:
            gcase = gcase + kwargs["code"] + " "
        gcase = gcase + case[:48 - len(gcase)] if (len(case) - len(gcase)) > 48 else gcase + case
        ff.write("%-48s%4d%4d%4d\n" % (gcase, idum, nw, nh))
        ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (rdim, zdim, rcentr, rleft, zmid))
        ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (rmaxis, zmaxis, simag, sibry, bcentr))
        ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (current, simag, 0.0, rmaxis, 0.0))
        ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (zmaxis, 0.0, sibry, 0.0, 0.0))
        for ii in range(0, len(fpol)):
            ff.write("%16.9E" % (fpol[ii]))
            if (ii + 1) % 5 == 0 and (ii + 1) != len(fpol):
                ff.write("\n")
        ff.write("\n")
        for ii in range(0, len(pres)):
            ff.write("%16.9E" % (pres[ii]))
            if (ii + 1) % 5 == 0 and (ii + 1) != len(pres):
                ff.write("\n")
        ff.write("\n")
        for ii in range(0, len(ffprim)):
            ff.write("%16.9E" % (ffprim[ii]))
            if (ii + 1) % 5 == 0 and (ii + 1) != len(ffprim):
                ff.write("\n")
        ff.write("\n")
        for ii in range(0, len(pprime)):
            ff.write("%16.9E" % (pprime[ii]))
            if (ii + 1) % 5 == 0 and (ii + 1) != len(pprime):
                ff.write("\n")
        ff.write("\n")
        kk = 0
        for ii in range(0, nh):
            for jj in range(0, nw):
                ff.write("%16.9E" % (psirz[ii, jj]))
                if (kk + 1) % 5 == 0 and (kk + 1) != nh * nw:
                    ff.write("\n")
                kk = kk + 1
        ff.write("\n")
        for ii in range(0, len(qpsi)):
            ff.write("%16.9E" % (qpsi[ii]))
            if (ii + 1) % 5 == 0 and (ii + 1) != len(qpsi):
                ff.write("\n")
        ff.write("\n")
        ff.write("%5d%5d\n" % (nbbbs, limitr))
        kk = 0
        for ii in range(0, nbbbs):
            ff.write("%16.9E" % (rbbbs[ii]))
            if (kk + 1) % 5 == 0 and (ii + 1) != nbbbs:
                ff.write("\n")
            kk = kk + 1
            ff.write("%16.9E" % (zbbbs[ii]))
            if (kk + 1) % 5 == 0 and (ii + 1) != nbbbs:
                ff.write("\n")
            kk = kk + 1
        ff.write("\n")
        kk = 0
        for ii in range(0, limitr):
            ff.write("%16.9E" % (rlim[ii]))
            if (kk + 1) % 5 == 0 and (kk + 1) != limitr:
                ff.write("\n")
            kk = kk + 1
            ff.write("%16.9E" % (zlim[ii]))
            if (kk + 1) % 5 == 0 and (kk + 1) != limitr:
                ff.write("\n")
            kk = kk + 1
        ff.write("\n")
    print('Output EQDSK file saved as %s.' % (str(eqpath)))
