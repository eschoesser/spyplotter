from .utils.logging import setup_log
logger = setup_log(__name__)

#Contains functions that are specific for PoWR models

def readWRPlotDatasets(filepath, keywords, dataset):
    """Read a dataset from a *.plot file which is an output file 

    :param filepath: model path or path with direct file name
    :type filepath: string
    :param keywords: key words in .plot file under which data sets are saved
    :type keywords: string
    :param dataset: list of number of data sets corresponding to each keyword
    :type dataset: _type_
    :raises KeyError: _description_
    :raises KeyError: _description_
    :return: _description_
    :rtype: _type_
    """
    
    x = []
    y = []
    with open(filepath,'r') as cformFileHandle:
        for keyword in keywords:
            
            startkey = "N=" 
            endkeys = ["N=", "FINISH", "END"]

            xdata = []
            ydata = []    
            readindex = -1
            nskip = dataset - 1
            foundkey = (keyword == "")
            for curline in cformFileHandle:  
                if (not foundkey): 
                    keypos = curline.find(keyword) 
                    if (keypos == -1):
                        continue
                    else:
                        foundkey = True

                if ((readindex == -1) and foundkey):
                    readindex = 0
                    
                if (readindex == 0):
                    if (curline.strip().startswith(startkey)):
                        if (nskip > 0):
                            nskip = nskip - 1
                        else:
                            readindex = 1
                    continue
                elif (readindex == 1 or readindex == 2):
        #           now we need to read out pairs of lines

                    rawline = curline.rstrip()
                    for endkey in endkeys:
                        if (rawline.strip().startswith(endkey)):
                            if (readindex == 2):
                                logger.error('FATAL ERROR: odd number of xy-lines')
                                raise ValueError
                            readindex = 99
                            break
                    xynewset = rawline.split()
                    if (readindex == 1):
                        for xynew in xynewset:
                            xdata.append(float(xynew))
                        readindex = 2
                    elif (readindex == 2):
                        for xynew in xynewset:
                            ydata.append(float(xynew))
                        readindex = 1
                elif (readindex > 9):
                    break

            if (not foundkey):
                logger.error(f'Could not find keyword {keyword}')
                raise KeyError 

            if (readindex == 0):
                logger.error(f'Could not find dataset {dataset}')
                raise KeyError 
            
        x += xdata
        y += ydata
        
    return x,y
