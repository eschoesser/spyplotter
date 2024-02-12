import re
from .utils.logging import setup_log
logger = setup_log(__name__)
from typing import List

#Contains functions that are tailored to read output files of PoWR models 
#and WRPlot input files

def readWRPlotDatasets(filepath, keywords:List, dataset:int):
    """
    Read an xy table from a *.plot file which is an output file 
    of a PoWR model

    :param filepath: model path or path with direct file name
    :type filepath: string
    :param keywords: key words in .plot file under which data sets are saved
    :type keywords: string
    :param dataset: list of number of data sets corresponding to each keyword
    :type dataset: integer
    :return: x and y of read xytable
    """
    
    x = []
    y = []
    with open(filepath,'r') as cformFileHandle:
        if isinstance(keywords, str):
            keywords = [keywords]
            
        for keyword in keywords:
            #read only lines in between startkey and endkey
            startkey = "N=" 
            endkeys = ["N=", "FINISH", "END"]

            xdata = []
            ydata = []    
            readindex = -1
            nskip = dataset - 1
            foundkey = (keyword == "")
            for curline in cformFileHandle:  
                if (not foundkey): 
                    #try to find key in current line
                    keypos = curline.find(keyword) 
                    if (keypos == -1):
                        #if not found, skip iteration and search in next line
                        continue
                    else:
                        #start to read found
                        foundkey = True

                if ((readindex == -1) and foundkey):
                    readindex = 0
                
                #test if found dataset is also corresponding to
                if (readindex == 0):
                    if (curline.strip().startswith(startkey)):
                        if (nskip > 0):
                            nskip = nskip - 1
                        else:
                            readindex = 1
                    continue
                elif (readindex == 1 or readindex == 2):
                    #now we need to read out pairs of lines
                    rawline = curline.rstrip()
                    for endkey in endkeys:
                        if (rawline.strip().startswith(endkey)):
                            #make sure there are only two columns
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

def read_params_from_kasdefs(filename):
    variables = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("\\VAR"):
                parts = line.strip().split("=")
                var_name = parts[0].split()[-1]
                var_value = parts[1].strip()
                variables[var_name] = float(var_value)
    return variables
        


def search_and_replace_math(input_string,search_pattern,replace_pattern):
    """Search and replace string pattern assuming math environment
    Makes sure that math environment is not destroyed by replacing everything by dollar signs

    :param input_string: original string in wrplot format
    :type input_string: str
    :param search_pattern: regular expression string for pattern
    :type search_pattern: str
    :param replace_pattern: How found pattern is translated
    :type replace_pattern: str
    :return: converted latex string
    :rtype: str
    """
    #split given string in math and non-math
    # Every odd index is math environment, and every even index is non-math environment
    string_list = re.split(r'\$(.*?)\$', input_string)
    new_string = ''
    for i in range(0,len(string_list)):
        if i%2==0:
            #non-math environment
            nonmath_string = re.sub(search_pattern,'$' + replace_pattern + '$', string_list[i])
            new_string += nonmath_string
        else:
            #within math environment, add dollar signs again because they are not stored otherwise
            math_string = '$' + re.sub(search_pattern, replace_pattern, string_list[i]) + '$'
            new_string += math_string
    return new_string

def wrplot_to_tex(text_string):
    #explicit math environment
    #\( ...\) -> $ ... $
    text_string1 = re.sub(r'\\\((.*?)\\\)', r'$\1$', text_string)
    
    #fractions
    #\{ ... \| ...\} -> \frac{ ..} {..}
    text_string2 = search_and_replace_math(text_string1,search_pattern=r'\\\{(.*?)\\\|(.*?)\\\}',replace_pattern=r'\\frac{\1}{\2}')
    
    #Subscript
    #&T ... &M -> $_{...}$
    text_string3 = search_and_replace_math(text_string2,search_pattern=r'&T(.*?)&M',replace_pattern=r'_{\1}')
    
    #Superscript
    #&H ... &M -> $^{...}$
    text_string4 = search_and_replace_math(text_string3,search_pattern=r'&H(.*?)&M',replace_pattern=r'^{\1}')
    
    #sub + superscript
    #&H ... &T ... &M -> $^..._...$
    text_string5 = search_and_replace_math(text_string4,search_pattern=r'&H(?!&M)(.?)&T(.*?)&M',replace_pattern=r'^{\1}_{\2}')
    
    #Italic math font
    #&R ... &N -> $ .... $
    text_string6 = search_and_replace_math(text_string5,search_pattern=r'&R(.*?)&N', replace_pattern=r'\1')
    
    #Change mathmatical symbols
    new_string = text_string6
    for key, value in search_replace_dict.items():
        new_string = search_and_replace_math(new_string,search_pattern=key,replace_pattern=value)
        
    # Format dictionary for text
    style_dict = {}

    for key, value in search_strip_addtextkwargs.items():
        # Search for key in given string and add kwargs if necessary
        if key in new_string:
            # If found, strip the key and add to the dictionary the value
            new_string = new_string.replace(key, "")
            style_dict.update(value)
    
    #Replace characters
    for key, value in search_replace.items():
        # Search for key in given string and add kwargs if necessary
        if key in new_string:
            # If found, strip the key
            new_string = new_string.replace(key, value)
            
    return new_string, style_dict


#when finding following characters, replace them:
search_replace = {
    "'": "",
    '"': ''
}

#when finding a character of following shape, strip it and add kwargs dictionary for ax.text
search_strip_addtextkwargs = {
    r'&0': {'color':'C00'},
    r'&1': {'color':'C01'},
    r'&2': {'color':'C02'},
    r'&3': {'color':'C03'},
    r'&4': {'color':'C04'},
    r'&5': {'color':'C05'},
    r'&6': {'color':'C06'},
    r'&7': {'color':'C07'},
    r'&8': {'color':'C08'},
    r'&9': {'color':'C09'},
    r'&I': {'style':'italic'},
    r'&F':{'weight':'bold'},
    r'&W':{'weight':'semi-bold'},
    r'&E':{'fontstretch':'ultra-condensed'},
    r'&B':{'fontstretch':'ultra-expanded'}
    }
    
#
search_replace_dict = {
        '\\\,': '\,',           '\\\S': '_{\\\odot}',
        '#a#': '\\\\alpha',     '#A#': '\\\\Alpha',
        '#b#': '\\\\beta',      '#B#': '\\\\Beta',
        '#g#': '\\\\gamma',     '#G#': '\\\\Gamma',
        '#d#': '\\\\delta',     '#D#': '\\\\Delta',
        '#e#': '\\\\epsilon',   '#E#': '\\\\Epsilon',
        '#z#': '\\\\zeta',      '#Z#': '\\\\Zeta',
        '#t#': '\\\\theta',     '#T#': '\\\\Theta',
        '#i#': '\\\\iota',      '#I#': '\\\\Iota',
        '#k#': '\\\\kappa',     '#K#': '\\\\Kappa',
        '#l#': '\\\\lambda',    '#L#': '\\\\Lambda',
        '#m#': '\\\\mu',        '#M#': '\\\\Mu',
        '#n#': '\\\\nu',        '#N#': '\\\\Nu',
        '#x#': '\\\\xi',        '#X#': '\\\\Xi',
        '#o#': '\\\\omicron',   '#O#': '\\\\Omicron',
        '#p#': '\\\\pi',        '#P#': '\\\\Pi',
        '#r#': '\\\\rho',       '#R#': '\\\\Rho',
        '#s#': '\\\\sigma',     '#S#': '\\\\Sigma',
        '#t#': '\\\\tau',       '#T#': '\\\\Tau',
        '#u#': '\\\\upsilon',   '#U#': '\\\\Upsilon',
        '#f#': '\\\\phi',       '#F#': '\\\\Phi',
        '#c#': '\\\\chi',       '#C#': '\\\\Chi',
        '#y#': '\\\\psi',       '#Y#': '\\\\Psi',
        '#w#': '\\\\omega',     '#W#': '\\\\Omega$'
    }