import re
from .utils.logging import setup_log

logger = setup_log(__name__)
from typing import List
import numpy as np

import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_p

# Contains functions that are tailored to read output files of PoWR models
# and WRPlot input files


def readWRPlotDatasets(filepath, keywords: List, dataset: int):
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
    with open(filepath, "r") as cformFileHandle:
        if isinstance(keywords, str):
            keywords = [keywords]

        for keyword in keywords:
            # read only lines in between startkey and endkey
            startkey = "N="
            endkeys = ["N=", "FINISH", "END"]

            xdata = []
            ydata = []
            readindex = -1
            nskip = dataset - 1
            foundkey = keyword == ""
            for curline in cformFileHandle:
                if not foundkey:
                    # try to find key in current line
                    keypos = curline.find(keyword)
                    if keypos == -1:
                        # if not found, skip iteration and search in next line
                        continue
                    else:
                        # start to read found
                        foundkey = True

                if (readindex == -1) and foundkey:
                    readindex = 0

                # test if found dataset is also corresponding to
                if readindex == 0:
                    if curline.strip().startswith(startkey):
                        if nskip > 0:
                            nskip = nskip - 1
                        else:
                            readindex = 1
                    continue

                if curline.strip().startswith("KASDEF"):
                    continue

                elif (readindex == 1 or readindex == 2):
                    #now we need to read out pairs of lines
                    rawline = curline.rstrip()
                    for endkey in endkeys:
                        if rawline.strip().startswith(endkey):
                            # make sure there are only two columns
                            if readindex == 2:
                                logger.error("FATAL ERROR: odd number of xy-lines")
                                raise ValueError
                            readindex = 99
                            break
                    xynewset = rawline.split()
                    if readindex == 1:
                        for xynew in xynewset:
                            xdata.append(float(xynew))
                        readindex = 2
                    elif readindex == 2:
                        for xynew in xynewset:
                            ydata.append(float(xynew))
                        readindex = 1
                elif readindex > 9:
                    break

            if not foundkey:
                logger.error(f"Could not find keyword {keyword}")
                raise KeyError

            if readindex == 0:
                logger.error(f"Could not find dataset {dataset}")
                raise KeyError

            x += xdata
            y += ydata

    return x, y


def read_params_from_kasdefs(filename):
    variables = {}
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("\\VAR"):
                parts = line.strip().split("=")
                var_name = parts[0].split()[-1]
                var_value = parts[1].strip()
                try:
                    variables[var_name] = float(var_value)
                except ValueError:
                    variables[var_name] = var_value
    return variables


def search_and_replace_math(input_string, search_pattern, replace_pattern):
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
    # split given string in math and non-math
    # Every odd index is math environment, and every even index is non-math environment
    string_list = re.split(r"\$(.*?)\$", input_string)
    new_string = ""
    for i in range(0, len(string_list)):
        if i % 2 == 0:
            # non-math environment
            nonmath_string = re.sub(
                search_pattern, "$" + replace_pattern + "$", string_list[i]
            )
            new_string += nonmath_string
        else:
            # within math environment, add dollar signs again because they are not stored otherwise
            math_string = (
                "$" + re.sub(search_pattern, replace_pattern, string_list[i]) + "$"
            )
            new_string += math_string
    return new_string


def wrplot_to_tex(text_string):
    # explicit math environment
    # \( ...\) -> $ ... $
    text_string1 = re.sub(r"\\\((.*?)\\\)", r"$\1$", text_string)

    # fractions
    # \{ ... \| ...\} -> \frac{ ..} {..}
    text_string2 = search_and_replace_math(
        text_string1,
        search_pattern=r"\\\{(.*?)\\\|(.*?)\\\}",
        replace_pattern=r"\\frac{\1}{\2}",
    )

    # Subscript
    # &T ... &M -> $_{...}$
    text_string3 = search_and_replace_math(
        text_string2, search_pattern=r"&T(.*?)&M", replace_pattern=r"_{\1}"
    )

    # Superscript
    # &H ... &M -> $^{...}$
    text_string4 = search_and_replace_math(
        text_string3, search_pattern=r"&H(.*?)&M", replace_pattern=r"^{\1}"
    )

    # sub + superscript
    # &H ... &T ... &M -> $^..._...$
    text_string5 = search_and_replace_math(
        text_string4,
        search_pattern=r"&H(?!&M)(.?)&T(.*?)&M",
        replace_pattern=r"^{\1}_{\2}",
    )

    # Italic math font
    # &R ... &N -> $ .... $
    text_string6 = search_and_replace_math(
        text_string5, search_pattern=r"&R(.*?)&N", replace_pattern=r"\1"
    )

    # Change mathmatical symbols
    new_string = text_string6
    for key, value in search_replace_dict.items():
        new_string = search_and_replace_math(
            new_string, search_pattern=key, replace_pattern=value
        )

    # Format dictionary for text
    style_dict = {}

    for key, value in search_strip_addtextkwargs.items():
        # Search for key in given string and add kwargs if necessary
        if key in new_string:
            # If found, strip the key and add to the dictionary the value
            new_string = new_string.replace(key, "")
            style_dict.update(value)

    # Replace characters
    for key, value in search_replace.items():
        # Search for key in given string and add kwargs if necessary
        if key in new_string:
            # If found, strip the key
            new_string = new_string.replace(key, value)

    # Strip any remaining &N
    new_string = re.sub(r"&N", "", new_string)

    # Strip any remaining &
    new_string = new_string.strip("&")

    return new_string, style_dict


def read_elements_datom_header(filename):
    """
    Read ELEMENT and ION definitions from the header of a file.

    Assumes:
    - Header lines start with '*'
    - Reading should stop as soon as a line does not start with '*'
    - Format examples:
        * ELEMENT H
        * ION I
        * ION II
        * ELEMENT MG
        * ION I   NLEVEL=1
    """

    elements = {}
    current_element = None

    with open(filename, "r") as f:
        for line in f:
            # Stop immediately if header ends
            if not line or line[0] != "*":
                break

            # Remove leading '*' and surrounding whitespace
            content = line[1:].strip()
            if not content:
                continue

            # Fast tokenization
            parts = content.split()
            keyword = parts[0]

            if keyword == "ELEMENT":
                # Element name is the second token
                if len(parts) > 1:
                    current_element = parts[1]
                    elements[current_element] = []

                    # Check for numeric range after element name
                    if len(parts) == 4:
                        try:
                            start = int(parts[2])
                            end = int(parts[3])
                            # Add each level as string
                            elements[current_element].extend(
                                i for i in range(start, end + 1)
                            )
                        except ValueError:
                            pass

            elif keyword == "ION" and current_element is not None:
                # Ionization stage is the second token
                if len(parts) > 1:
                    roman = parts[1]
                    ion_number = roman_to_int.get(roman)
                    if ion_number is not None:
                        elements[current_element].append(ion_number)

    return elements


def powrsplinpo(x, ydata, xdata, calcdfdx=False):

    N = len(xdata)

    dn = xdata[-1] - xdata[0]
    for i in range(1, N):
        dx = xdata[i] - xdata[i - 1]
        if dx * dn <= 0:
            print("FATAL ERROR: x values not in strictly monotonic order")
            return False

    # Find the interval
    ivfound = False
    for i in range(1, N):
        if (x - xdata[i - 1]) * (x - xdata[i]) <= 0.0:
            lip = i
            ivfound = True
            break

    if not ivfound:
        print("FATAL ERROR: X outside interpolation range")
        print("xmin, xmax, x = ", min(xdata), max(xdata), x)
        return False

    # Determination of the coefficients p1, p2, p3, p4
    # Set up the coefficient matrix
    d1 = 1.0 / (xdata[lip] - xdata[lip - 1])
    d2 = d1 * d1
    d3 = d1 * d2
    d23 = d2 / 3.0
    h11 = d3
    h12 = -d3
    h13 = d23
    h14 = 2.0 * d23
    h21 = -d1
    h22 = 2.0 * d1
    h23 = -0.333333333333333
    h24 = -0.666666666666666
    h31 = -d3
    h32 = d3
    h33 = -2.0 * d23
    h34 = -d23
    h41 = 2.0 * d1
    h42 = -d1
    h43 = 0.666666666666666
    h44 = 0.333333333333333

    # FOR THE BOUNDARY INTERVALS THE DERIVATIVE CANNOT EXTEND OVER THE BOUNDARY
    la = max(lip - 2, 0)
    lb = min(lip + 1, N - 1)

    # FUNCTION TO BE INTERPOLATED: ydata
    f1 = ydata[lip - 1]
    f2 = ydata[lip]

    # Standard version: zentrierte Ableitungen an den Stuetzstellen. Das
    #  ist bei nicht aequidistanten Stuetzstellen fragwuerdig, verringert
    #  aber andererseits das Ueberschwingen insbesondere wenn man nicht MONO verwendet
    f3 = (ydata[lip] - ydata[la]) / (xdata[lip] - xdata[la])
    f4 = (ydata[lb] - ydata[lip - 1]) / (xdata[lb] - xdata[lip - 1])

    s4 = (ydata[lip] - ydata[lip - 1]) / (xdata[lip] - xdata[lip - 1])

    #   We are not in the first interval:
    if la != lip - 2:
        s3 = s4
    else:
        s3 = (ydata[lip - 1] - ydata[lip - 2]) / (xdata[lip - 1] - xdata[lip - 2])

    #   We are not in the last interval:
    if lb != lip + 1:
        s5 = s4
    else:
        s5 = (ydata[lip + 1] - ydata[lip]) / (xdata[lip + 1] - xdata[lip])

    f3 = (np.sign(s3) + np.sign(s4)) * min(abs(s3), abs(s4), 0.5 * abs(f3))
    f4 = (np.sign(s4) + np.sign(s5)) * min(abs(s4), abs(s5), 0.5 * abs(f4))

    #   Calculate polynomial coefficients: P(vector) = h (maxtrix) * f(vector)
    p1 = h11 * f1 + h12 * f2 + h13 * f3 + h14 * f4
    p2 = h21 * f1 + h22 * f2 + h23 * f3 + h24 * f4
    p3 = h31 * f1 + h32 * f2 + h33 * f3 + h34 * f4
    p4 = h41 * f1 + h42 * f2 + h43 * f3 + h44 * f4

    #   Evaluation of the interpolation polynomial
    dxm = x - xdata[lip - 1]
    dx = xdata[lip] - x
    y = (p1 * dxm * dxm + p2) * dxm + (p3 * dx * dx + p4) * dx

    if calcdfdx:
        dfdx = 3.0 * p1 * dxm * dxm + p2 - 3.0 * p3 * dx * dx - p4
        return y, dfdx
    else:
        return y


# when finding following characters, replace them:
search_replace = {"'": "", '"': ""}

# when finding a character of following shape, strip it and add kwargs dictionary for ax.text
search_strip_addtextkwargs = {
    r"&0": {"color": "C00"},
    r"&1": {"color": "C01"},
    r"&2": {"color": "C02"},
    r"&3": {"color": "C03"},
    r"&4": {"color": "C04"},
    r"&5": {"color": "C05"},
    r"&6": {"color": "C06"},
    r"&7": {"color": "C07"},
    r"&8": {"color": "C08"},
    r"&9": {"color": "C09"},
    r"&I": {"style": "italic"},
    r"&F": {"weight": "bold"},
    r"&W": {"weight": "semi-bold"},
    r"&E": {"fontstretch": "ultra-condensed"},
    r"&B": {"fontstretch": "ultra-expanded"},
}

#
search_replace_dict = {
    "\\\,": "\,",
    "\\\S": "_{\\\odot}",
    "#a#": "\\\\alpha",
    "#A#": "\\\\Alpha",
    "#b#": "\\\\beta",
    "#B#": "\\\\Beta",
    "#g#": "\\\\gamma",
    "#G#": "\\\\Gamma",
    "#d#": "\\\\delta",
    "#D#": "\\\\Delta",
    "#e#": "\\\\epsilon",
    "#E#": "\\\\Epsilon",
    "#z#": "\\\\zeta",
    "#Z#": "\\\\Zeta",
    "#t#": "\\\\theta",
    "#T#": "\\\\Theta",
    "#i#": "\\\\iota",
    "#I#": "\\\\Iota",
    "#k#": "\\\\kappa",
    "#K#": "\\\\Kappa",
    "#l#": "\\\\lambda",
    "#L#": "\\\\Lambda",
    "#m#": "\\\\mu",
    "#M#": "\\\\Mu",
    "#n#": "\\\\nu",
    "#N#": "\\\\Nu",
    "#x#": "\\\\xi",
    "#X#": "\\\\Xi",
    "#o#": "\\\\omicron",
    "#O#": "\\\\Omicron",
    "#p#": "\\\\pi",
    "#P#": "\\\\Pi",
    "#r#": "\\\\rho",
    "#R#": "\\\\Rho",
    "#s#": "\\\\sigma",
    "#S#": "\\\\Sigma",
    "#t#": "\\\\tau",
    "#T#": "\\\\Tau",
    "#u#": "\\\\upsilon",
    "#U#": "\\\\Upsilon",
    "#f#": "\\\\phi",
    "#F#": "\\\\Phi",
    "#c#": "\\\\chi",
    "#C#": "\\\\Chi",
    "#y#": "\\\\psi",
    "#Y#": "\\\\Psi",
    "#w#": "\\\\omega",
    "#W#": "\\\\Omega$",
}

romans = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
    13: "XIII",
    14: "XIV",
    15: "XV",
    16: "XVI",
    17: "XVII",
    18: "XVIII",
    19: "XIX",
    20: "XX",
    21: "XXI",
    22: "XXII",
    23: "XXIII",
    24: "XXIV",
    25: "XXV",
    26: "XXVI",
    27: "XXVII",
    28: "XXVIII",
    29: "XXIX",
    30: "XXX",
}

roman_to_int = {v: k for k, v in romans.items()}
