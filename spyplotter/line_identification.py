import itertools
import numpy as np
import re
import yaml
import matplotlib.pyplot as plt
from itertools import chain
from astropy import units as u
from astropy.coordinates import SpectralCoord, SpectralQuantity

from .powr import wrplot_to_tex
from .utils.logging import setup_log
logger = setup_log(__name__)

class SpectralLine:
    def __init__(self, ion_name, wavelengths, plotting_style_dict={}):
        self.ion_name = ion_name
        
        #Check how nested wavelength array is
        #Make sure self.wavelengths is list of lists
        if isinstance(wavelengths,(int, float)):
            self.wavelengths = [[wavelengths]]
        elif isinstance(wavelengths[0],(int, float)):
            self.wavelengths = [wavelengths]
        elif isinstance(wavelengths[0], (list, tuple, np.ndarray)):
            self.wavelengths = wavelengths
        else:
            logger.error('wavelengths have wrong type. Make sure that it is a list/ tuple/ array like of floats.')
        
        self.plotting_style = plotting_style_dict

    def __str__(self):
        return f"SpectralLine({self.ion_name}:\n\t{self.wavelengths},\n\t{self.plotting_style})"

    def to_dict(self):
        return {self.ion_name: 
            {
            'wavelengths': self.wavelengths,
            'plotting_style': self.plotting_style
            }
        }

class LineIdentifier():
    #TODO: add units, possibility for unit adaptions,make sure it works even if entries of dict_lines have different units
    def __init__(self, spectral_lines={}):
        #have_all_units = all(isinstance(value, (u.quantity.Quantity,SpectralCoord,SpectralQuantity)) for value in dict_lines.values())
        
        super(LineIdentifier, self).__init__()
        
        self._spectral_lines = spectral_lines
        
    def __str__(self):
        str_dict = self.to_dict()
        result = ""
        for i, line in str_dict.items():
            result += f"{i}: {line}\n"
        return result
    
    @property
    def spectral_lines(self):
        # dictionary of SpectralLine 
        return self._spectral_lines
    
    @property
    def ions(self):
        return list(self._spectral_lines.keys())
    
    @property
    def wavelengths(self):
        # nested list of all wavelengths for all ions
        return [self._spectral_lines[ion_name].wavelengths for ion_name in self._spectral_lines.keys()]
    
    @property
    def wavelengths_flattened(self):
        #flattened list of all wavelengths for all ions
        return [item for sublist in self.wavelengths for subsublist in sublist for item in subsublist]
    
    def update_plotting_style(self, ion_name, new_plotting_style):
        if ion_name in self._spectral_lines:
            self._spectral_lines[ion_name].plotting_style = new_plotting_style
    
    def get_ion_lines(self,ion_name):
        #wavelengths of ion lines
        if ion_name in self._spectral_lines:
            return self._spectral_lines[ion_name].wavelengths
        else:
            logger.error('There are no lines for chosen ion')
            
    def add_spectral_line(self, spectral_line):
        """Add Spectral Line to Line Identification
        If ion already exists, plotting style is updated to newly given type

        :param spectral_line: contains information about added lines
        :type spectral_line: Spectral Line
        """
        ion_name = spectral_line.ion_name

        if ion_name in self._spectral_lines:
            wavelengths = spectral_line.wavelengths
            plotting_dict = spectral_line.plotting_style
            self._spectral_lines[ion_name].wavelengths.extend(wavelengths)
            self._spectral_lines[ion_name].plotting_style.update(plotting_dict)
        else:
            self._spectral_lines.update({ion_name:spectral_line})

    @classmethod
    def from_yaml(cls, file_path):
        # Read Line Identification class from yaml file
        with open(file_path, 'r') as yaml_file:
            spectral_lines_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return cls.from_dict(spectral_lines_dict)
    
    def to_yaml(self, file_path):
        # Write dictionary to yaml file
        with open(file_path, 'w') as yaml_file:
            yaml.dump(self.to_dict(), yaml_file, default_flow_style=False)
            
    @classmethod
    def from_dict(cls,spectral_lines_dict):
        # Create dictionary of SpectralLine type from dictionary of dictionaries
        spectral_lines = {}
        for ion, line in spectral_lines_dict.items():
            if isinstance(line,dict):
                if 'plotting_style' in line:
                    spectral_lines.update({ion:SpectralLine(ion_name=ion, wavelengths=line['wavelengths'],plotting_style_dict=line['plotting_style'])})
                else:
                    spectral_lines.update({ion:SpectralLine(ion_name=ion, wavelengths=line['wavelengths'])})
            elif isinstance(line,(float,int,list, tuple, np.ndarray)):
                spectral_lines.update({ion:SpectralLine(ion_name=ion, wavelengths=line)})
            
        return cls(spectral_lines)
    
    def to_dict(self):
        #Convert all SpectralLine objects to dictionary
        # Yields nested dictionary
        spectral_lines_dict = {}
        for line in self._spectral_lines.values():
            spectral_lines_dict.update(line.to_dict())
        return spectral_lines_dict
    
    @classmethod
    def from_powr_identfile(cls,filename,keyword='',x_unit:u.Unit=None):
        """Reads input ident file of WRPlot and converts it to
        a dictionary containing the information of the lines 
        and a dictionary containing the text style properties 

        :param filepath: file path
        :type filepath: string or Path
        :param keyword: keyword in ident file that is looked for to find 
                        corresponding set of lines
        :type keyword: string
        :return: dictionary containing string in wrplot format and line positions
        :rtype: _type_
        """
        endkeys = ["FINISH", "END"]
        spectral_lines = {}
        with open(filename,'r') as file:
            foundkey = (keyword == "")
            for line in file:  
                if (not foundkey): 
                    keypos = line.find(keyword)
                    if (keypos == -1):
                        #skip iteration if keyword was not found yet
                        continue
                    else:
                        foundkey = True
                        
                rawline = line.rstrip()
                # Define the known keywords at beginning of lines that are read
                known_keywords = [r'\IDENT','\IDMULT']
                for known_keyword in known_keywords:
                    # Use regular expression to match the known keyword, floats, and the rest of the string
                    pattern = r'({})\s+([\d.]+(?:\s+[\d.]+)*)\s*(.*)'.format(re.escape(known_keyword))

                    # Use re.match to find the pattern in the read line
                    match = re.match(pattern, rawline)
                    if match is not None:
                        #group string into beginning key word, floats and text_label
                        _, floats, text_label = match.groups()
                        floats_list = [float(num) for num in re.findall(r'[\d.]+', floats)]
                        #convert dictionary keys to latex format and plotting dictionary
                        ion_name, plotting_dict = wrplot_to_tex(text_label.strip())
                        
                        if ion_name in spectral_lines:
                            spectral_lines[ion_name].wavelengths.extend([floats_list])
                            spectral_lines[ion_name].plotting_style.update(plotting_dict)
                        else:
                            sl = SpectralLine(ion_name=ion_name, wavelengths=[floats_list], plotting_style_dict=plotting_dict)
                            spectral_lines.update({ion_name:sl})
                        break
                #if one of end keys is read, stop reading
                if any(rawline.strip().startswith(endkey) for endkey in endkeys):
                    break
                
        return cls(spectral_lines)
        
    def plot(self,
             base_yoff=1.02,
             root=0.05,
             stem=0.05,
             stem_xoff_rel_cen=0,
             text_yoff = 0.03,
             ax=None,
             line_kwargs={'linewidth':0.7,'color':'k'}, 
             text_kwargs={'fontsize':10,'rotation':90,'color':'k','ha':'center', 'va':'bottom'},
             default_kwargs=True):
        """

                                NAME
                                    (text_yoff)
            (stem)  _____C________|    (stem_xoff_rel_cen)
            (root) |          |
                                    (base_yoff)
            vvvvv\ /vvvvvvv\    /vvvvvvvvvvvvvvv-spectrum-vvvv
                V         |  |
                            \/

            stem_xoff_rel_cen : stem x offset relative to the center (C, average wavelength of the lambdas (lambN set))
            base_yoff         : base of the ident y offset
            root              : the length of the "root", the line which points to the spectral line
            stem              : the length of the "stem", the line which points to the label NAME (spectral line id)
            text_yoff         : text y offset relative to the top of the stem
            line_kwargs       : dictionary for customizing vlines and hlines
            default_kwargs    : if set True and new kwargs for text_kwargs and line_kwargs are chosen, the deafault values used but are updated

        """
        if ax is None:
            
            fig, ax = plt.subplots(figsize=(8,4))
        else: 
            fig = ax.get_figure()
            
        if default_kwargs:
            # use default plotting style and only update explicitly changed values
            line_kwargs_default={'linewidth':0.7,'color':'k'} 
            text_kwargs_default={'fontsize':10,'rotation':90,'color':'k','ha':'center', 'va':'bottom'}
            #Update line style
            line_kwargs_default.update(line_kwargs)
            line_kwargs = line_kwargs_default.copy()
            # Update text style
            text_kwargs_default.update(text_kwargs)
            text_kwargs = text_kwargs_default.copy()

        #flattened list of all wavelengths
        wavel = self.wavelengths_flattened
        #y value for vertical root lines
        ymin = base_yoff
        ymax = base_yoff+root
        #vertical root lines which point to spectral lines
        ax.vlines(wavel,ymin=ymin, ymax=ymax,**line_kwargs)

        #list of minimum and maximum x values if there are multiplets
        xmin_xmax_values = np.array([[min(value), max(value)] for ion in self.wavelengths for value in ion])
        
        # mask for xmin and xmax of multiplet lines
        mask = xmin_xmax_values[:, 0] != xmin_xmax_values[:, 1]
        xmin_xmax_multiplet_values = xmin_xmax_values[mask]
        #all horizontal lines are on same y value
        y = [base_yoff+root] * len(xmin_xmax_multiplet_values)
        #horizontal lines connecting root lines corresponding to one multiplet
        ax.hlines(y,xmin=xmin_xmax_multiplet_values[:,0],xmax=xmin_xmax_multiplet_values[:,1],**line_kwargs)

        #stem line x value to label
        stem_lamb = np.mean(xmin_xmax_values,axis=1) + stem_xoff_rel_cen
        #constant y values
        ymin = base_yoff+root
        ymax = base_yoff+root+stem
        #vertical stem lines connecting to text label
        ax.vlines(stem_lamb,ymin,ymax,**line_kwargs)         
            
        #print text labels
        i = 0
        ymax_text = 0
        y_text = ymax+text_yoff
        for line in self.spectral_lines.values():
            font_dict = text_kwargs.copy()
            font_dict.update(line.plotting_style)
            for wavel in line.wavelengths:
                text_object = ax.text(stem_lamb[i],y_text,s=line.ion_name,fontdict=font_dict)
                fig.canvas.draw()
                #find largest y coordinate of text to set ylim correctly later
                text_extent = text_object.get_window_extent()
                data_extent = text_extent.transformed(ax.transData.inverted())
                if data_extent.height + y_text > ymax_text: 
                    ymax_text = data_extent.height + y_text
                i+=1

        #coordinates of text of labels
        _,ymax = ax.get_ylim()
        if ymax < ymax_text:
            logger.warning(f'Text out of ylim, automatically adapting ymax now from ymax_old={ymax:.2f} to ymax_new={1.05 * ymax_text:.2f}')
            ax.set_ylim(None,1.05*ymax_text)
                
        return ax