import itertools
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SpectralCoord, SpectralQuantity

from .powr import readWRPlot_identfile,wrplot_to_tex
from .utils.logging import setup_log
logger = setup_log(__name__)

class LineIdentifier:
    #TODO: add units, possibility for unit adaptions,make sure it works even if entries of dict_lines have different units
    def __init__(self, dict_lines,font_dict_list=None,x_unit:u.Unit=None):
        have_all_units = all(isinstance(value, (u.quantity.Quantity,SpectralCoord,SpectralQuantity)) for value in dict_lines.values())
        
        self.dict_lines = dict_lines
        
        if font_dict_list is not None:
            assert len(dict_lines) == len(font_dict_list), "Number of entries in the line dictionary is not same as  length of the text kwargs."
        
        self.font_dict_list = font_dict_list
        
    @classmethod
    def from_powr_identfile(cls,filename,keyword='',x_unit:u.Unit=None):
        dict_lines, font_dict_list = readWRPlot_identfile(filepath=filename,keyword=keyword)
        print(dict_lines, font_dict_list)
        print(len(dict_lines), len(font_dict_list))
        return cls(dict_lines, font_dict_list,x_unit=x_unit)
        
    def plot(self,
             base_yoff=1.02,
             root=0.01,
             stem=0.1,
             stem_xoff_rel_cen=0,
             text_yoff = 0.03,
             ax=None,
             line_kwargs={'linewidth':0.7,'color':'k'}, 
             text_kwargs={'fontsize':10,'rotation':90,'color':'k','ha':'center', 'va':'bottom'}):
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

        """
        if ax is None:
            
            fig, ax = plt.subplots(figsize=(8,4))
        else: 
            fig = ax.get_figure()

        #flattened list of all wavelengths
        wavel = list(itertools.chain.from_iterable(self.dict_lines.values()))
        #y value for vertical root lines
        ymin = base_yoff
        ymax = base_yoff+root
        #vertical root lines which point to spectral lines
        ax.vlines(wavel,ymin=ymin, ymax=ymax,**line_kwargs)

        #list of minimum and maximum x values if there are multiplets
        xmin_xmax_values = np.array([[min(value), max(value)] for value in self.dict_lines.values()])
        
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

        #coordinates of text of labels
        y = base_yoff+root+stem+text_yoff
        _,ymax = ax.get_ylim()
        if ymax < y:
            logger.warning(f'Text out of ylim, automatically adapting ymax now from ymax_old={ymax:.2f} to ymax_new={y:.2f}')
            ax.set_ylim(None,1.05*y)
        #print text labels
        line_labels = list(self.dict_lines.keys())
        for i, x in enumerate(stem_lamb):
            if self.font_dict_list is None:
                ax.text(x,y,s=line_labels[i],fontdict=text_kwargs)
            else:
                font_dict = text_kwargs.copy()
                font_dict.update(self.font_dict_list[i])
                ax.text(x,y,s=line_labels[i],fontdict=font_dict)
                    
        return ax