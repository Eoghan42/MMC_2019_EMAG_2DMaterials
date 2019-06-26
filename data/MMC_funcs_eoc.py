import hyperspy.api as hs
import data.atomap_vers_eoc.api as am
from data.atomap_vers_eoc.atom_finding_refining import subtract_average_background
from data.atomap_vers_eoc.atom_finding_refining import normalize_signal
import numpy as np
from ase.io import read, write
from ase.visualize import view
import pandas as pd
import matplotlib.pyplot as plt

import periodictable as pt
import collections
import CifFile
import pyprismatic as pr
import scipy
import skimage
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from skimage.measure import compare_ssim as ssm
from scipy.ndimage.filters import gaussian_filter
from numpy import log
from collections import Counter
from data.atomap_vers_eoc.atom_finding_refining import get_atom_positions_in_difference_image
from numpy import mean
from data.atomap_vers_eoc.atom_finding_refining import _make_circular_mask




def load_data_and_sampling(filename, file_extension=None, invert_image=False, save_image=True):
    
    if '.' in filename:
        s = hs.load(filename)
    else:
        s = hs.load(filename + file_extension)
    #s.plot()

    # Get correct xy units and sampling
    if s.axes_manager[-1].scale == 1:
        real_sampling = 1
        s.axes_manager[-1].units = 'pixels'
        s.axes_manager[-2].units = 'pixels'
        print('WARNING: Image calibrated to pixels, you should calibrate to distance')
    elif s.axes_manager[-1].scale != 1:
        real_sampling = s.axes_manager[-1].scale
        s.axes_manager[-1].units = 'nm'
        s.axes_manager[-2].units = 'nm'
        
    # real_sampling = 
#    physical_image_size = real_sampling * len(s.data)
    
    if invert_image == True:
        s.data = np.divide(1, s.data)
        
        if save_image == True: 
            s.plot()
            plt.title('Image', fontsize = 20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname='Image.png',
                        transparent=True, frameon=False, bbox_inches='tight', 
                        pad_inches=None, dpi=300, labels=False)
            plt.close()
        else:
            pass
        
    else:
        if save_image == True: 
            s.plot()
            plt.title('Image', fontsize = 20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname='Image.png',
                        transparent=True, frameon=False, bbox_inches='tight', 
                        pad_inches=None, dpi=300, labels=False)
            plt.close()
        else:
            pass
    
    return s, real_sampling



def get_and_return_element(element_symbol):
    
    '''
    From the elemental symbol, e.g., 'H' for Hydrogen, returns Hydrogen as
    a periodictable.core.Element object for further use.
    
    Parameters
    ----------
    
    element_symbol : string, default None
        Symbol of an element from the periodic table of elements
    
    Returns
    -------
    A periodictable.core.Element object
    
    Examples
    --------
    >>> Moly = get_and_return_element(element_symbol='Mo')
    >>> print(Moly.covalent_radius)
    >>> print(Moly.number, Moly.symbol)
    
    '''
    
    for ele in pt.elements:
        if ele.symbol == element_symbol:
            chosen_element = ele
    return(chosen_element)


def atomic_radii_in_pixels(sampling, element_symbol):

    '''
    Get the atomic radius of an element in pixels, scaled by an image sampling
    
    Parameters
    ----------
    sampling : float, default None
        sampling of an image in units of nm/pix
    element_symbol : string, default None
        Symbol of an element from the periodic table of elements
    
    Returns
    -------
    Half the colavent radius of the input element in pixels
    
    Examples
    --------
    
    >>> import atomap.api as am
    >>> image = am.dummy_data.get_simple_cubic_signal()
    >>> # pretend it is a 5x5 nm image
    >>> image_sampling = 5/len(image.data) # units nm/pix
    >>> radius_pix_Mo = atomic_radii_in_pixels(image_sampling, 'Mo')
    >>> radius_pix_S = atomic_radii_in_pixels(image_sampling, 'S')
    
    '''
    
    element = get_and_return_element(element_symbol=element_symbol)
    
    # mult by 0.5 to get correct distance (google image of covalent radius)
    # divided by 10 to get nm
    radius_nm = (element.covalent_radius*0.5)/10
    
    radius_pix = radius_nm/sampling
    
    return(radius_pix)




def calibrate_distance_and_intensity(image, 
                                     cropping_area,
                                     separation,
                                     filename,
                                     percent_to_nn=0.2,
                                     mask_radius=None, 
                                     refine=True,
                                     scalebar_true=True):
    
    
    calibrate_intensity_using_sublattice_region(image=image, 
                                                cropping_area=cropping_area,
                                                separation=separation,
                                                percent_to_nn=0.2,
                                                mask_radius=None,
                                                refine=True,
                                                scalebar_true=True)
    
    if filename is not None:
        image.plot()
        plt.title('Calibrated Data ' + filename, fontsize = 20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='calibrated_data_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=300, labels=False)
        plt.close()

        image.save('calibrated_data_' + filename, overwrite=True)


def calibrate_intensity_using_sublattice_region(image, 
                                                cropping_area,
                                                separation, 
                                                percent_to_nn=0.2,
                                                mask_radius=None,
                                                refine=True,
                                                scalebar_true=False): #add max mean min etc.
    
    ''' 
    Calibrates the intensity of an image by using a sublattice, found with some
    atomap functions. The mean intensity of that sublattice is set to 1
    
    Parameters
    ----------
    image : HyperSpy 2D signal, default None
        The signal can be distance calibrated. If it is, set
        scalebar_true=True
    cropping_area : list of 2 floats, default None
        The best method of choosing the area is by using the atomap
        function "add_atoms_with_gui(image.data)". Choose two points on the 
        image. First point is top left of area, second point is bottom right.
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    scalebar_true : Bool, default False
        Set to True if the scale of the image is calibrated to a distance unit.

    Returns
    -------
    calibrated image data
    
    Example
    -------
    
    >>> image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> image.plot()
    >>> cropping_area = am.add_atoms_with_gui(image.data) # choose two points
    >>> calibrate_intensity_using_sublattice_region(image, cropping_area)
    >>> image.plot()

    '''    
    llim, tlim = cropping_area[0]
    rlim, blim = cropping_area[1]

    if image.axes_manager[0].scale != image.axes_manager[1].scale:
        raise ValueError("x & y scales don't match!")
        
    if scalebar_true == True:
        llim *= image.axes_manager[0].scale
        tlim *= image.axes_manager[0].scale
        rlim *= image.axes_manager[0].scale
        blim *= image.axes_manager[0].scale
    else:
        pass

    cal_area = hs.roi.RectangularROI(left=llim, right=rlim, top=tlim, bottom=blim)(image)
    atom_positions = am.get_atom_positions(cal_area, separation=separation, pca=True)
    #atom_positions = am.add_atoms_with_gui(cal_area, atom_positions)
    calib_sub = am.Sublattice(atom_positions, cal_area, color='r')
#    calib_sub.plot()
    if refine == True:
        calib_sub.find_nearest_neighbors()
        calib_sub.refine_atom_positions_using_center_of_mass(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        calib_sub.refine_atom_positions_using_2d_gaussian(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    else:
        pass
    #calib_sub.plot()
    calib_sub.get_atom_column_amplitude_max_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    calib_sub_max_list = calib_sub.atom_amplitude_max_intensity
    calib_sub_scalar = mean(a=calib_sub_max_list)
    image.data = image.data/calib_sub_scalar





def DG_filter(image, filename, d_inner, d_outer, delta, real_space_sampling, units='nm'):

    # Accuracy of calculation. Smaller = more accurate.
    # 0.01 means it will fit until intensity is 0.01 away from 0
    # 0.01 is a good starting value
    
    # Find the FWHM for both positive (outer) and negative (inner) gaussians
    # d_inner is the inner reflection diameter in units of 1/nm (or whatever unit you're working with)
    # I find these in gatan, should be a way of doing automatically.
    
    physical_image_size = real_space_sampling * len(image.data)
    reciprocal_sampling = 1/physical_image_size
    
    # Get radius
    reciprocal_d_inner = (d_inner/2)
    reciprocal_d_outer = (d_outer/2)
    reciprocal_d_inner_pix = reciprocal_d_inner/reciprocal_sampling
    reciprocal_d_outer_pix = reciprocal_d_outer/reciprocal_sampling
    
    fwhm_neg_gaus = reciprocal_d_inner_pix
    fwhm_pos_gaus = reciprocal_d_outer_pix
    
    #s = normalize_signal(subtract_average_background(s))
    image.axes_manager[0].scale = real_space_sampling
    image.axes_manager[1].scale = real_space_sampling
    image.axes_manager[0].units = units
    image.axes_manager[1].units = units
    
    # Get FFT of the image
    image_fft = image.fft(shift=True)
    #image_fft.plot()
    
    # Get the absolute value for viewing purposes
    image_amp = image_fft.amplitude
    
    # Positive Gaussian
    arr = make_gaussian(size=len(image.data), fwhm=fwhm_pos_gaus, center=None)
    nD_Gaussian = hs.signals.Signal2D(np.array(arr))
    #nD_Gaussian.plot()
    #plt.close()
    
    # negative gauss
    arr_neg = make_gaussian(size=len(image.data), fwhm=fwhm_neg_gaus, center=None)
    # Note that this step isn't actually neccessary for the computation, 
    #   we could just subtract when making the double gaussian below.
    #   However, we do it this way so that we can save a plot of the negative gaussian!
    #np_arr_neg = np_arr_neg
    nD_Gaussian_neg = hs.signals.Signal2D(np.array(arr_neg))
    #nD_Gaussian_neg.plot()
    
    
    neg_gauss_amplitude = 0.0 
    int_and_gauss_array = []

    for neg_gauss_amplitude in np.arange(0, 1+delta, delta):


        nD_Gaussian_neg_scaled = nD_Gaussian_neg*-1*neg_gauss_amplitude # NEED TO FIGURE out best number here!

        # Double Gaussian
        DGFilter = nD_Gaussian + nD_Gaussian_neg_scaled

        # Multiply the 2-D Gaussian with the FFT. This low pass filters the FFT.
        convolution = image_fft*DGFilter
        
        # Create the inverse FFT, which is your filtered image!
        convolution_ifft = convolution.ifft()
        #convolution_ifft.plot()
        minimum_intensity = convolution_ifft.data.min()
        maximum_intensity = convolution_ifft.data.max()
        
        int_and_gauss_array.append([neg_gauss_amplitude, minimum_intensity, maximum_intensity])

    np_arr_2 = np.array(int_and_gauss_array)
    x_axis = np_arr_2[:,0]
    y_axis = np_arr_2[:,1]
    zero_line = np.zeros_like(x_axis)
    idx = np.argwhere(np.diff(np.sign(zero_line-y_axis))).flatten()    
    neg_gauss_amplitude_calculated = x_axis[idx][0]    
    
    ''' Filtering the Image with the Chosen Negative Amplitude '''
    # positive gauss
    nD_Gaussian.axes_manager[0].scale = reciprocal_sampling
    nD_Gaussian.axes_manager[1].scale = reciprocal_sampling
    nD_Gaussian.axes_manager[0].units = '1/' + units
    nD_Gaussian.axes_manager[1].units = '1/' + units
    
    # negative gauss    
    nD_Gaussian_neg_used = nD_Gaussian_neg*-1*neg_gauss_amplitude_calculated # NEED TO FIGURE out best number here!
    nD_Gaussian_neg_used.axes_manager[0].scale = reciprocal_sampling
    nD_Gaussian_neg_used.axes_manager[1].scale = reciprocal_sampling
    nD_Gaussian_neg_used.axes_manager[0].units = '1/' + units
    nD_Gaussian_neg_used.axes_manager[1].units = '1/' + units    
    
    # Double Gaussian
    DGFilter_extra_dimension = nD_Gaussian + nD_Gaussian_neg_used
    DGFilter_extra_dimension.axes_manager[0].name='extra_dimension'
    
    '''how to change to just the 2 dimensiuons'''
    DGFilter = DGFilter_extra_dimension.sum('extra_dimension')
    
    DGFilter.axes_manager[0].scale = reciprocal_sampling
    DGFilter.axes_manager[1].scale = reciprocal_sampling
    DGFilter.axes_manager[0].units = '1/' + units
    DGFilter.axes_manager[1].units = '1/' + units
    
    # Multiply the 2-D Gaussian with the FFT. This filters the FFT.
    convolution = image_fft * DGFilter
    convolution_amp = convolution.amplitude
    
    image_filtered = convolution.ifft()
    
    image_filtered.axes_manager[0].scale = real_space_sampling
    image_filtered.axes_manager[1].scale = real_space_sampling
    image_filtered.axes_manager[0].units = units
    image_filtered.axes_manager[1].units = units
    
    
    if filename is not None:
        plt.figure()
        plt.plot(x_axis, y_axis)
        plt.plot(x_axis, zero_line)
        plt.plot(x_axis[idx], y_axis[idx], 'ro')
        plt.xlabel('Negative Gaussian Amplitude', fontsize = 16)
        plt.ylabel('Minimum Image Intensity', fontsize = 16)
        plt.title('Finding the Best DG Filter \n NG Amp = %' + filename %x_axis[idx][0], fontsize = 20)
        plt.legend(labels=('Neg. Gauss. Amp.', 'y = 0',), fontsize = 14)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.tight_layout()
        plt.show()
        plt.savefig(fname='minimising_negative_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=300)
        plt.close()
        
    
        #    if filename is not None:
        nD_Gaussian_neg_used.save('negative_gaussian_' + filename, overwrite=True)
        nD_Gaussian_neg_used.plot()
        plt.title('Negative Gaussian ' + filename, fontsize = 20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='negative_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=300, labels=False)
        plt.close()
    
        nD_Gaussian.save('positive_gaussian_' + filename, overwrite=True)
        nD_Gaussian.plot()
        plt.title('Positive Gaussian ' + filename, fontsize = 20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='positive_gaussian_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=300, labels=False)
        plt.close()
        

        DGFilter.save('double_gaussian_filter_' + filename, overwrite=True) # Save the .hspy file
        
        DGFilter.plot()
        plt.title('Double Gaussian Filter ' + filename, fontsize = 20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='double_gaussian_filter_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=300, labels=False)
        plt.close()
        
        
        
        convolution_amp.plot(norm='log')
        plt.title('FFT and Filter Convolved ' + filename, fontsize = 20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='FFT_and_filter_convolved_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=600, labels=False)
        plt.close()
    
        image_filtered.save('filtered_image_' + filename, overwrite=True)
        image_filtered.plot()
        plt.title('DG Filtered Image ' + filename, fontsize = 20)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fname='DG_filtered_image_' + filename + '.png',
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=600, labels=False)
        plt.close()
    
        ''' Saving the Variables for the image and filtered Image '''
        Filtering_Variables = collections.OrderedDict()
        Filtering_Variables['filename'] = [filename]
        Filtering_Variables['Image Size (nm)'] = [physical_image_size]
        Filtering_Variables['Image Size (pix)'] = [len(image.data)]
        Filtering_Variables['Real Space Sampling (nm/pix)'] = [real_space_sampling]
        Filtering_Variables['Reciprocal Space Sampling (1/nm/pix)'] = [reciprocal_sampling]
        Filtering_Variables['First Diffraction Ring (Diameter) (1/nm)'] = [d_inner]
        Filtering_Variables['Second Diffraction Ring (Diameter) (1/nm)'] = [d_outer]
        Filtering_Variables['First Diffraction Ring (Radius) (1/nm)'] = [reciprocal_d_inner]
        Filtering_Variables['Second Diffraction Ring (Radius) (1/nm)'] = [reciprocal_d_outer]
        Filtering_Variables['First Diffraction Ring (Radius) (pix)'] = [reciprocal_d_inner_pix]
        Filtering_Variables['Second Diffraction Ring (Radius) (pix)'] = [reciprocal_d_outer_pix]
        Filtering_Variables['Positive Gaussian FWHM (pix)'] = [fwhm_pos_gaus]
        Filtering_Variables['Negative Gaussian FWHM (pix)'] = [fwhm_neg_gaus]
        Filtering_Variables['Positive Gaussian Size'] = [len(image.data)]
        Filtering_Variables['Negative Gaussian Size'] = [len(image.data)]
        Filtering_Variables['Positive Gaussian FWHM'] = [fwhm_pos_gaus]
        Filtering_Variables['Negative Gaussian FWHM'] = [fwhm_neg_gaus]
        Filtering_Variables['Negative Gaussian Amplitude'] = [neg_gauss_amplitude_calculated]
        Filtering_Variables['Delta used for Calculation'] = [delta]
        Filtering_Variables_Table = pd.DataFrame(Filtering_Variables)
        Filtering_Variables_Table
        Filtering_Variables_Table.to_pickle('filtering_variables_table_' + filename + '.pkl')
        #Filtering_Variables_Table.to_csv('Filtering_Variables_Table.csv', sep=',', index=False)
    
    return(image_filtered)



def make_gaussian(size, fwhm, center):
        
    """ Make a square gaussian kernel.
    
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    
    arr=[] # output numpy array
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    arr.append(np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2))
    
    return(arr)




def count_atoms_in_sublattice_list(sublattice_list, filename=None):
    
    '''
    
    Returns
    -------
    
    Counter object
    
    
    >>> from collections import Counter
    >>> import atomap.api as am
    >>> import matplotlib.pyplot as plt
    >>> atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> sub1 = atom_lattice.sublattice_list[0]
    >>> sub2 = atom_lattice.sublattice_list[1]
    
    >>> for i in range(0, len(sub1.atom_list)):
    >>>     sub1.atom_list[i].elements = 'Ti_2'
    >>> for i in range(0, len(sub2.atom_list)):
    >>>     sub2.atom_list[i].elements = 'Cl_1'
    
    >>> added_atoms = count_atoms_in_sublattice_list(
    >>>     sublattice_list=[sub1, sub2],
    >>>     image_name=atom_lattice.name)

    Compare before and after
    >>> atom_lattice_before = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> no_added_atoms = count_atoms_in_sublattice_list(
    >>>     sublattice_list=atom_lattice_before.sublattice_list,
    >>>     image_name=atom_lattice_before.name)
    
    >>> if added_atoms == no_added_atoms:
    >>>     print('They are the same, you can stop refining')
    >>> else:
    >>>     print('They are not the same!')

    '''
    count_of_sublattice = Counter()
    for sublattice in sublattice_list:
        
        sublattice_info = print_sublattice_elements(sublattice)
        elements_in_sublattice = [atoms[0:1] for atoms in sublattice_info] # get just chemical info
        elements_in_sublattice = [y for x in elements_in_sublattice for y in x] # flatten to a list
        count_of_sublattice += Counter(elements_in_sublattice) # count each element

        #count_of_sublattice.most_common()

    if filename is not None:
        plt.figure()
        plt.scatter(x=count_of_sublattice.keys(), y=count_of_sublattice.values())
        plt.title('Elements in ' + filename, fontsize = 16)
        plt.xlabel('Elements', fontsize = 16)
        plt.ylabel('Count of Elements', fontsize = 16)
        plt.tight_layout()
        plt.savefig(fname='element_count_' + filename + '.png', 
                transparent=True, frameon=False, bbox_inches='tight', 
                pad_inches=None, dpi=300, labels=False)
        plt.close()
    else:
        pass
    
    return(count_of_sublattice)





def count_element_in_pandas_df(element, dataframe):
    '''
    >>> Mo_count = count_element_in_pandas_df(element='Mo', dataframe=df)
    '''
    count_of_element = Counter()
    
    for element_config in dataframe.columns:
#        print(element_config)
        if element in element_config:
            split_element = split_and_sort_element(element_config)
                
            for split in split_element:
                #split=split_element[1]
                if element in split[1]:
#                    print(element + ":" + str(split[2]*dataframe.loc[:, element_config]))
                    count_of_element += split[2]*dataframe.loc[:, element_config]
                
    return(count_of_element)



def count_all_individual_elements(individual_element_list, dataframe):
    
    '''
    >>> individual_element_list = ['Mo', 'S', 'Se']
    >>> element_count = count_all_individual_elements(individual_element_list, dataframe=df)
    >>> element_count
    '''
    
    element_count_dict = {}
    
    for element in individual_element_list:
        
        element_count = count_element_in_pandas_df(element=element, dataframe=dataframe)
        
        element_count_dict[element] = element_count
        
    return(element_count_dict)





def get_sublattice_intensity(sublattice, intensity_type='max', remove_background_method=None, 
                             background_sublattice=None, num_points=3, percent_to_nn=0.4, mask_radius=None):
    
    '''
    Finds the intensity for each atomic column using either max, mean, 
    min, total or all of them at once.
    
    The intensity values are taken from the area defined by 
    percent_to_nn.
    
    Results are stored in each Atom_Position object as 
    amplitude_max_intensity, amplitude_mean_intensity, 
    amplitude_min_intensity and/or amplitude_total_intensity 
    which can most easily be accessed through the sublattice object. 
    See the examples in get_atom_column_amplitude_max_intensity.
    
    Parameters
    ----------
    
    sublattice : sublattice object
        The sublattice whose intensities you are finding.
    intensity_type : string, default 'max'
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min', 'total' and
        'all'. 
    remove_background_method : string, default None
        Determines the method used to remove the background_sublattice
        intensities from the image. Options are 'average' and 'local'.
    background_sublattice : sublattice object, default None
        The sublattice used if remove_background_method is used.
    num_points : int, default 3
        If remove_background_method='local', num_points is the number 
        of nearest neighbour values averaged from background_sublattice
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    
    Returns
    -------
    2D numpy array

    Examples
    --------
    
    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_all = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='all', 
                                                   remove_background_method=None,
                                                   background_sublattice=None)

    >>> intensities_total = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='total', 
                                                   remove_background_method=None,
                                                   background_sublattice=None)
    
    >>> intensities_total_local = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='total', 
                                                   remove_background_method='local',
                                                   background_sublattice=sublattice)

    >>> intensities_max_average = get_sublattice_intensity(sublattice=sublattice, 
                                                   intensity_type='max', 
                                                   remove_background_method='average',
                                                   background_sublattice=sublattice)
        
    '''
    if percent_to_nn is not None:
        sublattice.find_nearest_neighbors()
    else:
        pass

    if remove_background_method == None and background_sublattice == None:
        if intensity_type == 'all':
            sublattice.get_atom_column_amplitude_max_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_max_intensity_list = []
            sublattice_max_intensity_list.append(sublattice.atom_amplitude_max_intensity)
            sublattice_max_intensity_list = sublattice_max_intensity_list[0]
            max_intensities = np.array(sublattice_max_intensity_list)
            
            sublattice.get_atom_column_amplitude_mean_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_mean_intensity_list = []
            sublattice_mean_intensity_list.append(sublattice.atom_amplitude_mean_intensity)
            sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
            mean_intensities = np.array(sublattice_mean_intensity_list)
            
            sublattice.get_atom_column_amplitude_min_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_min_intensity_list = []
            sublattice_min_intensity_list.append(sublattice.atom_amplitude_min_intensity)
            sublattice_min_intensity_list = sublattice_min_intensity_list[0]
            min_intensities = np.array(sublattice_min_intensity_list)
            
            sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_total_intensity_list = []
            sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
            sublattice_total_intensity_list = sublattice_total_intensity_list[0]
            total_intensities = np.array(sublattice_total_intensity_list)
            
            
#            sublattice_total_intensity_list, _, _ = sublattice.integrate_column_intensity() # maxradius should be changed to percent_to_nn!
#            total_intensities = np.array(sublattice_total_intensity_list)
            
            sublattice_intensities = np.column_stack((max_intensities, mean_intensities, min_intensities, total_intensities))
            return(sublattice_intensities)
          #  return max_intensities, mean_intensities, min_intensities, total_intensities
        
        elif intensity_type == 'max':
            sublattice.get_atom_column_amplitude_max_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_max_intensity_list = []
            sublattice_max_intensity_list.append(sublattice.atom_amplitude_max_intensity)
            sublattice_max_intensity_list = sublattice_max_intensity_list[0]
            max_intensities = np.array(sublattice_max_intensity_list)
            
            return(max_intensities)
        
        elif intensity_type == 'mean':
            sublattice.get_atom_column_amplitude_mean_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_mean_intensity_list = []
            sublattice_mean_intensity_list.append(sublattice.atom_amplitude_mean_intensity)
            sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
            mean_intensities = np.array(sublattice_mean_intensity_list)
            
            return(mean_intensities)
        
        elif intensity_type == 'min':
            sublattice.get_atom_column_amplitude_min_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_min_intensity_list = []
            sublattice_min_intensity_list.append(sublattice.atom_amplitude_min_intensity)
            sublattice_min_intensity_list = sublattice_min_intensity_list[0]
            min_intensities = np.array(sublattice_min_intensity_list)
            
            return(min_intensities)
        
        elif intensity_type == 'total':
            sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
            sublattice_total_intensity_list = []
            sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
            sublattice_total_intensity_list = sublattice_total_intensity_list[0]
            total_intensities = np.array(sublattice_total_intensity_list)
            
#            sublattice_total_intensity_list, _, _ = sublattice.integrate_column_intensity()
#            total_intensities = np.array(sublattice_total_intensity_list)
            return(total_intensities)
        
        else:
            raise ValueError('You must choose an intensity_type')
            
    elif remove_background_method == 'average':
        
        sublattice_intensity_list_average_bksubtracted = remove_average_background(sublattice=sublattice, 
                                                                               background_sublattice=background_sublattice, 
                                                                               intensity_type=intensity_type, 
                                                                               percent_to_nn=percent_to_nn,
                                                                               mask_radius=mask_radius)
        return(sublattice_intensity_list_average_bksubtracted)

    elif remove_background_method == 'local':
        
        sublattice_intensity_list_local_bksubtracted = remove_local_background(sublattice=sublattice,
                                                                            background_sublattice=background_sublattice,
                                                                            intensity_type=intensity_type,
                                                                            num_points=num_points,
                                                                            percent_to_nn=percent_to_nn,
                                                                            mask_radius=mask_radius)
        return(sublattice_intensity_list_local_bksubtracted)
    
    else:
        pass



def get_pixel_count_from_image_slice(
        self,
        image_data,
        percent_to_nn=0.40):
    """
    Fid the number of pixels in an area when calling
    _get_image_slice_around_atom()
    
    Parameters
    ----------
    
    image_data : Numpy 2D array
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    
    Returns
    -------
    The number of pixels in the image_slice
    
    Examples
    --------
    
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> atom0 = sublattice.atom_list[0]
    >>> pixel_count = atom0.get_pixel_count_from_image_slice(sublattice.image)
    
    """
    closest_neighbor = self.get_closest_neighbor()

    slice_size = closest_neighbor * percent_to_nn * 2
    data_slice, x0, y0 = self._get_image_slice_around_atom(
            image_data, slice_size)
    
    pixel_count = len(data_slice[0]) * len(data_slice[0]) 
    
    return(pixel_count)
    

def remove_average_background(sublattice, intensity_type,
                              background_sublattice, percent_to_nn=0.40,
                              mask_radius=None):
    
    '''
    Remove the average background from a sublattice intensity using
    a background sublattice. 
    
    Parameters
    ----------
    
    sublattice : sublattice object
        The sublattice whose intensities are of interest.
    intensity_type : string
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min' and 'all'. 
    background_sublattice : sublattice object
        The sublattice used to find the average background.
    percent_to_nn : float, default 0.4
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    
    Returns
    -------
    2D numpy array
    
    Examples
    --------
    
    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_all = remove_average_background(sublattice, intensity_type='all',
                                                    background_sublattice=sublattice)
    >>> intensities_max = remove_average_background(sublattice, intensity_type='max',
                                                background_sublattice=sublattice)

    '''
    background_sublattice.find_nearest_neighbors()
    background_sublattice.get_atom_column_amplitude_min_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    background_sublattice_min = []
    background_sublattice_min.append(background_sublattice.atom_amplitude_min_intensity)
    background_sublattice_mean_of_min = np.mean(background_sublattice_min)

    if intensity_type == 'all':
        sublattice.get_atom_column_amplitude_max_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]
        max_intensities = np.array(sublattice_max_intensity_list) - background_sublattice_mean_of_min
        
        sublattice.get_atom_column_amplitude_mean_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
        mean_intensities = np.array(sublattice_mean_intensity_list) - background_sublattice_mean_of_min
        
        sublattice.get_atom_column_amplitude_min_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_min_intensity_list = []
        sublattice_min_intensity_list.append(sublattice.atom_amplitude_min_intensity)
        sublattice_min_intensity_list = sublattice_min_intensity_list[0]
        min_intensities = np.array(sublattice_min_intensity_list) - background_sublattice_mean_of_min

#        sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn)
#        sublattice_total_intensity_list = []
#        sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
#        sublattice_total_intensity_list = sublattice_total_intensity_list[0]
#        total_intensities = np.array(sublattice_total_intensity_list) - background_sublattice_mean_of_min
        
        sublattice_intensities = np.column_stack((max_intensities, mean_intensities, min_intensities))
        return sublattice_intensities
    
    elif intensity_type == 'max':
        sublattice.get_atom_column_amplitude_max_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]
        max_intensities = np.array(sublattice_max_intensity_list) - background_sublattice_mean_of_min
        
        return max_intensities
    
    elif intensity_type == 'mean':
        sublattice.get_atom_column_amplitude_mean_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]
        mean_intensities = np.array(sublattice_mean_intensity_list) - background_sublattice_mean_of_min
        
        return mean_intensities
    
    elif intensity_type == 'min':
        sublattice.get_atom_column_amplitude_min_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_min_intensity_list = []
        sublattice_min_intensity_list.append(sublattice.atom_amplitude_min_intensity)
        sublattice_min_intensity_list = sublattice_min_intensity_list[0]
        min_intensities = np.array(sublattice_min_intensity_list) - background_sublattice_mean_of_min
        
        return min_intensities
    
    elif intensity_type == 'total':
        raise ValueError("Average background removal doesn't work with total intensity, yet")
#        sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn)
#        sublattice_total_intensity_list = []
#        sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
#        sublattice_total_intensity_list = sublattice_total_intensity_list[0]
#        total_intensities = np.array(sublattice_total_intensity_list) - background_sublattice_mean_of_min
#
#        return total_intensities

    else:
        pass




def remove_local_background(sublattice, background_sublattice, intensity_type,
                            num_points=3, percent_to_nn=0.40, mask_radius=None):

    '''
    Remove the local background from a sublattice intensity using
    a background sublattice. 
    
    Parameters
    ----------
    
    sublattice : sublattice object
        The sublattice whose intensities are of interest.
    intensity_type : string
        Determines the method used to find the sublattice intensities.
        The available methods are 'max', 'mean', 'min', 'total' and 'all'. 
    background_sublattice : sublattice object
        The sublattice used to find the local backgrounds.
    num_points : int, default 3
        The number of nearest neighbour values averaged from 
        background_sublattice
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    
    Returns
    -------
    2D numpy array
    
    Examples
    --------
    
    >>> import numpy as np
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.find_nearest_neighbors()
    >>> intensities_total = remove_local_background(sublattice, intensity_type='total',
                                                    background_sublattice=sublattice)
    >>> intensities_max = remove_local_background(sublattice, intensity_type='max',
                                                  background_sublattice=sublattice)

    '''
    # get background_sublattice intensity list
    
    if percent_to_nn is not None:
        sublattice.find_nearest_neighbors()
        background_sublattice.find_nearest_neighbors()
    else:
        pass

    background_sublattice.get_atom_column_amplitude_min_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    background_sublattice_min_intensity_list = []
    background_sublattice_min_intensity_list.append(background_sublattice.atom_amplitude_min_intensity)
    background_sublattice_min_intensity_list = background_sublattice_min_intensity_list[0]
    if intensity_type == 'all':
        raise ValueError("All intensities has not yet been implemented. Use max, mean or total instead")
    
    if num_points == 0:
        raise ValueError("num_points cannot be 0 if you wish to locally remove the background")
    
    if intensity_type == 'max':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(background_sublattice.atom_positions).T
        
        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_max_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_max_intensity_list = []
        sublattice_max_intensity_list.append(sublattice.atom_amplitude_max_intensity)
        sublattice_max_intensity_list = sublattice_max_intensity_list[0]
        
        # create list which will be output
        # therefore the original data is not changed!
        sublattice_max_intensity_list_bksubtracted = []
        
        # for each sublattice atom position, calculate the nearest 
        #   background_sublattice atom positions. 
        for p in range(0, len(sublattice_atom_pos)):
            
            xy_distances = background_sublattice_atom_pos - sublattice_atom_pos[p]
            
            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt( (xy_distances[i][0]**2) + (xy_distances[i][1]**2) )
                vector_array.append(vector)
            #convert to numpy array
            vector_array = np.array(vector_array)
                    
            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, k)[:k])
            # sum the chosen intensities and find the mean (or median - add this)
            local_background=0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]
            
            local_background_mean = local_background / k
            
            # subtract this mean local background intensity from the sublattice 
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_max_intensity[p]) - \
                                                    np.array(local_background_mean)
            
            sublattice_max_intensity_list_bksubtracted.append([sublattice_bksubtracted_atom])
            
        sublattice_max_intensity_list_bksubtracted = np.array(sublattice_max_intensity_list_bksubtracted)
            
        return(sublattice_max_intensity_list_bksubtracted[:,0])

    elif intensity_type == 'mean':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(background_sublattice.atom_positions).T
        
        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_mean_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_mean_intensity_list = []
        sublattice_mean_intensity_list.append(sublattice.atom_amplitude_mean_intensity)
        sublattice_mean_intensity_list = sublattice_mean_intensity_list[0]

        # create list which will be output
        # therefore the original data is not changed!
        sublattice_mean_intensity_list_bksubtracted = []
        
        # for each sublattice atom position, calculate the nearest 
        #   background_sublattice atom positions. 
        for p in range(0, len(sublattice_atom_pos)):
            
            xy_distances = background_sublattice_atom_pos - sublattice_atom_pos[p]
            
            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt( (xy_distances[i][0]**2) + (xy_distances[i][1]**2) )
                vector_array.append(vector)
            #convert to numpy array
            vector_array = np.array(vector_array)
                    
            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, k)[:k])
            # sum the chosen intensities and find the mean (or median - add this)
            local_background=0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]
            
            local_background_mean = local_background / k
            
            # subtract this mean local background intensity from the sublattice 
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_mean_intensity[p]) - \
                                                    local_background_mean
            
            sublattice_mean_intensity_list_bksubtracted.append([sublattice_bksubtracted_atom])
            
        sublattice_mean_intensity_list_bksubtracted = np.array(sublattice_mean_intensity_list_bksubtracted)
            
        return(sublattice_mean_intensity_list_bksubtracted[:,0])

    elif intensity_type == 'total':
        # get list of sublattice and background_sublattice atom positions
        # np.array().T will not be needed in newer versions of atomap
        sublattice_atom_pos = np.array(sublattice.atom_positions).T
        background_sublattice_atom_pos = np.array(background_sublattice.atom_positions).T
        
        # get sublattice intensity list
        # could change to my function, which allows choice of intensity type
        sublattice.get_atom_column_amplitude_total_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        sublattice_total_intensity_list = []
        sublattice_total_intensity_list.append(sublattice.atom_amplitude_total_intensity)
        sublattice_total_intensity_list = sublattice_total_intensity_list[0]

        # create list which will be output
        # therefore the original data is not changed!
        sublattice_total_intensity_list_bksubtracted = []
        
        # for each sublattice atom position, calculate the nearest 
        #   background_sublattice atom positions. 
        for p in range(0, len(sublattice_atom_pos)):

            xy_distances = background_sublattice_atom_pos - sublattice_atom_pos[p]
            
            # put all distances in this array with this loop
            vector_array = []
            for i in range(0, len(xy_distances)):
                # get distance from sublattice position to every background_sublattice position
                vector = np.sqrt( (xy_distances[i][0]**2) + (xy_distances[i][1]**2) )
                vector_array.append(vector)
            #convert to numpy array
            vector_array = np.array(vector_array)
                    
            # sort through the vector_array and find the 1st to kth smallest distance and find the
            #   corressponding index
            # num_points is the number of nearest points from which the background will be averaged
            k = num_points
            min_indices = list(np.argpartition(vector_array, range(k))[:k])
            # if you want the values rather than the indices, use:
            # vector_array[np.argpartition(vector_array, range(k))[:k]]
            # sum the chosen intensities and find the total (or median - add this)
            local_background=0
            for index in min_indices:
                local_background += background_sublattice.atom_amplitude_min_intensity[index]
            
            local_background_mean = local_background / k
            
            # for summing pixels around atom
            if mask_radius is None:
                pixel_count_in_region = get_pixel_count_from_image_slice(sublattice.atom_list[p], 
                                                                         sublattice.image,
                                                                         percent_to_nn)
            elif mask_radius is not None:
                mask = _make_circular_mask(centerX=sublattice.atom_list[p].pixel_x, 
                                            centerY=sublattice.atom_list[p].pixel_y,
                                            imageSizeX=sublattice.image.shape[0],
                                            imageSizeY=sublattice.image.shape[1],
                                            radius=mask_radius)
            
                pixel_count_in_region = len(sublattice.image[mask])

            local_background_mean_summed = pixel_count_in_region * local_background_mean

            # subtract this mean local background intensity from the sublattice 
            #   atom position intensity
            # indexing here is the loop digit p
            sublattice_bksubtracted_atom = np.array(sublattice.atom_amplitude_total_intensity[p]) - \
                                                    local_background_mean_summed
            
            sublattice_total_intensity_list_bksubtracted.append([sublattice_bksubtracted_atom])
            
        sublattice_total_intensity_list_bksubtracted = np.array(sublattice_total_intensity_list_bksubtracted)
            
        return(sublattice_total_intensity_list_bksubtracted[:,0])
    
    else:
        raise ValueError("You must choose a valid intensity_type. Try max, mean or total")




def split_and_sort_element(element, split_symbol=['_', '.']):
    
    '''
    Extracts info from input atomic column element configuration
    Split an element and its count, then sort the element for 
    use with other functions.
    
    Parameters
    ----------
    
    element : string, default None
        element species and count must be separated by the
        first string in the split_symbol list.
        separate elements must be separated by the second 
        string in the split_symbol list.
    split_symbol : list of strings, default ['_', '.']
        The symbols used to split the element into its name
        and count.
        The first string '_' is used to split the name and count 
        of the element.
        The second string is used to split different elements in
        an atomic column configuration.
    
    Returns
    -------
    list with element_split, element_name, element_count, and
    element_atomic_number
    
    Examples
    --------
    >>> import periodictable as pt
    >>> single_element = split_and_sort_element(element='S_1')
    >>> complex_element = split_and_sort_element(element='O_6.Mo_3.Ti_5')

    '''
    splitting_info = []
    
    if '.' in element:
    #if len(split_symbol) > 1:
        
        if split_symbol[1] == '.':
    
            stacking_element = element.split(split_symbol[1])
            for i in range(0, len(stacking_element)):
                element_split = stacking_element[i].split(split_symbol[0])
                element_name = element_split[0]
                element_count = int(element_split[1])
                element_atomic_number = pt.elements.symbol(element_name).number
                splitting_info.append([element_split, element_name, element_count, element_atomic_number])
        else:
            raise ValueError("To split a stacked element use: split_symbol=['_', '.']")

    #elif len(split_symbol) == 1:
    elif '.' not in element:
        element_split = element.split(split_symbol[0])
        element_name = element_split[0]
        element_count = int(element_split[1])
        element_atomic_number = pt.elements.symbol(element_name).number
        splitting_info.append([element_split, element_name, element_count, element_atomic_number])

    else:
        raise ValueError(
                "You must include a split_symbol. Use '_' to separate element and count. Use '.' to separate elements in the same xy position")

        
    return(splitting_info)



def scaling_z_contrast(numerator_sublattice, numerator_element, 
                       denominator_sublattice, denominator_element,
                       intensity_type, method, remove_background_method,
                       background_sublattice, num_points,
                       percent_to_nn=0.40, mask_radius=None, split_symbol='_'):
    # Make sure that the intensity_type input has been chosen. Could
    #   make this more flexible, so that 'all' could be calculated in one go
    #   simple loop should do that.
    if intensity_type == 'all':
        TypeError 
        print('intensity_type must be "max", "mean", or "min"')
    else:
        pass

    sublattice0 = numerator_sublattice
    sublattice1 = denominator_sublattice
    
    # use the get_sublattice_intensity() function to get the mean/mode intensities of
    #   each sublattice
    if type(mask_radius) is list:
        sublattice0_intensity = get_sublattice_intensity(
                sublattice0, intensity_type, remove_background_method, background_sublattice, 
                num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius[0])
        
        sublattice1_intensity = get_sublattice_intensity(
                sublattice1, intensity_type, remove_background_method, background_sublattice, 
                num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius[1])
    else:
        sublattice0_intensity = get_sublattice_intensity(
                sublattice0, intensity_type, remove_background_method, background_sublattice, 
                num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius)
        
        sublattice1_intensity = get_sublattice_intensity(
                sublattice1, intensity_type, remove_background_method, background_sublattice, 
                num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    
    
    if method == 'mean':
        sublattice0_intensity_method = np.mean(sublattice0_intensity)
        sublattice1_intensity_method = np.mean(sublattice1_intensity)
    elif method =='mode':
        sublattice0_intensity_method = scipy.stats.mode(np.round(sublattice0_intensity, decimals=2))[0][0]
        sublattice1_intensity_method = scipy.stats.mode(np.round(sublattice1_intensity, decimals=2))[0][0]
    
    # Calculate the scaling ratio and exponent for Z-contrast images
    scaling_ratio = sublattice0_intensity_method / sublattice1_intensity_method
    
    numerator_element_split = split_and_sort_element(element=numerator_element, split_symbol=split_symbol)
    denominator_element_split = split_and_sort_element(element=denominator_element, split_symbol=split_symbol)
    
    if len(numerator_element_split) == 1:
        scaling_exponent = log(denominator_element_split[0][2]*scaling_ratio) / (log(numerator_element_split[0][3]) - log(denominator_element_split[0][3]))
    else:
        pass # need to include more complicated equation to deal with multiple elements as the e.g., numerator
    
    return scaling_ratio, scaling_exponent, sublattice0_intensity_method, sublattice1_intensity_method




def find_middle_and_edge_intensities(sublattice, element_list, standard_element, scaling_exponent, split_symbol=['_', '.']):

    # Create a list which represents the peak points of the
    #   intensity distribution for each atom
    middle_intensity_list = []
    
    if isinstance(standard_element, str) == True:
        standard_split = split_and_sort_element(element=standard_element, split_symbol=split_symbol)
        standard_element_value = 0.0
        for i  in range(0, len(standard_split)):
            standard_element_value += standard_split[i][2]*(pow(standard_split[i][3], scaling_exponent))
    else:
        standard_element_value = standard_element
    # find the values for element_lists
    for i in range(0, len(element_list)):
        element_split = split_and_sort_element(element=element_list[i], split_symbol=split_symbol)
        element_value=0.0
        for p in range(0, len(element_split)):
            element_value += element_split[p][2]*(pow(element_split[p][3], scaling_exponent))
        atom = element_value / standard_element_value
        middle_intensity_list.append(atom)
        
    middle_intensity_list.sort()

    limit_intensity_list = [0.0]
    for i in range(0, len(middle_intensity_list)-1):
        limit = (middle_intensity_list[i] + middle_intensity_list[i+1])/2
        limit_intensity_list.append(limit)
            
    if len(limit_intensity_list) <= len(middle_intensity_list):
        max_limit = middle_intensity_list[-1] + (middle_intensity_list[-1]-limit_intensity_list[-1])
        limit_intensity_list.append(max_limit)
    else:
        pass

    return middle_intensity_list, limit_intensity_list




# choosing the percent_to_nn for this seems dodgy atm...
def find_middle_and_edge_intensities_for_background(elements_from_sub1, 
                                                    elements_from_sub2, 
                                                    sub1_mode, 
                                                    sub2_mode,
                                                    element_list_sub1, 
                                                    element_list_sub2, 
                                                    middle_intensity_list_sub1, 
                                                    middle_intensity_list_sub2):
    
    middle_intensity_list_background = [0.0]
    
    # it is neccessary to scale the background_sublattice intensities here already because otherwise
    #   the background_sublattice has no reference atom to base its mode intensity on. eg. in MoS2, first sub has Mo
    #   as a standard atom, second sub has S2 as a standard reference.
    
    for i in elements_from_sub1:
        index = element_list_sub1.index(i)
        middle = middle_intensity_list_sub1[index] * sub1_mode
        middle_intensity_list_background.append(middle)
        
    for i in elements_from_sub2:
        index = element_list_sub2.index(i)
        middle = middle_intensity_list_sub2[index] * sub2_mode
        middle_intensity_list_background.append(middle)

    middle_intensity_list_background.sort()
    
    limit_intensity_list_background = [0.0]
    for i in range(0, len(middle_intensity_list_background)-1):
        limit = (middle_intensity_list_background[i] + middle_intensity_list_background[i+1])/2
        limit_intensity_list_background.append(limit)
            
    if len(limit_intensity_list_background) <= len(middle_intensity_list_background):
        max_limit = middle_intensity_list_background[-1] + (middle_intensity_list_background[-1]-limit_intensity_list_background[-1])
        limit_intensity_list_background.append(max_limit)
    else:
        pass
    
    return middle_intensity_list_background, limit_intensity_list_background



def sort_sublattice_intensities(sublattice, 
                                intensity_type, 
                                middle_intensity_list, 
                                limit_intensity_list,
                                element_list, 
                                method, 
                                remove_background_method,
                                background_sublattice,
                                num_points,
                                intensity_list_real=False,
                                percent_to_nn=0.40, mask_radius=None):

    # intensity_list_real is asking whether the intensity values in your intensity_list for the current sublattice
    #   are scaled. Scaled meaning already multiplied by the mean or mode of said sublattice.
    #   Set to Tru for background sublattices. For more details see "find_middle_and_edge_intensities_for_background()"
    #   You can see that the outputted lists are scaled by the mean or mode, whereas in 
    #   "find_middle_and_edge_intensities()", they are not.
    
    
    sublattice_intensity = get_sublattice_intensity(
            sublattice, intensity_type, remove_background_method, background_sublattice,
            num_points, percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    
    for i in sublattice_intensity:
        if i<0:
            i=0.0000000001
            #raise ValueError("You have negative intensity. Bad Vibes")
    
    if intensity_list_real == False:

        if method == 'mean':
            scalar = np.mean(sublattice_intensity)
        elif method =='mode':
            scalar = scipy.stats.mode(np.round(sublattice_intensity, decimals=2))[0][0]
        
        if len(element_list) != len(middle_intensity_list):
            raise ValueError ('elements_list length does not equal middle_intensity_list length')
        else:
            pass
        
        elements_of_sublattice = []
        for p in range(0, (len(limit_intensity_list)-1)):
            for i in range(0, len(sublattice.atom_list)):
                if limit_intensity_list[p]*scalar < sublattice_intensity[i] < limit_intensity_list[p+1]*scalar:
                    sublattice.atom_list[i].elements = element_list[p]
                    elements_of_sublattice.append(sublattice.atom_list[i].elements)
                    
    elif intensity_list_real == True:
        if len(element_list) != len(middle_intensity_list):
            raise ValueError ('elements_list length does not equal middle_intensity_list length')
        else:
            pass
        
        elements_of_sublattice = []
        for p in range(0, (len(limit_intensity_list)-1)):
            for i in range(0, len(sublattice.atom_list)):
                if limit_intensity_list[p] < sublattice_intensity[i] < limit_intensity_list[p+1]:
                    sublattice.atom_list[i].elements = element_list[p]
                    elements_of_sublattice.append(sublattice.atom_list[i].elements)
    
    for i in range(0, len(sublattice.atom_list)):
        if sublattice.atom_list[i].elements == '':
            sublattice.atom_list[i].elements = 'H_0'
            elements_of_sublattice.append(sublattice.atom_list[i].elements)
        else:
            pass

    
    return elements_of_sublattice




def assign_z_height(sublattice, lattice_type, material):
    for i in range(0, len(sublattice.atom_list)):
        if material == 'mose2_one_layer':
            if lattice_type == 'chalcogen':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = '0.758'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.242, 0.758'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = '0.242, 0.758, 0.9'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = '0.242, 0.758'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = '0.242, 0.758, 0.9'
                else:
                    sublattice.atom_list[i].z_height = '0.758'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'transition_metal':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height ='0.5'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.5, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = '0.5, 0.95, 1'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = '0.5, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = '0.5, 0.95, 1'
                else:
                    sublattice.atom_list[i].z_height = '0.5'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'background':
                #if sublattice.atom_list[i].elements == 'H_0' or sublattice.atom_list[i].elements == 'vacancy':
                sublattice.atom_list[i].z_height = '0.95'
                #elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                #    sublattice.atom_list[i].z_height = [0.9]
                #else:
                #    sublattice.atom_list[i].z_height = []
                
            else:
                print("You must include a suitable lattice_type. This feature is limited")
                
                
        if material == 'mos2_one_layer':
            if lattice_type == 'chalcogen':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = '0.757' # from L Mattheis, PRB, 1973
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.242, 0.757'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height ='0.242, 0.757, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = '0.242, 0.757'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = '0.242, 0.757, 0.95'
                else:
                    sublattice.atom_list[i].z_height = '0.757'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'transition_metal':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = '0.5'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.5, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 2:
                    sublattice.atom_list[i].z_height = '0.5, 0.95, 1'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[1][2] == 1:
                    sublattice.atom_list[i].z_height = '0.5, 0.95'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] > 1:
                    sublattice.atom_list[i].z_height = '0.5, 0.95, 1'
                else:
                    sublattice.atom_list[i].z_height = '0.5'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'background':
                #if sublattice.atom_list[i].elements == 'H_0' or sublattice.atom_list[i].elements == 'vacancy':
                sublattice.atom_list[i].z_height = '0.95'
                #elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                #    sublattice.atom_list[i].z_height = [0.9]
                #else:
                #    sublattice.atom_list[i].z_height = []
                
            else:
                print("You must include a suitable lattice_type. This feature is limited")

        if material == 'mos2_two_layer':
            if lattice_type == 'chalcogen':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = '0.3725' # from L Mattheis, PRB, 1973
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 2:
                    sublattice.atom_list[i].z_height = '0.1275, 0.3725'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2:
                    sublattice.atom_list[i].z_height = '0.1275, 0.3725, 0.75'
                
                else:
                    sublattice.atom_list[i].z_height = '0.95'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'transition_metal':
                if len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                    sublattice.atom_list[i].z_height = '0.25'
                elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 2:
                    sublattice.atom_list[i].z_height = '0.25, 0.6275, 0.8725'

                else:
                    sublattice.atom_list[i].z_height = '0.95'
                    #raise ValueError("z_height is limited to only a handful of positions")
            elif lattice_type == 'background':
                #if sublattice.atom_list[i].elements == 'H_0' or sublattice.atom_list[i].elements == 'vacancy':
                sublattice.atom_list[i].z_height = '0.95'
                #elif len(split_and_sort_element(element=sublattice.atom_list[i].elements)) == 1 and split_and_sort_element(element=sublattice.atom_list[i].elements)[0][2] == 1:
                #    sublattice.atom_list[i].z_height = [0.9]
                #else:
                #    sublattice.atom_list[i].z_height = []
                
            else:
                print("You must include a suitable lattice_type. This feature is limited")




def print_sublattice_elements(sublattice):
    elements_of_sublattice = []
    for i in range(0, len(sublattice.atom_list)):
        sublattice.atom_list[i].elements
        sublattice.atom_list[i].z_height # etc.
        elements_of_sublattice.append([sublattice.atom_list[i].elements, 
                                       sublattice.atom_list[i].z_height, 
                                       sublattice.atom_amplitude_max_intensity[i],
                                       sublattice.atom_amplitude_mean_intensity[i],
                                       sublattice.atom_amplitude_min_intensity[i],
                                       sublattice.atom_amplitude_total_intensity[i]
                                       ])
    return elements_of_sublattice



def create_dataframe_for_cif(sublattice_list, element_list):
        """
        Parameters
        ----------
        
        """
        dfObj = pd.DataFrame(columns = ['_atom_site_label', 
                                        '_atom_site_occupancy', 
                                        '_atom_site_fract_x', 
                                        '_atom_site_fract_y',
                                        '_atom_site_fract_z',
                                        '_atom_site_adp_type',
                                        '_atom_site_B_iso_or_equiv',
                                        '_atom_site_type_symbol'])        

        # Start with the first sublattice in the list of sublattices given
        for sublattice in sublattice_list:
            # Go through each atom_list index one by one
            for i in range (0, len(sublattice.atom_list)):
                #check if the element is in the given element list
                if sublattice.atom_list[i].elements in element_list:
                    #this loop cycles through the length of the split element eg, 2 for 'Se_1.S_1' and
                    #   outputs an atom label and z_height for each
                    for k in range(0, len(split_and_sort_element(sublattice.atom_list[i].elements))):
                        if split_and_sort_element(sublattice.atom_list[i].elements)[k][2] >= 1:
                            atom_label = split_and_sort_element(sublattice.atom_list[i].elements)[k][1]
                            
                            if "," in sublattice.atom_list[i].z_height:
                                atom_z_height = float(sublattice.atom_list[i].z_height.split(",")[k])
                            else:
                                atom_z_height = float(sublattice.atom_list[i].z_height)
                            
                            # this loop checks the number of atoms that share 
                            # the same x and y coords.
                            for p in range(0, split_and_sort_element(sublattice.atom_list[i].elements)[k][2]):#len(sublattice.atom_list[i].z_height)):

                                if "," in sublattice.atom_list[i].z_height and split_and_sort_element(sublattice.atom_list[i].elements)[k][2] > 1:
                                    atom_z_height = float(sublattice.atom_list[i].z_height.split(",")[p])
                                else:
                                    pass

                                dfObj = dfObj.append({'_atom_site_label' : atom_label,
                                                      '_atom_site_occupancy' : 1.0, 
                                                      '_atom_site_fract_x' : format( sublattice.atom_list[i].pixel_x/len(sublattice.image[0,:]), '.6f'), 
                                                      '_atom_site_fract_y' : format( (len(sublattice.image[:,0])-sublattice.atom_list[i].pixel_y)/len(sublattice.image[:,0]), '.6f'),
                                                      '_atom_site_fract_z' : format( atom_z_height, '.6f'), #great touch
                                                      '_atom_site_adp_type' : 'Biso',
                                                      '_atom_site_B_iso_or_equiv' : format(1.0, '.6f'),
                                                      '_atom_site_type_symbol' : atom_label},
                                                    ignore_index=True)          #insert row

                                #value += split_and_sort_element(sublattice.atom_list[i].elements)[k][2]
        # need an option to save to the cuurent directory should be easy
#        dfObj.to_pickle('atom_lattice_atom_position_table.pkl')
#        dfObj.to_csv('atom_lattice_atom_position_table.csv', sep=',', index=False)
        return dfObj




def write_cif_from_dataframe(dataframe,
                             filename,
                             chemical_name_common, 
                             cell_length_a, 
                             cell_length_b, 
                             cell_length_c, 
                             cell_angle_alpha = 90, 
                             cell_angle_beta = 90, 
                             cell_angle_gamma = 90, 
                             space_group_name_H_M_alt = 'P 1', 
                             space_group_IT_number = 1):
        """
        Parameters
        ----------
        dataframe : dataframe object
            pandas dataframe containing rows of atomic position information
        chemical_name_common : string
            name of chemical
        cell_length_a, _cell_length_b, _cell_length_c : float
            lattice dimensions in angstrom
        cell_angle_alpha, cell_angle_beta, cell_angle_gamma : float
            lattice angles in degrees
        space_group_name_H-M_alt : string
            space group name
        space_group_IT_number : float
            
        
        """
    
        #create cif
        cif_file = CifFile.CifFile()
        
        #create block to hold values
        params = CifFile.CifBlock()
        
        cif_file['block_1'] = params
        
        #set unit cell properties
        params.AddItem('_chemical_name_common', chemical_name_common)
        params.AddItem('_cell_length_a', format(cell_length_a, '.6f'))
        params.AddItem('_cell_length_b', format(cell_length_b, '.6f'))
        params.AddItem('_cell_length_c', format(cell_length_c, '.6f'))
        params.AddItem('_cell_angle_alpha', cell_angle_alpha) 
        params.AddItem('_cell_angle_beta', cell_angle_beta)
        params.AddItem('_cell_angle_gamma', cell_angle_gamma)
        params.AddItem('_space_group_name_H-M_alt', space_group_name_H_M_alt)
        params.AddItem('_space_group_IT_number', space_group_IT_number)
        
        #loop 1 - _space_group_symop_operation_xyz
        params.AddCifItem(([['_space_group_symop_operation_xyz']], 
                           
                           [[['x, y, z']]]))
        
                           #[[['x, y, z', 
                              #'x, y, z+1/2']]]))
        
        #loop 2 - individual atom positions and properties
        params.AddCifItem(([['_atom_site_label',
                             '_atom_site_occupancy',
                             '_atom_site_fract_x',
                             '_atom_site_fract_y',
                             '_atom_site_fract_z',
                             '_atom_site_adp_type', 
                             '_atom_site_B_iso_or_equiv',
                             '_atom_site_type_symbol']],
        
                            [[dataframe['_atom_site_label'], 
                              dataframe['_atom_site_occupancy'], 
                              dataframe['_atom_site_fract_x'], 
                              dataframe['_atom_site_fract_y'], 
                              dataframe['_atom_site_fract_z'], 
                              dataframe['_atom_site_adp_type'], 
                              dataframe['_atom_site_B_iso_or_equiv'], 
                              dataframe['_atom_site_type_symbol']]]))
        
        #put it all together in a cif
        outFile = open(filename+".cif","w")
        outFile.write(str(cif_file))
        outFile.close()



def create_dataframe_for_xyz(sublattice_list, 
                             element_list,
                             x_distance, 
                             y_distance, 
                             z_distance,
                             filename, 
                             header_comment='top_level_comment'):
    """
    Parameters
    ----------
    
    Example
    -------
    
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = 'Mo_1'
            sublattice.atom_list[i].z_height = '0.5'
    >>> element_list = ['Mo_0', 'Mo_1', 'Mo_2']
    >>> x_distance, y_distance = 50, 50
    >>> z_distance = 5
    >>> dataframe = create_dataframe_for_xyz([sublattice], element_list,
                                 x_distance, y_distance, z_distance,
                                 save='dataframe',
                                 header_comment='Here is an Example')

    """
    df_xyz = pd.DataFrame(columns = ['_atom_site_Z_number',
                                    '_atom_site_fract_x', 
                                    '_atom_site_fract_y',
                                    '_atom_site_fract_z',
                                    '_atom_site_occupancy',
                                    '_atom_site_RMS_thermal_vib']) 
    
    # add header sentence
    df_xyz = df_xyz.append({'_atom_site_Z_number' : header_comment,
                            '_atom_site_fract_x' : '',
                            '_atom_site_fract_y' : '',
                            '_atom_site_fract_z' : '',
                            '_atom_site_occupancy' : '',
                            '_atom_site_RMS_thermal_vib' : ''},
                            ignore_index=True)
    
    # add unit cell dimensions
    df_xyz = df_xyz.append({'_atom_site_Z_number' : '',
                            '_atom_site_fract_x' : format(x_distance, '.6f'),
                            '_atom_site_fract_y' : format(y_distance, '.6f'),
                            '_atom_site_fract_z' : format(z_distance, '.6f'),
                            '_atom_site_occupancy' : '',
                            '_atom_site_RMS_thermal_vib' : ''},
                            ignore_index=True)  
    
    for sublattice in sublattice_list:
        # denomiator could also be: sublattice.signal.axes_manager[0].size
        
        for i in range (0, len(sublattice.atom_list)):
            if sublattice.atom_list[i].elements in element_list:
                #value = 0
                #this loop cycles through the length of the split element eg, 2 for 'Se_1.S_1' and
                #   outputs an atom label for each
                for k in range(0, len(split_and_sort_element(sublattice.atom_list[i].elements))):
                    if split_and_sort_element(sublattice.atom_list[i].elements)[k][2] >= 1:
                        atomic_number = split_and_sort_element(sublattice.atom_list[i].elements)[k][3]
                    
                        if "," in sublattice.atom_list[i].z_height:
                            atom_z_height = float(sublattice.atom_list[i].z_height.split(",")[k])
                        else:
                            atom_z_height = float(sublattice.atom_list[i].z_height)

                        # this loop controls the  z_height
                        for p in range(0, split_and_sort_element(sublattice.atom_list[i].elements)[k][2]):#len(sublattice.atom_list[i].z_height)):
    #could use ' ' + value to get an extra space between columns!
    # nans could be better than ''
    # (len(sublattice.image)-
                                
                            if "," in sublattice.atom_list[i].z_height and split_and_sort_element(sublattice.atom_list[i].elements)[k][2] > 1:
                                atom_z_height = float(sublattice.atom_list[i].z_height.split(",")[p])
                            else:
                                pass
                        
                            df_xyz = df_xyz.append({'_atom_site_Z_number' : atomic_number,
                                                  '_atom_site_fract_x' : format( sublattice.atom_list[i].pixel_x * (x_distance / len(sublattice.image[0,:])), '.6f'),
                                                  '_atom_site_fract_y' : format( sublattice.atom_list[i].pixel_y * (y_distance / len(sublattice.image[:,0])), '.6f'),
                                                  '_atom_site_fract_z' : format( atom_z_height * z_distance, '.6f'), # this is a fraction already, which is why we don't divide as in x and y
                                                  '_atom_site_occupancy' : 1.0, #might need to loop through the vancancies here?
                                                  '_atom_site_RMS_thermal_vib' : 0.05},
                                                ignore_index=True)          #insert row
    
    df_xyz = df_xyz.append({'_atom_site_Z_number' : int(-1),
                            '_atom_site_fract_x' : '',
                            '_atom_site_fract_y' : '',
                            '_atom_site_fract_z' : '',
                            '_atom_site_occupancy' : '',
                            '_atom_site_RMS_thermal_vib' : ''},
                            ignore_index=True)
    
    if filename is not None:
        df_xyz.to_csv(filename + '.xyz', sep=' ', header=False, index=False)

    return(df_xyz)







def simulate_with_prismatic(image,
                           xyz_filename,
                           filename,
                           calibration_area,
                           calibration_separation,
                           percent_to_nn=0.4,
                           mask_radius=None,
                           E0=60e3,
                           integrationAngleMin=0.085,
                           integrationAngleMax=0.186,
                           interpolationFactor=16,
                           realspacePixelSize=0.0654,
                           numFP=1,
                           probeSemiangle=0.030,
                           alphaBeamMax=0.032,
                           scanWindowMin=0.0,
                           scanWindowMax=1.0,
                           algorithm="prism",
                           numThreads=2):
                                                              
    ''' Image Position Loop '''
    
    if len(calibration_area) != 2:
        raise ValueError('calibration_area must be two points')
    
    real_sampling = image.axes_manager[0].scale
    real_sampling_exp_angs = real_sampling*10
    
    if str(real_sampling_exp_angs)[-1] == '5':
        real_sampling_sim_angs = real_sampling_exp_angs + 0.000005
    else:        
        pass

    simulation_filename = xyz_filename + '.XYZ'

    p = open(simulation_filename)

    pr_sim = pr.Metadata(filenameAtoms=simulation_filename)
    pr_sim.probeStepX = pr_sim.probeStepY = round(real_sampling_sim_angs, 6)
    pr_sim.detectorAngleStep = 0.001
    pr_sim.save3DOutput = False
    
    # param inputs
    pr_sim.E0 = E0
    pr_sim.integrationAngleMin = integrationAngleMin
    pr_sim.integrationAngleMax = integrationAngleMax
    pr_sim.interpolationFactorX = pr_sim.interpolationFactorY = interpolationFactor
    pr_sim.realspacePixelSizeX = pr_sim.realspacePixelSizeY = realspacePixelSize
    pr_sim.numFP = numFP
    pr_sim.probeSemiangle = probeSemiangle
    pr_sim.alphaBeamMax = alphaBeamMax # in rads
    pr_sim.scanWindowXMin = pr_sim.scanWindowYMin = scanWindowMin
    pr_sim.scanWindowYMax = pr_sim.scanWindowXMax = scanWindowMax
    pr_sim.algorithm = algorithm
    pr_sim.numThreads = numThreads
    pr_sim.filenameOutput = filename + '.mrc'

    pr_sim.go()

    simulation = hs.load('prism_2Doutput_' + pr_sim.filenameOutput)
    
    simulation.axes_manager[0].name='extra_dimension'
    simulation = simulation.sum('extra_dimension')
    simulation.axes_manager[0].scale = real_sampling
    simulation.axes_manager[1].scale = real_sampling
    simulation.axes_manager = image.axes_manager

    calibrate_distance_and_intensity(image=simulation, 
                                     cropping_area=calibration_area,
                                     separation=calibration_separation,
                                     filename=filename,
                                     percent_to_nn=percent_to_nn,
                                     mask_radius=mask_radius,
                                     scalebar_true=True)
    
    return(simulation)





def compare_image_to_filtered_image(image_to_filter, 
                                    reference_image,
                                    filename,
                                    delta_image_filter,
                                    cropping_area,
                                    separation,
                                    max_sigma=6,
                                    percent_to_nn=0.4,
                                    mask_radius=None,
                                    refine=False):
    '''
    Gaussian blur an image and compare to other image using mse and ssm. 
    Good for finding the best gaussian blur for the simulation by 
    comparing to an experimental image.
    
    >>> new_sim_data = compare_image_to_filtered_image(
                                    image_to_filter=simulation, 
                                    reference_image=atom_lattice_max)
    
    '''
    image_to_filter_data = image_to_filter.data
    reference_image_data = reference_image.data 

    mse_number_list = []
    ssm_number_list = []

    for i in np.arange(0, max_sigma+delta_image_filter, delta_image_filter):
        
        image_to_filter_data_filtered = gaussian_filter(image_to_filter_data, 
                                                        sigma=i)
        temp_image_filtered = hs.signals.Signal2D(image_to_filter_data_filtered)
#        temp_image_filtered.plot()
        calibrate_intensity_using_sublattice_region(image=temp_image_filtered, 
                                                cropping_area=cropping_area,
                                                separation=separation, 
                                                percent_to_nn=percent_to_nn,
                                                mask_radius=mask_radius,
                                                refine=refine)

        mse_number, ssm_number = measure_image_errors(
                imageA=reference_image_data,
                imageB=temp_image_filtered.data,
                filename=None)
        
        mse_number_list.append([mse_number, i])
        ssm_number_list.append([ssm_number, i])

    mse = [mse[:1] for mse in mse_number_list]
    mse_indexing = [indexing[1:2] for indexing in mse_number_list]
    ssm = [ssm[:1] for ssm in ssm_number_list]
    ssm_indexing = [indexing[1:2] for indexing in ssm_number_list]

    ideal_mse_number_index = mse.index(min(mse))
    ideal_mse_number = float(format(mse_number_list[ideal_mse_number_index][1], '.1f'))
    
    ideal_ssm_number_index = ssm.index(max(ssm))
    ideal_ssm_number = float(format(ssm_number_list[ideal_ssm_number_index][1], '.1f'))

    # ideal is halway between mse and ssm indices
    ideal_sigma = (ideal_mse_number + ideal_ssm_number)/2
    ideal_sigma_y_coord = (float(min(mse)[0]) + float(max(ssm)[0]))/2

    image_to_filter_filtered = gaussian_filter(image_to_filter_data,
                                        sigma=ideal_sigma)

    image_filtered = hs.signals.Signal2D(image_to_filter_filtered)

    calibrate_intensity_using_sublattice_region(image=image_filtered, 
                                            cropping_area=cropping_area,
                                            separation=separation, 
                                            percent_to_nn=percent_to_nn,
                                            mask_radius=mask_radius,
                                            refine=refine)

    if filename is not None:

        plt.figure()
        plt.scatter(x=ssm_indexing, y=ssm, label='ssm', marker = 'x', color='magenta')
        plt.scatter(x=mse_indexing, y=mse, label='mse', marker = 'o', color='b')
        plt.scatter(x=ideal_sigma, y=ideal_sigma_y_coord, label='\u03C3 = ' + str(round(ideal_sigma,2)), marker = 'D', color='k')
        plt.title("MSE & SSM vs. Gauss Blur " + filename, fontsize = 20)
        plt.xlabel("\u03C3 (Gaussian Blur)", fontsize = 16)
        plt.ylabel("MSE (0) and SSM (1)", fontsize = 16)
        plt.legend()
        plt.tight_layout
        plt.show()
        plt.savefig(fname='MSE_SSM_gaussian_blur_' + filename + '.png', 
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=300, labels=False)

    return(image_filtered)




def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def measure_image_errors(imageA, imageB, filename):
    
    ''' 
    Measure the Mean Squared Error (mse) and Structural Similarity Index (ssm)
    between two images.
    
    Parameters
    ----------

    imageA, imageB : 2D NumPy array, default None
        Two images between which to measure mse and ssm
    plot_details : Bool, default False
        Graphically plot the difference (A - B) and the mse, ssm on a 
        scatter plot.

    Returns
    -------

    mse_number, ssm_number : float
    returned subtracted image is imageA - imageB
    
    Example
    -------
    
    >>> imageA = am.dummy_data.get_simple_cubic_signal().data
    >>> imageB = am.dummy_data.get_simple_cubic_with_vacancies_signal().data
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageB,
                                                      plot_details=True)

    Showing the ideal case of both images being exactly equal   
    >>> imageA = am.dummy_data.get_simple_cubic_signal().data
    >>> imageB = am.dummy_data.get_simple_cubic_signal().data
    >>> mse_number, ssm_number = measure_image_errors(imageA, imageA,
                                                      plot_details=True)
    
    '''
    
    mse_number = mse(imageA, imageB)
    ssm_number = ssm(imageA, imageB)
    

    
    if filename is not None:
        plt.figure()
        plt.suptitle("MSE: %.6f, SSIM: %.6f" % (mse_number, ssm_number) + filename)
        
        plt.subplot(2, 2, 1)
        plt.imshow(imageA)
        plt.title('imageA')
        plt.axis("off")
        
        plt.subplot(2, 2, 2)
        plt.imshow(imageB)
        plt.title('imageB')
        plt.axis("off")
        
        plt.subplot(2, 2, 3)
        plt.imshow(imageA - imageB)
        plt.title('imageA - imageB')
        plt.axis("off")
        
        plt.subplot(2, 2, 4)
        plt.scatter(mse_number.size, mse_number, color='r',
                    marker='x', label='mse')
        plt.scatter(ssm_number.size, ssm_number, color='b',
                    marker='o', label='ssm')
        plt.title('MSE & SSM')
        plt.legend()
        plt.show()

        plt.savefig(fname='MSE_SSM_single_image_' + filename + '.png', 
            transparent=True, frameon=False, bbox_inches='tight', 
            pad_inches=None, dpi=300, labels=False)

    return(mse_number, ssm_number)







def image_difference_position(sublattice_list,
                               simulation_image,
                               pixel_threshold,
                               filename=None,
                               mask_radius=5,
                               num_peaks=5,
                               add_sublattice=False,
                               sublattice_name='sub_new'):
    
    '''
    >>> sublattice = am.dummy_data.get_simple_cubic_with_vacancies_sublattice(
                                                    image_noise=True)
    >>> simulation_image = am.dummy_data.get_simple_cubic_signal()
    
    >>> for i in range(0, len(sublattice.atom_list)):
                sublattice.atom_list[i].elements = 'Mo_1'
                sublattice.atom_list[i].z_height = '0.5'
    
    >>> # Check without adding a new sublattice
    >>> image_difference_position(sublattice_list=[sublattice],
                                                      simulation_image=simulation_image,
                                                      pixel_threshold=1,
                                                      mask_radius=5,
                                                      num_peaks=5,
                                                      plot_details=True,
                                                      add_sublattice=False)

    >>> # Add a new sublattice
    >>> # if you have problems with mask_radius, increase it!
    >>> # Just a gaussian fitting issue, could turn it off!
    >>> sub_new = image_difference_position(sublattice_list=[sublattice],
                                                      simulation_image=simulation_image,
                                                      pixel_threshold=10,
                                                      mask_radius=10,
                                                      num_peaks=5,
                                                      plot_details=True,
                                                      add_sublattice=True)
    '''
    image_for_sublattice = sublattice_list[0]
    diff_image = hs.signals.Signal2D(image_for_sublattice.image - simulation_image.data)
    diff_image_inverse = hs.signals.Signal2D(
            simulation_image.data - image_for_sublattice.image)

    # below function edit of get_atom_positions. Just allows num_peaks from
    # sklearn>find_local_maximum
    atom_positions_diff_image = get_atom_positions_in_difference_image(
            diff_image, num_peaks=num_peaks)
    atom_positions_diff_image_inverse = get_atom_positions_in_difference_image(
            diff_image_inverse, num_peaks=num_peaks)

    diff_image_sub = am.Sublattice(atom_positions_diff_image, diff_image)
    diff_image_sub.refine_atom_positions_using_center_of_mass(
            mask_radius=mask_radius)
    diff_image_sub.refine_atom_positions_using_2d_gaussian(
            mask_radius=mask_radius)
    atom_positions_sub_diff = np.array(diff_image_sub.atom_positions).T

    #sublattice.plot()

    diff_image_sub_inverse = am.Sublattice(atom_positions_diff_image_inverse, 
                                           diff_image_inverse)
    diff_image_sub_inverse.refine_atom_positions_using_center_of_mass(
            mask_radius=mask_radius)
    diff_image_sub_inverse.refine_atom_positions_using_2d_gaussian(
            mask_radius=mask_radius)
    atom_positions_sub_diff_inverse = np.array(
            diff_image_sub_inverse.atom_positions).T

    #these should be inputs for the image_list below

    atom_positions_diff_all = np.concatenate(
            (atom_positions_sub_diff, atom_positions_sub_diff_inverse))

    atom_positions_sub_new = []

    for atom in range(0, len(atom_positions_diff_all)):

        new_atom_distance_list = []

        for sublattice in sublattice_list:
            sublattice_atom_pos = np.array(sublattice.atom_positions).T

            for p in range(0, len(sublattice_atom_pos)):

                xy_distances = atom_positions_diff_all[atom] - \
                                                sublattice_atom_pos[p]

                # put all distances in this array with this loop
                vector_array = []
                vector = np.sqrt( (xy_distances[0]**2) + \
                                  (xy_distances[1]**2) )
                vector_array.append(vector)

                new_atom_distance_list.append(
                        [vector_array, atom_positions_diff_all[atom], 
                         sublattice])

        # use list comprehension to get the distances on their own, the [0] is 
        #changing the list of lists to a list of floats
        new_atom_distance_sublist = [sublist[:1][0] for sublist in 
                                     new_atom_distance_list]
        new_atom_distance_min = min(new_atom_distance_sublist)

        new_atom_distance_min_index = new_atom_distance_sublist.index(
                                            new_atom_distance_min)
        
        new_atom_index = new_atom_distance_list[new_atom_distance_min_index]

        if new_atom_index[0][0] > pixel_threshold: # greater than 10 pixels

            if len(atom_positions_sub_new) == 0:
                atom_positions_sub_new = [np.ndarray.tolist(new_atom_index[1])]
            else:
                atom_positions_sub_new.extend([np.ndarray.tolist(new_atom_index[1])])
        else:
            pass

    if len(atom_positions_sub_new) == 0:
        print("No New Atoms")
    elif len(atom_positions_sub_new) != 0 and add_sublattice == True:
        print("New Atoms Found! Adding to a new sublattice")

        sub_new = am.Sublattice(atom_positions_sub_new, sublattice_list[0].image,
                                name=sublattice_name, color='cyan')
#        sub_new.refine_atom_positions_using_center_of_mass(mask_radius=mask_radius)
#        sub_new.refine_atom_positions_using_2d_gaussian(mask_radius=mask_radius)

    else:
        pass
    
    
    try:
        sub_new
    except NameError:
        sub_new_exists = False
    else:
        sub_new_exists = True


    if filename is not None:
        '''
        diff_image.plot()
        diff_image_sub.plot()
        diff_image_inverse.plot()
        diff_image_sub_inverse.plot()
        '''
        
        plt.figure()
        plt.suptitle('Image Difference Position' + filename)
        
        plt.subplot(1, 2, 1)
        plt.imshow(diff_image.data)
        plt.title('diff')
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(diff_image_inverse.data)
        plt.title('diff_inv')
        plt.axis("off")
        plt.show()
        
        plt.savefig(fname='pos_diff_' + filename + '.png', 
        transparent=True, frameon=False, bbox_inches='tight', 
        pad_inches=None, dpi=300, labels=False)
        
        if sub_new_exists == True:
            sub_new.plot()
            plt.title(sub_new.name + filename, fontsize = 20)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(fname='pos_diff_' + sub_new.name + filename + '.png', 
                transparent=True, frameon=False, bbox_inches='tight', 
                pad_inches=None, dpi=300, labels=False)

    return sub_new if sub_new_exists == True else None












def change_sublattice_atoms_via_intensity(sublattice, image_diff_array, darker_or_brighter,
                                          element_list):
    # get the index in sublattice from the image_difference_intensity() output, 
    #   which is the image_diff_array input here.
    # then, depending on whether the image_diff_array is for atoms that should
    # be brighter or darker, set a new element to that sublattice atom_position
    if image_diff_array.size == 0:
        pass
    else:
        print('Changing some atoms')
        for p in image_diff_array[:,0]:
            p = int(p) # could be a better way to do this within image_difference_intensity()
            
            elem = sublattice.atom_list[p].elements
            if elem in element_list:
                atom_index = element_list.index(elem)
    
                if darker_or_brighter == 0:
                    if '_0' in elem:
                        pass
                    else:
                        new_atom_index = atom_index - 1
                        if len(sublattice.atom_list[p].z_height) == 2:
                            z_h = sublattice.atom_list[p].z_height
                            sublattice.atom_list[p].z_height = [(z_h[0] + z_h[1])/2]
                        else:
                            pass
                        new_atom = element_list[new_atom_index]

                elif darker_or_brighter == 1:
                    new_atom_index = atom_index + 1
                    if len(sublattice.atom_list[p].z_height) == 2:
                        z_h = sublattice.atom_list[p].z_height
                        sublattice.atom_list[p].z_height = [(z_h[0] + z_h[1])/2]
                    else:
                        pass
                        new_atom = element_list[new_atom_index]

                elif new_atom_index < 0:
                    raise ValueError("You don't have any smaller atoms")
                elif new_atom_index >= len(element_list):
                    raise ValueError("You don't have any bigger atoms")
    
#                new_atom = element_list[new_atom_index]
    
            elif elem == '':
                raise ValueError("No element assigned for atom %s. Note that this \
                                 error only picks up first instance of fail" %p)
            elif elem not in element_list:
                raise ValueError("This element isn't in the element_list")
            
            try: 
                new_atom
            except NameError:
                pass
            else:
                sublattice.atom_list[p].elements = new_atom




def image_difference_intensity(sublattice,
                               simulation_image,
                               element_list,
                               filename,
                               percent_to_nn=0.40,
                               mask_radius=None,
                               change_sublattice=True):

    ''' 
    Find the differences in a sublattice's atom_position intensities. 
    Change the elements of these atom_positions depending on this difference of
    intensities.
    
    Parameters
    ----------

    sublattice : Atomap Sublattice object, default None
        The sublattice whose elements you are refining.
    simulation image : HyperSpy 2D signal, default None
        The image you wish to refine with, usually an image simulation of the
        sublattice.image
    percent_to_nn : float, default 0.40
        Determines the boundary of the area surrounding each atomic 
        column, as fraction of the distance to the nearest neighbour.
    plot_details : Bool, default False
        Plot the sublattice, simulation_image, difference between them,
        and a histogram of the chosen intensities, overlaid with mean and 3
        standard deviations from the mean

    Returns
    -------
    Nothing - changes the elements within the sublattice object.

    Example
    -------
    
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> simulation_image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = 'Mo_1'
            sublattice.atom_list[i].z_height = [0.5]
    >>> element_list = ['H_0', 'Mo_1', 'Mo_2']
    >>> image_difference_intensity(sublattice=sublattice,
                                   simulation_image=simulation_image,
                                   element_list=element_list)

    with some image noise and plotting the images
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice(image_noise=True)
    >>> simulation_image = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> for i in range(0, len(sublattice.atom_list)):
            sublattice.atom_list[i].elements = 'Mo_1'
            sublattice.atom_list[i].z_height = [0.5]
    >>> element_list = ['H_0', 'Mo_1', 'Mo_2']
    >>> image_difference_intensity(sublattice=sublattice,
                                   simulation_image=simulation_image,
                                   element_list=element_list,
                                   plot_details=True)
    
    '''

    sublattice_atom_positions = np.array(sublattice.atom_positions).T  # np.array().T needs to be taken away for newer atomap versions        
    
    diff_image = hs.signals.Signal2D(sublattice.image - simulation_image.data)

    # create sublattice for the 'difference' data
    diff_sub = am.Sublattice(atom_position_list=sublattice_atom_positions, image=diff_image)

    if percent_to_nn is not None:
        sublattice.find_nearest_neighbors()
        diff_sub.find_nearest_neighbors()
    else:
        pass
    
    # Get the intensities of these sublattice positions
    diff_sub.get_atom_column_amplitude_mean_intensity(percent_to_nn=percent_to_nn, mask_radius=mask_radius)
    diff_mean_ints = np.array(diff_sub.atom_amplitude_mean_intensity, ndmin=2).T
    #diff_mean_ints = np.array(diff_mean_ints, ndmin=2).T

    # combine the sublattice_atom_positions and the intensities for
    # future indexing
    positions_intensities_list = np.append(sublattice_atom_positions,
                                      diff_mean_ints, axis=1)
    # find the mean and std dev of this distribution of intensities
    mean_ints = np.mean(diff_mean_ints)
    std_dev_ints = np.std(diff_mean_ints)
    
    # plot the mean and std dev on each side of intensities histogram
    std_from_mean = np.array([mean_ints-std_dev_ints, mean_ints+std_dev_ints,
                    mean_ints-(2*std_dev_ints), mean_ints+(2*std_dev_ints),
                    mean_ints-(3*std_dev_ints), mean_ints+(3*std_dev_ints),
                    mean_ints-(4*std_dev_ints), mean_ints+(4*std_dev_ints)
                    ], ndmin=2).T
    y_axis_std = np.array([len(diff_mean_ints)/100] * len(std_from_mean),
                    ndmin=2).T
    std_from_mean_array = np.concatenate( (std_from_mean, y_axis_std), axis=1)
    std_from_mean_array = np.append(std_from_mean, y_axis_std, axis=1)

    #if the intensity if outside 3 sigma, give me those atom positions
    # and intensities (and the index!)
    outliers_bright, outliers_dark = [], []
    for p in range(0, len(positions_intensities_list)):
        x, y = positions_intensities_list[p,0], positions_intensities_list[p,1]
        intensity = positions_intensities_list[p,2]

        if positions_intensities_list[p,2] > std_from_mean_array[7,0]:
            outliers_bright.append([p, x, y, intensity])
        elif positions_intensities_list[p,2] < std_from_mean_array[6,0]:
            outliers_dark.append([p, x, y, intensity])
    # Now we have the details of the not correct atom_positions
    outliers_bright = np.array(outliers_bright)
    outliers_dark = np.array(outliers_dark)
    
    if change_sublattice == True:
        # Now make the changes to the sublattice for both bright and dark arrays
        change_sublattice_atoms_via_intensity(sublattice=sublattice,
                                              image_diff_array=outliers_bright,
                                              darker_or_brighter=1,
                                              element_list=element_list)
        change_sublattice_atoms_via_intensity(sublattice=sublattice,
                                              image_diff_array=outliers_dark,
                                              darker_or_brighter=0,
                                              element_list=element_list)

    else:
        pass

    if filename is not None:
#        sublattice.plot()
#        simulation_image.plot()
#        diff_image.plot()
        diff_sub.plot()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title("Image Differences with " + sublattice.name + " Overlay", fontsize = 16)
        plt.savefig(fname="Image Differences with " + sublattice.name + "Overlay.png", 
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=300, labels=False)
        
        plt.figure()
        plt.hist(diff_mean_ints, bins=50, color='b', zorder=-1)
        plt.scatter(mean_ints, len(diff_mean_ints)/50, c='red', zorder=1)
        plt.scatter(std_from_mean_array[:,0], std_from_mean_array[:,1], c='green', zorder=1)
        plt.title("Histogram of " + sublattice.name + " Intensities", fontsize = 16)
        plt.xlabel("Intensity (a.u.)", fontsize = 16)
        plt.ylabel("Count", fontsize = 16)
        plt.tight_layout()
        plt.show()
        plt.savefig(fname="Histogram of " + sublattice.name + " Intensities.png", 
                    transparent=True, frameon=False, bbox_inches='tight', 
                    pad_inches=None, dpi=300, labels=False)

    else:
        pass