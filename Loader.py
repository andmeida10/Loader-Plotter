import numpy as np
import arrayfire as af
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit
import os
import imageio as imo
from datetime import datetime as now


class meshPlot:
    '''
    Class that contains information about the mesh of the data loaded.

    Only works with ArrayFire file format in the following protocol: SF_XXXX.af
    :: For different suffix and prefix the *load_parameters()* has to be changed

    At the moment it only has the hability to process 1D input data and turn them in a 2D (or semi-3D) output figure/movie.

    The format of movie/figures can be altered in the specific function, namely *plot_sequence(InPath)*

    **The Initializer does**
    Initiate the object to proceed with the data loading and dump them in a specific requeired format as output.

    :param InPath: The relative path where the group of ArrayFire data is located.
    :param Plot1D: A Boolean parameter to inform the class if the user wants the set of input data to be converted to a 2D plot.
    :param PlotSlices: A Boolean parameter to inform the class if the user wants a Semi-3D plot of the group data available in the InPath folder.
    :param PlotTemporal: A Boolean parameter to inform the class if the user wants a output in the 2D color-contrast mode graph (pcolor/imshow format)
    :param Mov: A simple Boolean to inform the class if the user wants a final video of the movement to be rendered.
    '''

    def __init__(self, InPath, Plot1D=True, PlotSlices=False, PlotTemporal=False, Mov=False):

        self.key = 'SF'  # The prefix of the files names
        self.PlotTemporal = PlotTemporal
        self.Plot1D = Plot1D
        self.PlotSlices = PlotSlices

        _dim, _dt, _dh, _N, _lim = self.load_parameters()
        self.dim = _dim
        self.dt = _dt

        self.dx = _dh[0];
        self.dy = _dh[1];
        self.dz = _dh[2]
        self.Nx = _N[0];
        self.Ny = _N[1];
        self.Nz = _N[2]
        self.limx = _lim[0];
        self.limy = _lim[1];
        self.limz = _lim[2]

        # Analyze which data will be accessed.
        self.NumberFiles = 0
        self.List_Files = []
        # Collect all names of files and the respective step index
        for file in os.listdir(InPath + '/'):
            if file.endswith('.af'):
                self.List_Files.append([file, float(file.split('_')[1].split('.')[0])])
        self.List_Files.sort(key=lambda s: s[1])
        self.NumberFiles = len(self.List_Files)

        # For statistical data!
        self.norm = np.zeros(self.NumberFiles)
        self.maxPos = np.zeros(self.NumberFiles)
        self.FWHM = np.zeros(self.NumberFiles)
        self.StatisticalPath = 'Statistics'

        # Contruction of the Axis Values
        self.x = np.arange(0, self.Nx) * self.dx
        self.y = np.arange(0, self.Ny) * self.dy
        self.z = np.arange(0, self.Nz) * self.dz

        # Plot the images and if set, produce the movie
        self.plot_sequence(InPath, DoMovie=Mov)

    def dissSechAbs(self, xx, A, B, O, x0, d):
        '''
        Defined the function to analyze Bright Solitons. These are one of the many types of
        solitons observed in these kind of equations. Bright solitons are the ones studied so far

        This version, before returning the final array converts the data to its absolute value so that Real Data analysis can be performed.

        :param xx: The collection of x-axis data where the soliton is based
        :param A: This real parameter represents the amplitude of the soliton.
        :param B: This is the squeezing factor. It is also inversely proportional to the FWHM parameter
        :param O: Related to the propagation chirp of the soliton wave-form
        :param x0: Displacement of the maximum in the x-axis base.
        :param d: Exponent factor related to the imaginary part of the power.
        :return: This function return an *Numpy* array describing the Bright Soliton that fulfills the given parameters.
        '''

        func = np.zeros(int(self.Nx))
        func = A * np.exp( (1 + d*1j) * np.log( 1.0/np.cosh( B * (xx - x0) ) ) )*np.exp( (-O*1j)*(xx - x0) )
        func = np.abs(func)
        return func

    def dissSech(self, xx, A, B, O, x0, d):
        '''
        This function has the same behavior as the one called *dissSechAbs* but has no absolute
        convertion to be handled for statistical analysis.
        Be careful when using it.
        '''

        func = np.zeros(int(self.Nx))
        func = A * np.exp( (1 + d*1j) * np.log( 1.0/np.cosh( B * (xx - x0) ) ) )*np.exp( (-O*1j)*(xx - x0) )
        return func

    def load_parameters(self):
        '''
        Reads the file with parameters of the simulation and returns a set
        of data in the following order:
        :: Dimensions, Time Step, Spatial Step, Number of Points, Vector Limits

        No arguments must be passed and a file *parameters.dat* in a defined protocol format has to exists with accurate mesh information
        '''
        f = open("parameters.dat", 'r')

        f.readline()
        dim = np.int(f.readline())
        f.readline()
        dt = np.float(f.readline())
        f.readline()
        dh = np.zeros(3)
        for i in np.arange(3):
            dh[i] = np.float(f.readline())

        f.readline()
        N = np.zeros(3)
        for i in np.arange(3):
            N[i] = np.float(f.readline())

        f.readline()
        lim = np.zeros(3)
        for i in np.arange(3):
            lim[i] = np.float(f.readline())

        f.close()
        return dim, dt, dh, N, lim

    def load_envelope(self, filename):
        '''
         Function to load the file in ArrayFire format (*.af*) and convert it to
         a 1D Array in the NumPy format so it can be handled and plotted.

         :param filename: The file name of the ArrayFire format file to load and convert to *Numpy* Array format.
        '''
        data_af = af.array.read_array(filename, key=self.key)
        return np.array(data_af, order='F')

    def plot_sequence(self, InPath, DoMovie):
        '''
        Function to plot the set of saved data with the matplotlib
        plot functions. This makes the output beautiful!!!

        Statistical informations are saved to the folder named "Statistical".
        Those informations are fundamental to perform analysis of the propagation.

        :param InPath: String value indicating the relative path of all *ArrayFire* (*.af*) files containing the data computed by the GPU
        :param DoMovie: Simple Boolean variable indicating if the user pretends a final Movie to be rendered. (typical fps can be changed in the source code)
        '''

        filenames = [] # To make the movie at the end

        if self.dim != 1:
            raise IndexError('The current version only allow loading 1D data-files')

        if self.Plot1D or self.PlotSlices:
            PlotPath = 'Plot_' + InPath.split('/')[-1]
            print( 'Directory to Save Plots: ' + str(PlotPath))
            if not os.path.exists(PlotPath):
                os.makedirs(PlotPath)
            else: # This allows that files never get overwriten
                i = 1
                NewPlotPath = PlotPath
                while os.path.exists(NewPlotPath):
                    NewPlotPath = PlotPath + '_' + str(i)
                    i+=1
                PlotPath = NewPlotPath
                os.makedirs(PlotPath)

        max_value = 0
        Extent = [0, self.Nx * self.dx, 0, 0]

        if self.Plot1D:
            for i in np.arange(0, self.NumberFiles):
                envelope = self.load_envelope(InPath + '/' + self.List_Files[i][0])

                 # Does the little trick to keep the Y axis constant along the movie!
                if i == 0:
                    max_value = np.max(np.abs(envelope) ** 2)
                envelope[0] = max_value

                # ----------- Data to Perfetct Plot!!! -----------
                t = self.dt * self.List_Files[i][1]
                pl.title('Test and Change Latter')

                pl.plot(np.linspace(Extent[0], Extent[1], self.Nx), np.abs(envelope) ** 2 )
                pl.xlabel('X')
                pl.ylabel('Y')

                pl.savefig(PlotPath + '/image%.2f.png' % i, format='png', dpi=300)
                filenames.append(PlotPath + '/image%.2f.png' % i)
                pl.clf()
                print('Envelope %i out of %i saved' % (i+1, self.NumberFiles))

        if self.PlotSlices:

            # For now, it will be saved in the movie folder. Latter can be changed
            Movie_folder = "Movie/"
            if not os.path.exists(Movie_folder):
                os.makedirs(Movie_folder)

            fig = pl.figure()
            ax = fig.gca(projection='3d')
            dataYs = np.zeros(self.NumberFiles)

            for i in np.arange(0, self.NumberFiles):
                envelope = self.load_envelope(InPath + '/' + self.List_Files[i][0])

                # ----------- Data to Perfetct Plot!!! -----------
                t = self.dt * self.List_Files[i][1]
                dataYs[i] = t

                xs = np.linspace(Extent[0], Extent[1], self.Nx)
                ys = np.ones(len(envelope)) * t
                zs = np.abs(envelope) ** 2
                ax.plot(xs, ys, zs, color='black')

                # The user wants to save the statistical information about the signal propagation:
                # Create the arrays to store the Statistical Data with the correct length.

                # The data we want to retract:
                # Norm of the signal
                # The FWHM for the case of a BRIGHT soliton!!
                # Maximum position as propagation along the Z axis.

                self.norm[i] = np.sum(zs)
                self.maxPos[i] = np.argmax(zs)

                parameters = curve_fit(self.dissSechAbs, xs, zs)[0]
                self.FWHM[i] = 2.634 * 1.0/parameters[1]



                print('Envelope %i out of %i ploted' % (i+1, self.NumberFiles))

            if not os.path.exists(self.StatisticalPath):
                os.makedirs(self.StatisticalPath)
            with open(self.StatisticalPath + '/zPropagation.txt', 'wb') as f:
                np.savetxt(f, dataYs, newline="\r\n")
            with open(self.StatisticalPath + '/normData.txt', 'wb') as f:
                np.savetxt(f, self.norm, newline="\r\n")
            with open(self.StatisticalPath + '/maxPosData.txt', 'wb') as f:
                np.savetxt(f, self.maxPos*self.dx, newline="\r\n")
            with open(self.StatisticalPath + '/FWHMData.txt', 'wb') as f:
                np.savetxt(f, self.FWHM, newline="\r\n")

            pl.show() # It will stop here until i close the graph!
            # Manual save has to be performed because one can rotate the graph to a
            # defined position!!

        if DoMovie:
            if self.Plot1D == False:
                raise AttributeError('You can not do the movie without plot the data')
            print('Making the movie...')
            Movie_folder = "Movie/"
            if not os.path.exists(Movie_folder):
                os.makedirs(Movie_folder)
            images = []
            for filename in filenames:
                images.append(imo.imread(filename))
            imo.mimsave(Movie_folder + "/" + PlotPath + "_" + \
                    str(now.now().hour) + str(now.now().minute)+str(now.now().second) + \
                    "_movie.mkv", images, fps=20)
            print('Complete!')

        if self.PlotTemporal:
            print('Doing the 2D Temporal Plot Graph.')
            # For now, it will be saved in the movie folder. Latter can be changed
            Movie_folder = "Movie/"
            if not os.path.exists(Movie_folder):
                os.makedirs(Movie_folder)

            TemporalData = np.zeros(( self.NumberFiles, self.Nx ))

            for i in np.arange(0, self.NumberFiles):
                envelope = self.load_envelope(InPath + '/' + self.List_Files[i][0])

                # Does the little trick to keep the Y axis constant along the movie!
                if i == 0:
                    max_value = np.max(np.abs(envelope) ** 2)
                envelope[0] = max_value

                TemporalData[i, :] = np.abs(envelope)**2
                print('Envelope %i out of %i loaded' % (i + 1, self.NumberFiles))

            pl.figure()
            xaxis = np.linspace(0, self.Nx * self.dx, self.Nx)
            yaxis = np.linspace(0, self.NumberFiles * self.dt, len(TemporalData[:,0]))
            pl.pcolormesh(xaxis, yaxis, TemporalData, cmap='inferno')
            pl.xlim(xaxis[0], xaxis[-1])
            pl.ylim(yaxis[0], yaxis[-1])

            pl.title('Simple 2D plot')
            pl.xlabel('Section-Cut')
            pl.ylabel('Propagated Distance')

            # The temporal graph is overriding if a previous is present!
            pl.savefig(Movie_folder + '/TemporalPropagation.png', format='png', dpi=500)
            pl.close()
        return 0


if __name__ == '__main__':
    # 1 simple example of usage of the class
    InPath = 'Data'
    loader = meshPlot(InPath, Plot1D=False, PlotSlices=True, PlotTemporal=False, Mov=False)