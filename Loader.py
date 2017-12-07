import numpy as np
import arrayfire as af
import matplotlib.pyplot as pl
import os
import imageio as imo
from datetime import datetime as now


class meshPlot:
    '''
    Class that contains information about the mesh of the data loaded
    '''

    def __init__(self, InPath, Mov=False):

        self.key = 'SF' # The prefix of the files names

        _dim, _dt, _dh, _N, _lim = self.load_parameters()
        self.dim = _dim
        self.dt = _dt

        self.dx = _dh[0]    ; self.dy = _dh[1]      ; self.dz = _dh[2]
        self.Nx = _N[0]     ; self.Ny = _N[1]       ; self.Nz = _N[2]
        self.limx = _lim[0] ; self.limy = _lim[1]   ; self.limz = _lim[2]

        # Contruction of the Axis Values
        self.x = np.arange(0, self.Nx) * self.dx
        self.y = np.arange(0, self.Ny) * self.dy
        self.z = np.arange(0, self.Nz) * self.dz

        #Plot the images and if set, produce the movie
        self.plot_sequence(InPath, DoMovie=Mov)


    def load_parameters(self):
        '''
        Reads the file with parameters of the simulation and returns a set
        of data in the following order:
        dimentions, time step, number of points, vector limits
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
         Function to load the file in ArrayFire format and convert it to
         an array of NumPy format.
        '''
        data_af = af.array.read_array(filename, key=self.key)
        return np.array(data_af, order='F')

    def plot_sequence(self, InPath, DoMovie):
        '''
        Function to plot the set of saved data with the matplotlib
        plot functions. This makes the output beautiful!!!
        '''

        filenames = [] # To make the movie at the end

        if self.dim != 1:
            raise IndexError('The current version only allow loading 1D data-files')

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

        List_Files = []

        # Colect all names of files and the respective step index
        for file in os.listdir(InPath + '/'):
            if file.endswith('.af'):
                List_Files.append([file, float(file.split('_')[1].split('.')[0])])
        List_Files.sort(key=lambda s: s[1])
        max_value = 0
        Extent = [0, self.Nx * self.dx, 0, 0]

        for i in np.arange(0, len(List_Files)):
            envelope = self.load_envelope(InPath + '/' + List_Files[i][0])

             # Does the little trick to keep the Y axis constant along the movie!
            if i == 0:
                max_value = np.max(np.abs(envelope) ** 2)
            envelope[0] = max_value

            # ----------- Data to Perfetct Plot!!! -----------
            t = self.dt * List_Files[i][1]
            pl.title('Test and Change Latter')

            pl.plot(np.linspace(Extent[0], Extent[1], self.Nx), np.abs(envelope) ** 2 )
            pl.xlabel('X')
            pl.ylabel('Y')

            pl.savefig(PlotPath + '/image%.2f.png' % i, format='png', dpi=300)
            filenames.append(PlotPath + '/image%.2f.png' % i)
            pl.clf()
            print('Envelope %i out of %i saved' % (i+1, len(List_Files)))

        if DoMovie:
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
        return 0

InPath = 'Data'
loader = meshPlot(InPath, Mov = True)