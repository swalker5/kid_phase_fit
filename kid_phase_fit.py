'''
kid_phase_fit.py

SW 2/2023
'''


import math
import cmath
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.io import loadmat, savemat
from scipy import interpolate
from tqdm import tqdm
import numpy.ma as ma
import glob
import pickle
from scipy import linalg
from scipy import optimize
import pandas as pd
import scipy.stats
from scipy.optimize import curve_fit
import lmfit
from scipy.stats import chisquare
import netCDF4
from timeit import default_timer as timer
from matplotlib.backends.backend_pdf import PdfPages
import csv
import yaml
import argparse

# set plotting fontsize
plt.rcParams.update({'font.size': 14})


def magS21(i,q):
    return np.sqrt(i**2+q**2) # similar to np.abs

    
def logmag(z): # similar to magS21_dB(i,q)
    return 20.0*np.log10(np.abs(z)) # np.sqrt(z.real**2+z.imag**2)


def complexS21(i,q):
    return i + 1.j*q # I + jQ


def phase2(z): # have another phase function below
    x = np.angle(z)
    x_cp = np.copy(x)
    y = np.unwrap(x_cp)
    return y


def findcabledelay(f,z):
    # assumes f in GHz, identical to MATLAB code
    
    theta = phase2(z) # includes phase unwrapping
    grad = np.gradient(theta)/np.gradient(f)/2./np.pi
    
    fig = plt.figure()
    ax = plt.subplot(211)
    plt.plot(f, theta)
    plt.xlabel('f [GHz]')
    plt.ylabel('Phase')
    
    ax = plt.subplot(212)
    plt.plot(f, grad)
    tau = np.mean(grad)
    #plt.plot(f, tau,'r--') # horizontal line
    #plt.axhline(y=tau, color='r', linestyle='--')
    plt.xlabel('f [GHz]')
    plt.ylabel('tau [ns]')
    plt.title('tau = ' + str(tau))
    fig.tight_layout()
    plt.show()
    print(tau)
    
    return tau


def fit_cable_delay(gain_f,gain_z):
    # gain f in Hz
    gain_phase = phaseS21(gain_z.real,gain_z.imag)
    gain_phase_cp = np.copy(gain_phase)
    gain_phase = np.unwrap(gain_phase) # unwrap phase with numpy
    
    p_phase = np.polyfit(gain_f,gain_phase,1)
    tau = p_phase[0]/(2.*np.pi)
    poly_func_phase = np.poly1d(p_phase)
    fit_data_phase = poly_func_phase(gain_f)

    return tau*1e9,fit_data_phase,gain_phase


# refer to https://matplotlib.org/stable/gallery/misc/transoffset.html#sphx-glr-gallery-misc-transoffset-py
# for switching between projections using ax and subplot
def removecable(f,z,tau,verbose=False,showplot=False): 
    # f,z,tau,f1
    #f1 = f[0]
    if tau > 0:
        tau = -1.*tau
    
    if(f[0]>1e7):
        if verbose:
            print('Converting f from Hz to GHz before removing cable delay')
        f = f/1e9
    f1 = f[0]
    #print(f)
    #print(f1)
    #print(f-f1)
    z1 = z*np.exp(-1.j*2.*np.pi*(f)*tau) # (f-f1) # assumes tau is already negative

    if showplot:
        #fig = plt.figure()
        #ax = plt.subplot(221, projection='polar')
        fig = plt.subplots(2, 2)

        plt.subplot(211)
        #ax = plt.subplot(222, projection='rectilinear') # default projection
        plt.plot(f,logmag(z),'c')
        plt.plot(f,logmag(z1),'k')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('$|S_{21}|^2$')
        #plt.show()

        plt.subplot(222,polar=True) # np.arctan2(np.imag(self.z),np.real(self.z))
        plt.plot(phase2(z),np.abs(z),'c')
        plt.plot(phase2(z1),np.abs(z1),'k')
        ##polar2(z,color='c')
        ##polar2(z1,color='k')
        plt.show()
    
    return z1


def find_nearest_indice(x,val):
    idx = np.searchsorted(x, val, side="left")
    if idx > 0 and (idx == len(x) or math.fabs(val - x[idx-1]) < math.fabs(val - x[idx])):
        return idx - 1 #x[idx-1]
    else:
        return idx #x[idx]


def trimdata(f,z,f0,Q,numspan=1):
    f0id = find_nearest_indice(f, f0)
    idspan =  (f0/Q) / (f[1]-f[0])
    
    idstart = int(np.floor(np.amax([f0id - idspan*numspan,0])))

    idstop = int(np.floor(np.amin([f0id + idspan*numspan,len(f)])))+1

    if idstop > len(f):
        idstop = len(f)

    fb = f[idstart:idstop]
    zb = z[idstart:idstop]
    return fb, zb


def smoothdata2(y, N): # more matlab like
    # compute moving average with window size N
    
    if N%2==0: # checks if N is even
        N = N - 1

    y_out = np.convolve(y, np.ones(N, dtype=int), mode='valid')/N
    r = np.arange(1,N-1,2)
    start = np.cumsum(y[:N-1])[::2]/r
    stop = (np.cumsum(y[:-N:-1])[::2]/r)[::-1]
    y_smooth = np.concatenate((start, y_out, stop))
    return y_smooth


# put this in one of the classes instead?
def estpara3(f,z,verbose=False,**keywords):
    # to do: add guess for phi0?

    if ('result0' in keywords):
        if verbose:
            print('Using linear regime fit result0')
        result0 = keywords['result0']
        f0g = result0['f0'] #[0][0]
        Qg = result0['Q'] #[0][0]
        v = np.amin(np.abs(f - f0g))
        idf0 = np.argmin(np.abs(f - f0g))
        iddf = np.abs(find_nearest_indice(f, f0g - f0g/Qg/2.) - find_nearest_indice(f, f0g + f0g/Qg/2.))
    else:
        z1 = np.mean(z[:10])
        z2 = np.mean(z[len(f)-11:len(f)])
        zr = (z1 + z2)/2.
        dmax = np.amax(np.abs(z - z1) + np.abs(z - z2))
        idf0 = np.argmax(np.abs(z - z1) + np.abs(z - z2))
        z0 = z[idf0]
        f0g = f[idf0]
        zcg = (zr + z0)/2.
        rg = np.abs(zr - z0)/2.
        zz1 = z[:idf0+1]
        r01 = np.amin(np.abs(np.abs(zz1-z0) - np.sqrt(2.)*rg))
        id3db1 = np.argmin(np.abs(np.abs(zz1-z0) - np.sqrt(2.)*rg))
        zz2 = z[idf0+1:len(z)]
        r02 = np.amin(np.abs(np.abs(zz2-z0) - np.sqrt(2.)*rg))
        id3db2 = np.argmin(np.abs(np.abs(zz2-z0) - np.sqrt(2.)*rg))
        iddf = np.amax([np.abs(idf0 - id3db1), np.abs(id3db2)])
        Qg = f0g / (2.*iddf*(f[1]-f[0]))

    # set to correct dtypes  
    f0g = float(f0g)
    Qg = float(Qg)
    idf0 = int(idf0)
    iddf = int(iddf)

    return f0g, Qg, idf0, iddf


def plot_summary_for_index(data,ii,fig=None,ax=None):
    if not fig: fig, ax = plt.subplots(2, 2)

    ax[0,0].plot(data[:,ii,0],data[:,ii,1])
    ax[0,0].set(ylabel='i')

    ax[0,1].plot(data[:,ii,0],data[:,ii,2])
    ax[0,1].set(ylabel='q',xlabel='Freq (Hz)')

    y=magS21(data[:,ii,1],data[:,ii,2])
    ax[1,0].plot(data[:,ii,0],y)
    ax[1,0].set(ylabel='mag S21 (lin)',xlabel='Freq (Hz)')

    ax[1,1].plot(data[:,ii,1],data[:,ii,2])
    ax[1,1].set(xlabel='i',ylabel='q')
    return fig,ax


class LoadVNAsweep():
    def __init__(self,filename):
        self.filename = filename

        if self.filename[-3:] == 'mat':
            self.data = loadmat(self.filename)
            self.freqs = self.data['f']
            self.z = self.data['z']
        elif self.filename[-3:] == 'csv':
            #self.data = read_vna_data(self.filename)
            self.freqs, self.z = read_vna_data(self.filename)

    def read_vna_data(self): # input_filename
        freqs = []
        real = []
        imag = []
        with open(self.filename, "r") as f:
            raw_lines = [raw_line.strip() for raw_line in f.readlines()]
        for line_index, raw_line in tqdm(list(enumerate(raw_lines))):
            if raw_lines[line_index][0] == "#":
                pass
            else:
                split = raw_lines[line_index].split(",")
                freqs.append(float(split[0]))
                real.append(float(split[1]))
                imag.append(float(split[2]))

        return np.asarray(freqs),np.asarray(real+1j*np.asarray(imag))


class Circlefit():
    ''' Class to perform a circle fit on data '''
    def __init__(self,z,scaling=False):
        ''' data needs to be complex (like I + jQ)
        '''
        self.z = z 
        self.scaling = scaling
        self.r, self.zc, self.residue = self.nsphere_fit_test(scaling=self.scaling)

    # based on scikit-guess package
    def nsphere_fit_test(self,scaling): # not using axis right now # axis=-1
        """
        Fit an n-sphere to ND data.

        The center and radius of the n-sphere are optimized using the Coope
        method. The sphere is described by

        .. math::

           \left \lVert \vec{x} - \vec{c} \right \rVert_2 = r

        Parameters
        ----------
        x : array-like
            The n-vectors describing the data. Usually this will be a nxm
            array containing m n-dimensional data points.
        axis : int
            The axis that determines the number of dimensions of the
            n-sphere. All other axes are effectively raveled to obtain an
            ``(m, n)`` array.
        scaling : bool
            If `True`, scale and offset the data to a bounding box of -1 to
            +1 during computations for numerical stability. Default is
            `False`.

        Return
        ------
        r : scalar
            The optimal radius of the best-fit n-sphere for `x`.
        c : array
            An array of size `x.shape[axis]` with the optimized center of
            the best-fit n-sphere.

        References
        ----------
        - [Coope]_ "\ :ref:`ref-cfblanls`\ "
        """
    
        x = np.column_stack((self.z.real, self.z.imag)) # or rather xy
        x = np.asarray(x) #preprocess(x, float=True, axis=axis)
        n = x.shape[-1]
        x = x.reshape(-1, n)
        m = x.shape[0]

        B = np.empty((m, n+1), dtype=x.dtype)
        X = B[:,:-1]
        X[:] = x
        B[:,-1] = 1 # based on Coope's method, column vector bj = [aj ... 1]
        #print(np.shape(B))
        #print(np.shape(x))

        if scaling:
            print('Using scaling')
            xmin = X.min()
            xmax = X.max()
            scale = 0.5 * (xmax - xmin)
            offset = 0.5 * (xmax + xmin)
            X -= offset
            X /= scale

        d = np.sum(X**2, axis=-1) # square(X).sum(axis=-1) # sum of aj^2
        #print(np.shape(d))

        # least squares of By = d or minimize |d - B y|
        y, residue, rank_B, B_sing = linalg.lstsq(B, d, overwrite_a=True, overwrite_b=True) # *_
        #print(np.shape(y))

        # to transform back to original coordinates to recover c or xi and r
        # need to change residues for c and r too?
        zc = 0.5 * y[:-1] # c
        r = np.sqrt(y[-1] + np.sum(zc**2)) # sqrt(y[-1] + square(c).sum())

        if scaling:
            r *= scale
            zc *= scale
            zc += offset

        zc_complex = zc[0] + 1.j*zc[1]

        return r, zc_complex, residue


class ResonanceFitterSingleTone():
    ''' Class to extract resonance parameters following algorithm in Gao Thesis Appendix E
        1) remove cable delay
        2) find circle x0,y0 and radius
        3) rotate and translate to origin
        4) fit phase versus frequency to extract fr, Qr

        Class to load resonator data and perform a simple or 
        nonlinear phase fit (user-specified) on a single tone
    '''
    def __init__(self,f,z,tau,numspan=2,tone_freq_lo=0,window_width=0,\
                 pherr_threshold_num=10,pherr_threshold=0.2,\
                 verbose=False,**keywords):
        # f in Hz (will also work with GHz?), z is complex, tau in ns
        ''' fit the resonance following method described in Jiansong Gao thesis Appendix E

            data needs to be complex (like I + jQ)
        '''

        self.f = f
        self.z = z # z_real+1j*z_imag
        self.tau = tau
        self.tone_freq_lo = tone_freq_lo
        self.numspan = numspan
        self.ang = phase2(self.z)
        self.pherr_threshold_num = int(pherr_threshold_num)
        self.pherr_threshold = pherr_threshold

        # new argument for specifying output to print to user
        self.verbose = verbose
        
        self.window_width = window_width
        
        if ('weight_type' in keywords):
            self.use_weight = True
            self.weight_type = keywords['weight_type']
            if self.window_width == 0:
                print('To use weighting, need to specify a window width')
                self.use_weight = False
        else:
            self.use_weight = False

        if ('result0' in keywords):
            self.result0 = keywords['result0'] # dictionary of fit results
            self.result = self.fits21simnlin()
        else:
            self.result = self.fits21sim() # result0 is a linear fit with fits21sim

        self.z1 = removecable(self.f,self.z,self.tau,verbose=self.verbose)
    
        if ('result0' in keywords):
            self.f0g, self.Qg, self.idf0, self.iddf = estpara3(self.f,self.z1,verbose=self.verbose,result0=self.result0)
        else:
            self.f0g, self.Qg, self.idf0, self.iddf = estpara3(self.f,self.z1,verbose=self.verbose)
    
        # fit circle using data points f0g +- 2*f0g/Qg
        self.id1 = np.amax([self.idf0-2*self.iddf,0])
        self.id2 = np.amin([self.idf0+2*self.iddf,len(self.f)])
    
        if self.id2 < len(self.f):
            self.id2 = self.id2 + 1
    
        #r, zc, c_residue = circlefittest(z1[id1:id2])
        circle_obj = Circlefit(self.z1[self.id1:self.id2])
        self.r = float(circle_obj.r)
        self.zc = circle_obj.zc
        self.c_residue = circle_obj.residue
    
        self.theta_zc = np.angle(self.zc)

        # rotation and translation to center
        self.z2 = (self.zc - self.z1)*np.exp(-1.j*self.theta_zc) # same as center_circle function

        self.ft, self.zt = trimdata(self.f,self.z2,self.f0g,self.Qg,self.numspan) # estv['f0'],estv['Q']
        self.f0gid = find_nearest_indice(self.f,self.f0g)


    def fphaselin(self, ft, Q, f0, phi): # f is the independent variable here
        xg = (ft/f0) - 1. # also written as (f-f0)/f0
    
        ang = phi + 2.*np.arctan(-2.*Q*xg)
        return ang


    def calcphasebounds(self, Q, f0, phi, fmargin = 0.02, phimargin = 0.05):
        # order is Q, f0, and phi
        bounds = ((1000., f0-fmargin*f0, phi-phimargin*np.abs(phi)), (1e7, f0+fmargin*f0, phi+phimargin*np.abs(phi)))
        return bounds


    def windowlorentz(self,f):
        # f_c is the resonant frequency or probe tone frequency + LO
        f_c = self.tone_freq_lo
        x = 2. / self.window_width * (f - f_c) # 2/W * (f - f_c)
        uncertainty = np.sqrt(1. + np.square(x)) # sqrt(1 + x^2)
        return uncertainty


    def windowgauss(self,f):
        # f_c is the resonant frequency or probe tone frequency + LO
        f_c = self.tone_freq_lo
        x = 2. / self.window_width * (f - f_c) # 2/W * (f - f_c)
        uncertainty = np.exp(np.square(x) / 2.) # exp(x^2 / 2)
        return uncertainty


    def fitphasetest(self, ft, angt, Q, f0, phi, **keywords):
    
        p0 = [Q, f0, phi]
        
        if self.use_weight:
            if self.weight_type == 'lorentz':
                uncertainty_angt = self.windowlorentz(ft)
            elif self.weight_type == 'gauss':
                uncertainty_angt = self.windowgauss(ft)
            else:
                uncertainty_angt = np.ones(np.shape(ft))
        else:
            uncertainty_angt = np.ones(np.shape(ft))
    
        if ('bounds' in keywords):
            bounds = keywords['bounds']
            if self.verbose:
                print('Fitting data with user-specified bounds')
                print(bounds)
            popt, pcov = curve_fit(self.fphaselin, xdata=ft, ydata=angt, p0=p0, bounds=bounds)
        elif self.use_weight:
            print('Fitting data with weighting')
            popt, pcov = curve_fit(self.fphaselin, xdata=ft, ydata=angt, p0=p0, sigma=uncertainty_angt, absolute_sigma=False)
        else:
            if self.verbose:
                print('Fitting data')
            popt, pcov = curve_fit(self.fphaselin, xdata=ft, ydata=angt, p0=p0)
    
        fit_result = self.fphaselin(ft,popt[0],popt[1],popt[2])
        p0_result = self.fphaselin(ft,p0[0],p0[1],p0[2])
    
        return popt, pcov, fit_result, p0_result, uncertainty_angt


    def fitphase2(self,f,z,f0g,Qg,numspan=2,showplot=False,use_bounds=False):
        # give z2 for correct result
    
        zinf = (z[0]+z[-1])/2.
        ezinf = zinf/np.abs(zinf)

        z1_1 = z/(-ezinf)
        ang = phase2(z1_1) + np.angle(-ezinf)

        hnumsmopts = int(np.floor((f0g/Qg/3.)/(f[1]-f[0])))
        hnumsmopts = int(np.amax([np.amin([hnumsmopts, 20]),0]))

        if hnumsmopts-1 < 0:
            hnumsmopts_l = 0
        else:
            hnumsmopts_l = hnumsmopts-1

        N = int(hnumsmopts*2)
        zm = smoothdata2(z,N)

        zm_inf = (zm[0]+zm[-1])/2.
        ezminf = zm_inf/np.abs(zm_inf)
        zm1_1 = zm/(-ezminf)
        angm = phase2(zm1_1) + np.angle(-ezminf)
    
        dangm = angm[hnumsmopts_l:len(angm)] - angm[0:len(angm)-hnumsmopts+1]

        fd = f[hnumsmopts_l:len(angm)] - f[0:len(angm)-hnumsmopts+1]

        dangmin = np.amin(dangm[hnumsmopts_l:len(dangm)-hnumsmopts+1])
        idx = np.argmin(dangm[hnumsmopts_l:len(dangm)-hnumsmopts+1])

        f0 = f[idx + hnumsmopts]
        # print(f0) # identical to matlab

        fstep = f[1]-f[0]
        idx = find_nearest_indice(f, f0)
        wid = int(np.floor(f0/Qg/3./fstep))

        fmid = f[np.amax([idx-wid,0]):np.amin([idx+wid+1,len(f)])]
    
        xmid = (f0-fmid)/f0
        angmid = ang[np.amax([idx-wid,0]):np.amin([idx+wid+1,len(f)])]

        slope, intercept, r, p, se = scipy.stats.linregress(xmid,angmid)

        phi = intercept
        Q = np.abs(slope/4.)
    
        ft, zt = trimdata(f,z,f0,Q,numspan)
        ft, angt = trimdata(f,ang,f0,Q,numspan)
    
        if use_bounds:
            bounds = self.calcphasebounds(Q, f0, phi)
            popt, pcov, fit_result, p0_result, uncertainty_angt = self.fitphasetest(ft, angt, Q, f0, phi, bounds = bounds)
        else:
            popt, pcov, fit_result, p0_result, uncertainty_angt = self.fitphasetest(ft, angt, Q, f0, phi)

        # order is Q, f0, phi
        fphasetest = self.fphaselin(ft, popt[0], popt[1], popt[2])

        if self.verbose:
            print('initial guess for f0 and Q (given): ', f0g, Qg)
            print('guess for f0 from smoothed data: ', f0)
            print('guess from linear fit for phi, Q: ', phi,Q)
            print('fit results for Q, f0, and phi are ', popt)
    
        pherr = angt - fphasetest #np.abs(angt - fphasetest)

        return popt, pcov, pherr, fit_result, p0_result, ft, zt, angt, ang, uncertainty_angt, ezinf


    def fits21sim(self,showplot=False,use_bounds=False):
    
        result = {}
        result['tau'] = self.tau

        numspan = self.numspan
    
        z1 = removecable(self.f,self.z,self.tau,verbose=self.verbose)
    
        f0g, Qg, idf0, iddf = estpara3(self.f,z1,verbose=self.verbose)
    
        # fit circle using data points f0g +- 2*f0g/Qg
        id1 = np.amax([idf0-2*iddf,0])
        id2 = np.amin([idf0+2*iddf,len(self.f)])
    
        if id2 < len(self.f): # this is needed I think
            id2 = id2 + 1
    
        #r, zc, c_residue
        circle_obj = Circlefit(z1[id1:id2])
        result['r'] = float(circle_obj.r)
        result['zc'] = circle_obj.zc
        result['c_residue'] = circle_obj.residue
    
        if self.verbose:
            print('f0, Q, idf0, iddf guess ', f0g, Qg, idf0, iddf) # matches matlab
            print('r, zc: ', result['r'],result['zc'])

        theta_zc = np.angle(result['zc'])

        # rotation and translation to center
        z2 = (result['zc'] - z1)*np.exp(-1.j*theta_zc)
    
        # now fit to data below
        # order is Q, f0, phi
        if use_bounds:
            popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, uncertainty_angt, ezinf = self.fitphase2(self.f,z2,f0g,Qg,numspan,use_bounds=use_bounds)
        else:
            popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, uncertainty_angt, ezinf = self.fitphase2(self.f,z2,f0g,Qg,numspan)
    
        result['popt'] = popt
        result['Q'] = popt[0]
        result['f0'] = popt[1]
        result['phi'] = popt[2]
    
        result['pcov'] = pcov
        result['pherr'] = pherr
        result['popt_err'] = np.sqrt(np.diag(pcov)) # one standard deviation errors of fit parameters # std
    
        result['fit_result'] = fit_result
        result['fit_result2'] = self.fphaselin(self.f,result['Q'],result['f0'],result['phi'])
        result['fit_resultt'] = self.fphaselin(ft,result['Q'],result['f0'],result['phi']) # Q, f0, phi
        result['x0_result'] = x0_result
    
        result['ft'] = ft
        result['zt'] = zt
        result['ang'] = ang
        result['angt'] = angt
        result['uncertainty_angt'] = uncertainty_angt
        result['ezinf'] = ezinf
    
        idft1 = find_nearest_indice(self.f,ft[0])
        widft = len(ft)

        zt_n = self.z[idft1:idft1+widft] # idft1+widft-1 before, matches matlab
        ft_n = self.f[idft1:idft1+widft]
        result['zt_n'] = zt_n
        result['ft_n'] = ft_n

        result['zd'] = -result['zc']/np.abs(result['zc'])*np.exp(1.j*result['phi'])*result['r']*2.
        result['zinf'] = result['zc'] - result['zd']/2.
    
        result['Qc'] = np.abs(result['zinf'])/np.abs(result['zd'])*result['Q']
        result['Qi'] = 1./(1./result['Q']-1./result['Qc'])
    
        result['zf0'] = result['zinf'] + result['zd']

        #projection of zf0 into zinf
        l = np.real(result['zf0']*np.conj(result['zinf']))/np.abs(result['zinf'])**2

        # Q/Qi0 = l
        result['Qi0'] = result['Q']/l

        result['f00id'] = find_nearest_indice(self.f, result['f0'])
        result['f0_corr'] = result['f0'] # same for linear version
        result['f0id'] = result['f00id'] # same for linear version
        
        # for converting from ang to IQ space
        iq_model = 1. - (((result['Q']/result['Qc'])*np.exp(1.j*result['phi']))/(1. + 2.j*result['Q']*((self.f/result['f0']) - 1.)))
        iq_modelt = 1. - (((result['Q']/result['Qc'])*np.exp(1.j*result['phi']))/(1. + 2.j*result['Q']*((result['ft']/result['f0']) - 1.)))
        result['iq_model'] = iq_model # z2 = A*iq_model --> A = z2/iq_model
        result['iq_modelt'] = iq_modelt
        
        result['A_mag'] = result['r']
        result['A_magt'] = result['r']
        
        ang_to_iq = result['A_mag']*(np.cos(result['fit_result2']-np.angle(-result['ezinf'])) + 1.j*np.sin(result['fit_result2']-np.angle(-result['ezinf'])))*np.abs(result['iq_model'])
        angt_to_iq = result['A_magt']*(np.cos(result['fit_resultt']-np.angle(-result['ezinf'])) + 1.j*np.sin(result['fit_resultt']-np.angle(-result['ezinf'])))*np.abs(result['iq_modelt'])
        result['ang_to_z2'] = ang_to_iq*-result['ezinf']
        result['angt_to_z2'] = angt_to_iq*-result['ezinf']
        
        result['ang_to_z1'] = result['zc'] - (result['ang_to_z2']/np.exp(-1.j*np.angle(result['zc'])))
        result['angt_to_z1'] = result['zc'] - (result['angt_to_z2']/np.exp(-1.j*np.angle(result['zc'])))
        
        result['ang_to_z'] = result['ang_to_z1']/(np.exp(-1.j*2.*np.pi*(self.f/1e9)*self.tau)) # f in GHz, tau (negative) in ns
        result['angt_to_z'] = result['angt_to_z1']/(np.exp(-1.j*2.*np.pi*(result['ft']/1e9)*self.tau)) # f in GHz, tau (negative) in ns
    
        return result

 
    def fphasenlin(self, f, Q, alpha, f00, phi, r, z): # f is the independent variable here
        # self, f, Q, alpha, f00, phi, r, z
        A = z - r*np.exp(1.j*(phi+np.pi))
        f0 = f00 - alpha*np.abs(A)**2
        x = (f - f0)/f0
    
        ang = phi + 2.*np.arctan(-2.*Q*x)
        return ang


    def fphasenlin2(self, f, Q, alpha, f00, phi, r, z): # f is the independent variable here
        # self, f, Q, alpha, f00, phi, r, z
        A = z - r*np.exp(1.j*(phi+np.pi))
        f0 = f00 - alpha*np.abs(A)**2
        x = (f - f0)/f0
    
        ang = phi + 2.*np.arctan(-2.*Q*x)
        return ang, x


    def calcphasenlinbounds(self, Q, alpha, f0, phi, fmargin = 0.02, phimargin = 0.05):
        # order is Q, alpha, f0, and phi
        bounds = ((1000., -np.inf, f0-fmargin*f0, phi-phimargin*np.abs(phi)), (1e7, np.inf, f0+fmargin*f0, phi+phimargin*np.abs(phi)))
        return bounds


    def fitphasenlintest2(self, ft, angt, Q, alpha, f0, phi, r, zt, **keywords):
    
        #f0_corr = f01 - alpha*(2.*r)**2 # new, 1/13/23
        p0 = [Q, alpha, f0, phi] #[Q, alpha, f0_corr, phi]
        
        if self.use_weight:
            if self.weight_type == 'lorentz':
                uncertainty_angt = self.windowlorentz(ft)
            elif self.weight_type == 'gauss':
                uncertainty_angt = self.windowgauss(ft)
            else:
                uncertainty_angt = np.ones(np.shape(ft))
        else:
            uncertainty_angt = np.ones(np.shape(ft))

        if ('bounds' in keywords):
            if self.verbose:
                print('Fitting data with user-specified bounds')
                print(bounds)
            bounds = keywords['bounds']
            if ('err' in keywords):
                if self.verbose:
                    print('Using calculated error')
                err = keywords['err']
                popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0,
                               sigma=err, bounds=bounds)
            else:
                try:
                    popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0,
                                   bounds=bounds)
                except RuntimeError:
                    print('Failed to fit')
                    pass
        else:
            if self.verbose:
                print('Fitting data')
            if ('err' in keywords):
                if self.verbose:
                    print('Using calculated error')
                err = keywords['err']
                # this is giving me issues
                popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0,sigma=err)
            elif self.use_weight:
                print('Fitting data with weighting')
                try:
                    popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0, sigma=uncertainty_angt, absolute_sigma=False)
                    fit_result, xt = self.fphasenlin2(ft,popt[0],popt[1],popt[2],popt[3],r,zt)
                except RuntimeError:
                    print('Failed to fit data')
                    popt = [1.,1.,1.,1.]
                    pcov = [1.,1.,1.,1.]
                    fit_result = 1.
                    xt = 1.
            else:
                try:
                    popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0)
                    fit_result, xt = self.fphasenlin2(ft,popt[0],popt[1],popt[2],popt[3],r,zt)
                except RuntimeError:
                    print('Failed to fit data')
                    popt = [1.,1.,1.,1.]
                    pcov = [1.,1.,1.,1.]
                    fit_result = 1.
                    xt = 1.

        p0_result = self.fphasenlin(ft,p0[0],p0[1],p0[2],p0[3],r,zt)
        return popt, pcov, fit_result, p0_result, xt, uncertainty_angt


    def fitphasenlin_c2(self,f,z,estv,numspan=2,use_bounds=False):
    
        zinf = (z[0]+z[-1])/2.
        ezinf = zinf/np.abs(zinf)

        z1_1 = z/(-ezinf) # z1 in matlab, renamed due to redundancy

        ang = phase2(z1_1) + np.angle(-ezinf)

        # linear regime used with estpara # -20 dBm data and linear fit # import data to test for now
        ft, angt = trimdata(f,ang,estv['f0'],estv['Q'],numspan)
        ft, zt = trimdata(f,z,estv['f0'],estv['Q'],numspan)

        f00g = estv['f0']
        Qgg = estv['Q'] # originally Qg
        z1_n = np.mean(zt[:5]) # z1
        z2_n = np.mean(zt[-5:]) # z2
        v = np.amax(np.abs(zt-z1_n) + np.abs(zt-z2_n))
        idf0 = np.argmax(np.abs(zt-z1_n) + np.abs(zt-z2_n))
        phig = estv['phi']
        alphag = np.abs(ft[idf0] - f00g)/(2.*estv['r'])**2

        # order is Q, alpha, f0, phi
        # not fitting to estv['r'], zt
        if use_bounds:
            bounds = self.calcphasenlinbounds(Qgg, alphag, f00g, phig)
            popt, pcov, fit_result, x0_result, xt, uncertainty_angt = self.fitphasenlintest2(ft, angt, Qgg, alphag, f00g, phig, estv['r'], zt, 
                                                                bounds = bounds)
        else:
            popt, pcov, fit_result, x0_result, xt, uncertainty_angt = self.fitphasenlintest2(ft, angt, Qgg, alphag, f00g, phig, estv['r'], zt)

        if np.mean(popt) == 1.:
            flag = 1.
        else:
            flag = 0

        if self.verbose:
            print('fit results for Q, alpha, f0, and phi are ', popt)

        # get error
        var = np.sum((angt-fit_result)**2)/(angt.shape[0] - 1)
        err = np.ones(angt.shape[0])*np.sqrt(var)

        pherr = angt - fit_result #np.abs(angt - fit_result)
        f00 = popt[2]
    
        return popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, var, err, xt, flag, uncertainty_angt, ezinf


    def fits21simnlin(self,use_bounds=False,showplot=False):
        # assumes data has already been loaded and corrected
    
        numspan = self.numspan

        result = {}
        result['tau'] = self.tau
    
        z1 = removecable(self.f,self.z,self.tau,verbose=self.verbose)
    
        f0g, Qg, idf0, iddf = estpara3(self.f,z1,verbose=self.verbose,result0=self.result0)
    
        # fit circle using data points f0g +- 2*f0g/Qg
        id1 = np.amax([idf0-2*iddf,0])
        id2 = np.amin([idf0+2*iddf,len(self.f)])
    
        if id2 < len(self.f):
            id2 = id2 + 1

        #r, zc, c_residue
        circle_obj = Circlefit(z1[id1:id2])
        result['r'] = float(circle_obj.r)
        result['zc'] = circle_obj.zc
        result['c_residue'] = circle_obj.residue
    
        if self.verbose:
            print('f0, Q, idf0, iddf guess ', f0g, Qg, idf0, iddf) # matches matlab
            print('r, zc: ', result['r'],result['zc'])

        # create estv
        estv = {}
        estv['f0'] = f0g
        estv['Q'] = Qg
        estv['r'] = result['r']
        estv['phi'] = self.result0['phi']

        theta_zc = np.angle(result['zc'])

        # rotation and translation to center
        z2 = (result['zc'] - z1)*np.exp(-1.j*theta_zc)
    
        # now fit to data below
        if use_bounds:
            popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, var, err, xt, flag, uncertainty_angt, ezinf = self.fitphasenlin_c2(self.f,z2,estv,numspan,use_bounds=use_bounds)
        else:
            popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, var, err, xt, flag, uncertainty_angt, ezinf = self.fitphasenlin_c2(self.f,z2,estv,numspan)
    
        result['popt'] = popt # Q, alpha, f0, phi
        result['Q'] = popt[0]
        result['alpha'] = popt[1]
        result['f0'] = popt[2]
        result['phi'] = popt[3]
    
        result['pcov'] = pcov
        result['pherr'] = pherr
        result['popt_err'] = np.sqrt(np.diag(pcov)) # one standard deviation errors of fit parameters # std
    
        result['fit_result'] = fit_result # on trimmed data, ft, zt
        result['fit_result2'] = self.fphasenlin(self.f,result['Q'],result['alpha'],result['f0'],result['phi'],result['r'],z2) # Q, alpha, f0, phi # fit ang
        result['fit_resultt'] = self.fphasenlin(ft,result['Q'],result['alpha'],result['f0'],result['phi'],result['r'],zt) # fit ang
        result['x0_result'] = x0_result
    
        result['ft'] = ft
        result['zt'] = zt
        result['ang'] = ang
        result['angt'] = angt
        result['uncertainty_angt'] = uncertainty_angt
        result['ezinf'] = ezinf

        result['fit_var'] = var
        result['fit_err'] = err
        result['x'] = xt
    
        idft1 = find_nearest_indice(self.f,ft[0])
        widft = len(ft)
    
        zt_n = self.z[idft1:idft1+widft] # idft1+widft-1 before, matches matlab
        ft_n = self.f[idft1:idft1+widft]
        result['zt_n'] = zt_n
        result['ft_n'] = ft_n

        result['zd'] = -result['zc']/np.abs(result['zc'])*np.exp(1.j*result['phi'])*result['r']*2
    
        result['zinf'] = result['zc'] - result['zd']/2.

        Qc_part = np.abs(np.abs(result['zc'])/(2.*result['r']*np.exp(1.j*result['phi'])) + 0.5)
        result['Qc'] = Qc_part*result['Q']

        result['Qi'] = 1./(1./result['Q']-1./result['Qc'])
        result['bif'] = (result['alpha']*(2.*result['r'])**2)/(result['f0']/result['Q'])
        result['dff0'] = result['alpha']*(2.*result['r'])**2/result['f0']
        result['zf0'] = result['zinf'] + result['zd']

        # ignored delta r term for now
        result['bif_err'] = np.sqrt((4.*result['r']**2*result['Q']/result['f0'])**2*result['popt_err'][1]**2 +
                                   (-4.*result['r']**2*result['alpha']*result['Q']/result['f0']**2)**2*result['popt_err'][2]**2 +
                                   (4.*result['r']**2*result['alpha']/result['f0'])**2*result['popt_err'][0]**2)

        result['Qc_err'] = np.sqrt((np.abs(np.abs(result['zc'])*-1.j/(2.*result['r']*np.exp(1.j*result['phi'])))*
                                    result['Q'])**2*result['popt_err'][3]**2 + 
                                    np.abs(np.abs(result['zc'])/(2.*result['r']*np.exp(1.j*result['phi'])) + 0.5)**2*
                                    result['popt_err'][0]**2)

        result['Qi_err'] = np.sqrt((Qc_part/(Qc_part-1.))**2*result['popt_err'][0]**2 + 
                                    (-1./((1./result['Q']) - (1./(Qc_part*result['Q'])))**2*(1./(Qc_part**2*result['Q']))*
                                     np.abs(-1.j*np.abs(result['zc'])/(2.*result['r']*np.exp(1.j*result['phi']))))**2*
                                    result['popt_err'][3]**2)


        # ignored delta zc and delta r terms for now
        result['zd_err'] = np.sqrt(((-result['zc']/np.abs(result['zc']))*2.*result['r']*1.j*np.exp(1.j*result['phi']))**2*result['popt_err'][3]**2)
        result['zinf_err'] = np.sqrt((result['zc']/np.abs(result['zc'])*1.j*np.exp(1.j*result['phi'])*result['r'])**2*result['popt_err'][3]**2)

        #projection of zf0 into zinf
        l = np.real(result['zf0']*np.conj(result['zinf']))/np.abs(result['zinf'])**2
        result['l'] = l
        # ignored delta zc term for now, should only be real
        result['l_err'] = np.sqrt(np.real(result['zc']/result['zinf']**2)**2*np.abs(result['zd_err'])**2)
    
    
        # Q/Qi0 = l
        result['Qi0'] = result['Q']/l
        result['Qi0_err'] = np.sqrt(result['popt_err'][0]**2/result['l']**2 + result['Q']**2/result['l']**4*result['l_err']**2)

        
        # f0 from fit
        result['f00id'] = find_nearest_indice(self.f, result['f0']) # fit f0 called f00 in matlab code

        # similar to corrected f0 in fphasenlin
        result['f0_corr'] = result['f0'] - result['alpha']*(2.*estv['r'])**2
        result['f0id'] = find_nearest_indice(self.f,result['f0_corr'])
        #print('f0id',result['f0id'])

        if flag == 0:
            if sum(np.abs(result['pherr']) > self.pherr_threshold) > self.pherr_threshold_num:
                flag = 2

        result['flag'] = flag
        
        # for converting from ang to IQ space
        iq_model = 1. - (((result['Q']/result['Qc'])*np.exp(1.j*result['phi']))/(1. + 2.j*result['Q']*((self.f/result['f0_corr']) - 1.)))
        iq_modelt = 1. - (((result['Q']/result['Qc'])*np.exp(1.j*result['phi']))/(1. + 2.j*result['Q']*((result['ft']/result['f0_corr']) - 1.)))
        result['iq_model'] = iq_model # z2 = A*iq_model --> A = z2/iq_model
        result['iq_modelt'] = iq_modelt
        
        result['A_mag'] = result['r']
        result['A_magt'] = result['r']
        
        ang_to_iq = result['A_mag']*(np.cos(result['fit_result2']-np.angle(-result['ezinf'])) + 1.j*np.sin(result['fit_result2']-np.angle(-result['ezinf'])))
        angt_to_iq = result['A_magt']*(np.cos(result['fit_resultt']-np.angle(-result['ezinf'])) + 1.j*np.sin(result['fit_resultt']-np.angle(-result['ezinf'])))
        result['ang_to_z2'] = ang_to_iq*-result['ezinf']
        result['angt_to_z2'] = angt_to_iq*-result['ezinf']
        
        result['ang_to_z1'] = result['zc'] - (result['ang_to_z2']/np.exp(-1.j*np.angle(result['zc'])))
        result['angt_to_z1'] = result['zc'] - (result['angt_to_z2']/np.exp(-1.j*np.angle(result['zc'])))
        
        result['ang_to_z'] = result['ang_to_z1']/(np.exp(-1.j*2.*np.pi*(self.f/1e9)*self.tau)) # f in GHz, tau (negative) in ns
        result['angt_to_z'] = result['angt_to_z1']/(np.exp(-1.j*2.*np.pi*(result['ft']/1e9)*self.tau)) # f in GHz, tau (negative) in ns
    
        return result
    
    
    # helper functions based on Hannes's code
    def plotComplexPlane(self,z):
        plt.plot(np.real(z),np.imag(z))
        plt.xlabel('real')
        plt.ylabel('imag')

    
    def rescaleCircle(self,z,r):
        return z/float(r)


    def rotateCircle(self,z,phi):
        return z*np.e**(-1.j*phi/180.*np.pi) # or np.exp


    def plotPolar(self,z):
        plt.subplot(111, polar=True)
        plt.plot(phase2(z),np.abs(z)) # np.arctan2(np.imag(z),np.real(z))


    def plot(self):
        # make the fitted array for plotting purposes

        s = np.linspace(0,2.*np.pi,100)
        z_circlefit = self.zc+self.r*np.e**(1.j*s)
        z_circlefit_mini = self.zc+self.r*0.001*np.e**(1.j*s)
        fig = plt.subplots(2, 2)

        # upper left, s21 magnitude.  Raw data, trimmed and fit.
        plt.subplot(221)
        plt.plot(self.f,logmag(self.z1)) #,'o-') # bo
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$|S_{21}|^2$')


        plt.subplot(222,polar=True)
        plt.plot(phase2(self.z),np.abs(self.z),'c')
        plt.plot(phase2(self.z1),np.abs(self.z1),'blue')
        plt.plot(phase2(z_circlefit),np.abs(z_circlefit),'r--') # z1[id1:id2] # k--
        plt.plot(phase2(z_circlefit_mini),np.abs(z_circlefit_mini),'r+')
        plt.plot(np.angle(self.result['zinf']),np.abs(self.result['zinf']),'k*')
        plt.plot(np.angle(self.result['zf0']),np.abs(self.result['zf0']),'ko')
        plt.plot(phase2(self.z2),np.abs(self.z2),'b')
        plt.plot(phase2(self.zt),np.abs(self.zt),'m')

        # similar to corrected f0 in fphasenlin
        plt.plot(self.result['ang'][self.result['f0id']],np.abs(self.z2[self.result['f0id']]),'ko')
        # f0 from fit
        plt.plot(self.result['ang'][self.result['f00id']],np.abs(self.z2[self.result['f00id']]),'mo')


        plt.subplot(223)
        plt.plot(self.f,self.result['ang'],'.')
        color = plt.gca().lines[-1].get_color() # get most recent line color

        plt.plot(self.result['ft'],self.result['angt'],'o',color=color)
        plt.plot(self.result['ft'],self.result['fit_result'],'k--')

        # trimmed data that was fit
        plt.plot(self.result['ft'][0],self.result['angt'][0],'c+')
        plt.plot(self.result['ft'][-1],self.result['angt'][-1],'c+')

        # similar to corrected f0 in fphasenlin
        plt.plot(self.f[self.result['f0id']],self.result['ang'][self.result['f0id']],'ko')
        # f0 from fit
        plt.plot(self.f[self.result['f00id']],self.result['ang'][self.result['f00id']],'mo')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (rad)')


        plt.subplot(224)
        plt.plot(self.result['ft'],self.result['pherr'])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Residuals (rad)')

        plt.tight_layout()
        plt.show()


class ResonanceFitterSingleTonePowSweep():
    '''
        Class to load resonator data and perform a simple or 
        nonlinear phase fit (user-specified) on a single tone,
        now for a power sweep (compared to class ResonanceFitterSingleTone)
    '''
    def __init__(self,filedir,data,res,tone_freq_lo,powlist_full,powlist,\
                 tau,numspan=2,window_width=0,a_predict_pow=-11.,\
                 a_predict_guess=0.2,a_predict_threshold=0.2,\
                 pherr_threshold_num = 10,pherr_threshold=0.2,\
                 verbose=False,save_fig=False,save_pdf=False,\
                 filename='toltec',**keywords):
        # f in Hz (will also work with GHz?), z is complex, tau in ns
        '''
            data needs to be complex (like I + jQ)
        '''

        self.filedir = filedir # if filedir == 0, use data
        self.data = data # if data == 0, use filedir, can we make this even clearer?
        self.res = res # integer referring to resonator number
        self.tone_freq_lo = tone_freq_lo #[0][self.res]
        self.powlist_full = powlist_full
        self.powlist = powlist # in dBm
        self.powlist_W = 10.**(np.asarray(powlist)/10.)/1000.

        self.intersect_powlist,self.indices_powlist,self.indices_powlist_full = np.intersect1d(powlist,powlist_full,return_indices=True)
        self.indices_0 = self.indices_powlist_full[0]

        self.tau = tau
        self.numspan = numspan

        self.a_predict_pow = a_predict_pow # in dBm
        self.a_predict_guess = a_predict_guess
        self.a_predict_threshold = a_predict_threshold

        self.pherr_threshold = pherr_threshold # as a percentage of the phase fit residuals # sets pherr threshold
        self.pherr_threshold_num = int(pherr_threshold_num) # sets number of points allowed above pherr threshold before flagging

        # new argument for specifying output to print to user
        self.verbose = verbose

        self.save_fig = save_fig
        self.save_pdf = save_pdf
        self.filename = filename
        
        self.window_width = window_width

        # makes a dictionary with each power as a key, seem to work
        self.result = {}
        self.f = {}
        self.z = {}
        self.z1 = {}

        self.f0g = {}
        self.Qg = {}
        self.idf0 = {}
        self.iddf = {}

        self.id1 = {}
        self.id2 = {}

        self.r = {}
        self.zc = {}
        self.c_residue = {}
        self.theta_zc = {}
        self.z2 = {}

        self.ft = {}
        self.zt = {}

        self.f0gid = {}

        self.omega_r = {}
        self.E_star = {}
        self.E_star_list = []
        self.bif_list = []
        self.flag_list = []

        self.fit_flag = {}
        self.fit_flag_list = []
        
        if ('weight_type' in keywords):
            self.weight_type = keywords['weight_type']
            self.use_weight = True
            if window_width == 0:
                print('To use weighting, need to specify a window width')
                self.use_weight = False
        else:
            self.use_weight = False

        if ('result0' in keywords):
            self.result0 = keywords['result0']
        elif self.filedir == 0: # use self.data
            #print('Fitting data at ' + str(drive_atten[kk]) + ' dBm')
            z0_mag_test =  magS21(self.data[self.indices_0][:,self.res][:,1],self.data[self.indices_0][:,self.res][:,2])
            z1_mag_test = magS21(self.data[self.indices_0+1][:,self.res][:,1],self.data[self.indices_0+1][:,self.res][:,2])

            if (np.abs(np.mean(z0_mag_test)-np.amin(z0_mag_test)) < 2e4): # make user variable
                extra_one = 1
                print('USING second lowest power instead for fit')
                print('lowest pow indice',self.indices_0+extra_one)
                self.powlist = self.powlist[1:]
                self.powlist_W = self.powlist_W[1:]
                data_0 = self.data[self.indices_0+extra_one]
                fit_pows = range(0,len(self.powlist))
            else:
                print('lowest pow indice',self.indices_0)
                extra_one = 0
                data_0 = self.data[self.indices_0] # 0
                fit_pows = range(0,len(self.powlist))
            f_temp_0 = data_0[:,self.res][:,0]
            z_temp_0 = data_0[:,self.res][:,1] + 1.j*data_0[:,self.res][:,2]
        elif self.data == 0: # use self.filedir
            #kk = self.powlist[0] #-20 # str(kk)
            data = loadmat(self.filedir + str(self.powlist[0]) + '.0dBm.mat') # need to make this more modular
            f_temp = data['f'][:,self.res]
            z_temp = data['z'][:,self.res] # tone_freq_lo=0,window_width=0

            if self.use_weight:
                self.result0 = ResonanceFitterSingleTone(f_temp,z_temp,self.tau,self.numspan,self.tone_freq_lo,self.window_width,weight_type=self.weight_type).result
            else:
                self.result0 = ResonanceFitterSingleTone(f_temp,z_temp,self.tau,self.numspan).result

        #print('test',fit_pows,self.indices_0) # fit_pows = range(0,len(powlist))
        for kk in fit_pows: #range(0,len(self.powlist)): # self.powlist
            kk_p = self.powlist[kk]
            #print(self.powlist[fit_pows[-1]]) # -10 as expected
            print('Fitting ' + str(kk_p) + ' dBm data') # str(kk)
            if self.filedir == 0: # use self.data
                data_kk = self.data[self.indices_0+kk+extra_one]
                self.f[kk_p] = data_kk[:,self.res][:,0]
                z_temp = data_kk[:,self.res][:,1] + 1.j*data_kk[:,self.res][:,2]
            elif self.data == 0: # use self.filedir
                data_kk = loadmat(self.filedir + str(kk_p) + '.0dBm.mat')
                self.f[kk_p] = data_kk['f'][:,self.res]
                z_temp = data_kk['z'][:,self.res]

    
            if kk==fit_pows[0]: #kk_p==self.powlist[0]:
                self.z0 = z_temp
                if self.use_weight:
                    try:
                        self.result0 = ResonanceFitterSingleTone(f_temp_0,z_temp_0,self.tau,self.numspan,self.tone_freq_lo,self.window_width,weight_type=self.weight_type).result
                        self.fit_flag[kk_p] = 0
                        pass
                    except:
                        self.fit_flag[kk_p] = 1
                        continue
                else:
                    try:
                        self.result0 = ResonanceFitterSingleTone(f_temp_0,z_temp_0,self.tau,self.numspan).result
                        self.fit_flag[kk_p] = 0
                        pass
                    except:
                        self.fit_flag[kk_p] = 1
                        continue

            self.corr = self.z0[-1]/z_temp[-1] #self.z0[0]/z_temp[0]
            self.z[kk_p] = z_temp*self.corr
            if self.use_weight:
                try:
                    self.result[kk_p] = ResonanceFitterSingleTone(self.f[kk_p],self.z[kk_p],self.tau,self.numspan,self.tone_freq_lo,self.window_width,pherr_threshold_num=self.pherr_threshold_num,pherr_threshold=self.pherr_threshold,result0=self.result0,weight_type=self.weight_type)
                    self.fit_flag[kk_p] = 0
                    pass
                except:
                    print('Skipping fit for this resonator')
                    self.fit_flag[kk_p] = 1
                    continue
            else:
                try:
                    self.result[kk_p] = ResonanceFitterSingleTone(self.f[kk_p],self.z[kk_p],self.tau,self.numspan,self.tone_freq_lo,result0=self.result0)
                    self.fit_flag[kk_p] = 0
                    pass
                except:
                    print('Skipping fit for this resonator')
                    self.fit_flag[kk_p] = 1
                    continue
            # from the fit so need to be after it
            if kk==fit_pows[0]: #kk_p==self.powlist[0]:
                self.Q_0 = self.result[kk_p].result['Q']
                self.Qc_0 = self.result[kk_p].result['Qc']
                self.bif_0 = self.result[kk_p].result['bif']
                self.omega_r_0 = 2.*np.pi*self.result[kk_p].result['f0_corr']

            self.omega_r[kk_p] = self.omega_r_0 + (2.*np.pi*self.result[kk_p].result['f0_corr'] - self.omega_r_0)
            self.E_star[kk_p] = 2.*self.result[kk_p].result['Q']**3*self.powlist_W[kk]/(self.result[kk_p].result['Qc']*self.omega_r[kk_p]*self.result[kk_p].result['bif'])

            self.fit_flag_list.append(self.fit_flag[kk_p])
            self.E_star_list.append(self.E_star[kk_p])
            self.bif_list.append(self.result[kk_p].result['bif'])
            self.flag_list.append(self.result[kk_p].result['flag'])

            self.z1[kk_p] = removecable(self.f[kk_p],self.z[kk_p],self.tau,verbose=self.verbose)
            self.f0g[kk_p], self.Qg[kk_p], self.idf0[kk_p], self.iddf[kk_p] = estpara3(self.f[kk_p],self.z1[kk_p],verbose=self.verbose,result0=self.result0)
    
            # fit circle using data points f0g +- 2*f0g/Qg
            self.id1[kk_p] = np.amax([self.idf0[kk_p]-2*self.iddf[kk_p],0])
            self.id2[kk_p] = np.amin([self.idf0[kk_p]+2*self.iddf[kk_p],len(self.f[kk_p])])
    
            if self.id2[kk_p] < len(self.f[kk_p]):
                self.id2[kk_p] = self.id2[kk_p] + 1
    
            #r, zc, c_residue
            circle_obj = Circlefit(self.z1[kk_p][self.id1[kk_p]:self.id2[kk_p]])
            self.r[kk_p] = float(circle_obj.r)
            self.zc[kk_p] = circle_obj.zc
            self.c_residue[kk_p] = circle_obj.residue
    
            self.theta_zc[kk_p] = np.angle(self.zc[kk_p])

            # rotation and translation to center
            self.z2[kk_p] = (self.zc[kk_p] - self.z1[kk_p])*np.exp(-1.j*self.theta_zc[kk_p]) # same as center_circle function

            self.ft[kk_p], self.zt[kk_p] = trimdata(self.f[kk_p],self.z2[kk_p],self.f0g[kk_p],self.Qg[kk_p],self.numspan) # estv['f0'],estv['Q']
            self.f0gid[kk_p] = find_nearest_indice(self.f[kk_p],self.f0g[kk_p])

        # don't calculate E_star with negative bif values for now
        self.flag_filter = np.asarray(self.flag_list[1:]) != 2 #np.all(np.asarray(self.flag_list[1:]) != 2)
        self.bif_neg_filter = np.asarray(self.bif_list[1:]) > 0 # pos only
        self.bif_pos_filter = np.asarray(self.bif_list[1:]) < 0 # neg only
        #print(self.flag_filter.size)
        #print(self.bif_list[1:])
        #print('flags', np.asarray(self.bif_list[1:])[self.flag_filter].size)
        #print('pos bif', np.asarray(self.bif_list[1:])[self.bif_neg_filter].size)
        self.bif_flag_filter = (np.asarray(self.flag_list[1:]) != 2) & (np.asarray(self.bif_list[1:]) > 0)
        print('fit flags', np.asarray(self.fit_flag_list))
        print('all flags', np.asarray(self.flag_list))
        print('all bif', np.asarray(self.bif_list))
        print('pos bif', np.asarray(self.bif_list[1:])[self.bif_flag_filter])
        if np.asarray(self.bif_list[1:])[self.flag_filter].size == 0:
             self.E_star_2 = 0
             self.a_predict = 0
             self.a_predict_pow2 = 0
             self.Pro_test_low = 0
             self.Pro_test_low_dBm = 0
             self.a_predict_off = 0
             self.Pro_test_low_guess = 0
             self.Pro_test_low_guess_dBm = 0
             self.a_predict_flag = 1
        elif np.asarray(self.bif_list[1:])[self.bif_flag_filter].size == 0:
             self.E_star_2 = 0
             self.a_predict = 0
             self.a_predict_pow2 = 0
             self.Pro_test_low = 0
             self.Pro_test_low_dBm = 0
             self.a_predict_off = 0
             self.Pro_test_low_guess = 0
             self.Pro_test_low_guess_dBm = 0
             self.a_predict_flag = 1
        else:
             self.bif_list_pos = np.abs(self.bif_list)
             self.E_star_2 = np.mean(np.asarray(self.E_star_list[1:])[self.bif_flag_filter])
             self.a_predict = np.asarray(self.bif_list[1:])[self.bif_flag_filter][0]
             self.a_predict_pow2 = np.asarray(self.powlist[1:])[self.bif_flag_filter][0]
             print('hello',self.a_predict,self.a_predict_pow2)
             print(self.E_star_2)

             self.Pro_test_low = self.Qc_0*self.omega_r_0*self.a_predict*self.E_star_2/(2.*self.Q_0**3)
             self.Pro_test_low_dBm = 10.*np.log10(self.Pro_test_low) + 30. # convert W to dBm
             self.a_predict_off = self.Pro_test_low_dBm/self.a_predict_pow2 #self.a_predict_pow
             self.Pro_test_low_guess = self.Qc_0*self.omega_r_0*self.a_predict_guess*self.E_star_2/(2.*self.Q_0**3)
             self.Pro_test_low_guess_dBm = 10.*np.log10(self.Pro_test_low_guess) + 30. # convert W to dBm
             print('checking',np.abs(self.a_predict_off-1.))
             if np.abs(self.a_predict_off-1.) > self.a_predict_threshold:
                 self.a_predict_flag = 1
             else:
                 self.a_predict_flag = 0
# check failed fit flag in file and remove
# otherwise check that we can insert a_predict_pow and calculate what to expect

    def fphasenlin2(self, f, Q, alpha, f00, phi, r, z): # f is the independent variable here
        # self, f, Q, alpha, f00, phi, r, z
        A = z - r*np.exp(1.j*(phi+np.pi))
        f0 = f00 - alpha*np.abs(A)**2
        x = (f - f0)/f0 #(f - f0)/f[0] before
    
        ang = phi + 2.*np.arctan(-2.*Q*x)
        return ang, x


    # helper functions based on Hannes's code
    def plotComplexPlane(self,z):
        plt.plot(np.real(z),np.imag(z))
        plt.xlabel('real')
        plt.ylabel('imag')


    def rescaleCircle(self,z,r):
        return z/float(r)


    def rotateCircle(self,z,phi):
        return z*np.e**(-1.j*phi/180.*np.pi) # or np.exp


    def plotPolar(self,z):
        plt.subplot(111, polar=True)
        plt.plot(phase2(z),np.abs(z)) # np.arctan2(np.imag(z),np.real(z))


    def plot(self):
        # make the fitted array for plotting purposes

        s = np.linspace(0,2.*np.pi,100)

        fig, axs = plt.subplots(2, 2)
        ax2 = plt.subplot(222,polar=True)
        for kk in self.powlist:
            if self.fit_flag[kk] == 1:
                pass
            elif self.result[kk].result['flag'] == 1.:
                pass
            else:
                z_circlefit = self.zc[kk]+self.r[kk]*np.e**(1.j*s)
                z_circlefit_mini = self.zc[kk]+self.r[kk]*0.001*np.e**(1.j*s)

                # upper left, s21 magnitude.  Raw data, trimmed and fit.
                axs[0, 0].plot(self.f[kk],logmag(self.z1[kk])) #,'o-') # bo

                ax2.plot(phase2(self.z[kk]),np.abs(self.z[kk]),'c')
                ax2.plot(phase2(self.z1[kk]),np.abs(self.z1[kk]),'blue')
                ax2.plot(phase2(z_circlefit),np.abs(z_circlefit),'r--') # z1[id1:id2] # k--
                ax2.plot(phase2(z_circlefit_mini),np.abs(z_circlefit_mini),'r+')

                ax2.plot(np.angle(self.result[kk].result['zinf']),np.abs(self.result[kk].result['zinf']),'k*')
                ax2.plot(np.angle(self.result[kk].result['zf0']),np.abs(self.result[kk].result['zf0']),'ko')
                ax2.plot(phase2(self.z2[kk]),np.abs(self.z2[kk]),'b')
                ax2.plot(phase2(self.zt[kk]),np.abs(self.zt[kk]),'m')

                # similar to corrected f0 in fphasenlin
                ax2.plot(self.result[kk].result['ang'][self.result[kk].result['f0id']],np.abs(self.z2[kk][self.result[kk].result['f0id']]),'ko')
                # f0 from fit
                ax2.plot(self.result[kk].result['ang'][self.result[kk].result['f00id']],np.abs(self.z2[kk][self.result[kk].result['f00id']]),'mo')

                color=next(ax2._get_lines.prop_cycler)['color'] # get most recent line color
                axs[1, 0].plot(self.f[kk],self.result[kk].result['ang'],'.')

                axs[1, 0].plot(self.result[kk].result['ft'],self.result[kk].result['angt'],'o',color=color)

                axs[1, 0].plot(self.result[kk].result['ft'],self.result[kk].result['fit_result'],'k--')

                # trimmed data that was fit
                axs[1, 0].plot(self.result[kk].result['ft'][0],self.result[kk].result['angt'][0],'c+')
                axs[1, 0].plot(self.result[kk].result['ft'][-1],self.result[kk].result['angt'][-1],'c+')

                # similar to corrected f0 in fphasenlin
                axs[1, 0].plot(self.f[kk][self.result[kk].result['f0id']],self.result[kk].result['ang'][self.result[kk].result['f0id']],'ko')
                # f0 from fit
                axs[1, 0].plot(self.f[kk][self.result[kk].result['f00id']],self.result[kk].result['ang'][self.result[kk].result['f00id']],'mo')

                axs[1, 1].plot(self.result[kk].result['ft'],self.result[kk].result['pherr'],marker='o') # self.angle -self.fitted_angle

            
        axs[0, 0].set_xlabel('Frequency (Hz)')
        axs[0, 0].set_ylabel('$|S_{21}|^2$')
        
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Phase (rad)')
        
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Residuals (rad)')
        
        plt.tight_layout()
        
        if self.save_fig:
            plt.savefig(self.filename + '_res' + str(self.res) + '_fits.png', format='png') # dpi=600
        elif self.save_pdf:
            pass
        else:
            plt.show()
        
    # use powlist by default?
    def plot_mag(self,show_powlist):
        if isinstance(show_powlist, int) or isinstance(show_powlist, float):
            show_powlist = [show_powlist]

        plt.figure()
        for kk in show_powlist:
            if self.fit_flag[kk] == 1:
                pass
            elif self.result[kk].result['flag'] == 1.:
                pass
            else:
                plt.plot(self.f[kk],logmag(self.z1[kk]), marker='.',linestyle='None') #,'o-') # bo
                plt.plot(self.f[kk],logmag(self.result[kk].result['ang_to_z1']),'gray')
                plt.plot(self.result[kk].result['ft'],logmag(self.result[kk].result['angt_to_z1']),'k')
                plt.plot(self.f[kk][self.result[kk].result['f0id']],logmag(self.z1[kk])[self.result[kk].result['f0id']],'ko')
                # f0 from fit
                plt.plot(self.f[kk][self.result[kk].result['f00id']],logmag(self.z1[kk])[self.result[kk].result['f00id']],'mo')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$|S_{21}|^2$')
        plt.tight_layout()
        plt.show()


    def plot_iq(self,show_powlist):
        if isinstance(show_powlist, int) or isinstance(show_powlist, float):
            show_powlist = [show_powlist]

        s = np.linspace(0,2.*np.pi,100)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for kk in show_powlist:
            if self.fit_flag[kk] == 1:
                pass
            elif self.result[kk].result['flag'] == 1.:
                pass
            else:
                z_circlefit = self.zc[kk]+self.r[kk]*np.e**(1.j*s)
                z_circlefit_mini = self.zc[kk]+self.r[kk]*0.001*np.e**(1.j*s)
            
                ax.plot(phase2(self.z[kk]),np.abs(self.z[kk]),'c')
                ax.plot(phase2(self.z1[kk]),np.abs(self.z1[kk]),'blue')
                ax.plot(phase2(z_circlefit),np.abs(z_circlefit),'r--') # z1[id1:id2] # k--
                ax.plot(phase2(z_circlefit_mini),np.abs(z_circlefit_mini),'r+')
                ax.plot(np.angle(self.result[kk].result['zinf']),np.abs(self.result[kk].result['zinf']),'k*')
                ax.plot(np.angle(self.result[kk].result['zf0']),np.abs(self.result[kk].result['zf0']),'ko')
                ax.plot(phase2(self.z2[kk]),np.abs(self.z2[kk]),'b')
                ax.plot(phase2(self.zt[kk]),np.abs(self.zt[kk]),'m')

                # similar to corrected f0 in fphasenlin
                ax.plot(self.result[kk].result['ang'][self.result[kk].result['f0id']],np.abs(self.z2[kk][self.result[kk].result['f0id']]),'ko')
                # f0 from fit
                ax.plot(self.result[kk].result['ang'][self.result[kk].result['f00id']],np.abs(self.z2[kk][self.result[kk].result['f00id']]),'mo')
        plt.tight_layout()
        plt.show()

    
    def plot_iq_rect(self,show_powlist):
        if isinstance(show_powlist, int) or isinstance(show_powlist, float):
            show_powlist = [show_powlist]

        s = np.linspace(0,2.*np.pi,100)

        plt.figure()
        for kk in show_powlist:
            if self.fit_flag[kk] == 1:
                pass
            elif self.result[kk].result['flag'] == 1.:
                pass
            else:
                # plotting z
                plt.plot(self.z[kk].real,self.z[kk].imag, marker='.',linestyle='None')
                plt.plot(self.result[kk].result['ang_to_z'].real,self.result[kk].result['ang_to_z'].imag,'gray')
                plt.plot(self.result[kk].result['angt_to_z'].real,self.result[kk].result['angt_to_z'].imag,'k')
                plt.plot(self.z[kk].real[self.result[kk].result['f0id']],self.z[kk].imag[self.result[kk].result['f0id']],'ko')
                plt.plot(self.z[kk].real[self.result[kk].result['f00id']],self.z[kk].imag[self.result[kk].result['f00id']],'mo')
                
                # plotting z1
                #plt.plot(self.z1[kk].real,self.z1[kk].imag, marker='.',linestyle='None')
                #plt.plot(self.result[kk].result['ang_to_z1'].real,self.result[kk].result['ang_to_z1'].imag,'gray')
                #plt.plot(self.result[kk].result['angt_to_z1'].real,self.result[kk].result['angt_to_z1'].imag,'black')
                #plt.plot(self.z1[kk].real[self.result[kk].result['f0id']],self.z1[kk].imag[self.result[kk].result['f0id']],'ko')
                #plt.plot(self.z1[kk].real[self.result[kk].result['f00id']],self.z1[kk].imag[self.result[kk].result['f00id']],'mo')
                
                # plotting z2
                #plt.plot(self.z2[kk].real,self.z2[kk].imag, marker='.',linestyle='None')
                #plt.plot(self.result[kk].result['ang_to_z2'].real,self.result[kk].result['ang_to_z2'].imag,'gray')
                #plt.plot(self.result[kk].result['angt_to_z2'].real,self.result[kk].result['angt_to_z2'].imag,'black')
                #plt.plot(self.z2[kk].real[self.result[kk].result['f0id']],self.z2[kk].imag[self.result[kk].result['f0id']],'ko') # 'ko'
                #plt.plot(self.z2[kk].real[self.result[kk].result['f00id']],self.z2[kk].imag[self.result[kk].result['f00id']],'mo')
        plt.xlabel('I (adu)')
        plt.ylabel('Q (adu)')
        plt.tight_layout()
        plt.show()

    def plot_phase(self,show_powlist):
        if isinstance(show_powlist, int) or isinstance(show_powlist, float):
            show_powlist = [show_powlist]

        plt.figure()
        for kk in show_powlist:
            if self.fit_flag[kk] == 1:
                pass
            elif self.result[kk].result['flag'] == 1.:
                pass
            else:
                plt.plot(self.f[kk],self.result[kk].result['ang'],'.')
                color = plt.gca().lines[-1].get_color() # get most recent line color

                plt.plot(self.result[kk].result['ft'],self.result[kk].result['angt'],'o',color=color)
                plt.plot(self.result[kk].result['ft'],self.result[kk].result['fit_result'],'k--')

                # trimmed data that was fit
                plt.plot(self.result[kk].result['ft'][0],self.result[kk].result['angt'][0],'c+')
                plt.plot(self.result[kk].result['ft'][-1],self.result[kk].result['angt'][-1],'c+')

                # similar to corrected f0 in fphasenlin
                plt.plot(self.f[kk][self.result[kk].result['f0id']],self.result[kk].result['ang'][self.result[kk].result['f0id']],'ko')
                # f0 from fit
                plt.plot(self.f[kk][self.result[kk].result['f00id']],self.result[kk].result['ang'][self.result[kk].result['f00id']],'mo')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Residuals (rad)')
        plt.tight_layout()
        plt.show()


    def plot_pherr(self,show_powlist):
        if isinstance(show_powlist, int) or isinstance(show_powlist, float):
            show_powlist = [show_powlist]

        plt.figure()
        for kk in show_powlist: # self.powlist
            if self.result[kk].result['flag'] == 1.:
                pass
            else:
                plt.plot(self.result[kk].result['ft'],self.result[kk].result['pherr']) # self.angle - self.fitted_angle
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Residuals (rad)')
        plt.tight_layout()
        plt.show()


    def plot_fit_results(self):
        # make the fitted array for plotting purposes

        fig, axs = plt.subplots(2, 2)
        for kk in self.powlist:
            if self.fit_flag[kk] == 1:
                pass
            elif self.result[kk].result['flag'] == 1.:
                pass
            else:
                # fr vs microwave power
                color=next(axs[0, 0]._get_lines.prop_cycler)['color']
                axs[0, 0].plot(kk,self.result[kk].result['f0_corr'],'o',color=color)

                # Qi vs microwave power
                axs[0, 1].plot(kk,self.result[kk].result['Qi0'],'o')

                # Qc vs microwave power
                axs[1, 0].plot(kk,self.result[kk].result['Qc'],'o')

                # a vs microwave power
                axs[1, 1].plot(kk,self.result[kk].result['bif'],'o')

        axs[0, 0].set_xlabel('Drive Power (dBm)')
        axs[0, 0].set_ylabel('$f_{r}$ (Hz)')

        axs[0, 1].set_xlabel('Drive Power (dBm)')
        axs[0, 1].set_ylabel('$Q_{i}$')

        axs[1, 0].set_xlabel('Drive Power (dBm)')
        axs[1, 0].set_ylabel('$Q_{c}$')

        axs[1, 1].set_xlabel('Drive Power (dBm)')
        axs[1, 1].set_ylabel('Nonlinearity Parameter a')

        plt.tight_layout()

        if self.save_fig:
            plt.savefig(self.filename + '_res' + str(self.res) + '_fit_results.png', format='png') # dpi=600
        elif self.save_pdf:
            pass
        else:
            plt.show()


class ToltecLOsweep():
    def __init__(self,nc_filename,use_drive_atten=False,use_sense_atten=False,use_tone_amps=False,save_tone_freq_lo=False,reorder_data=False):
        self.nc_filename = nc_filename
        self.nc = netCDF4.Dataset(self.nc_filename)
        self.use_drive_atten = use_drive_atten
        self.use_sense_atten = use_sense_atten
        self.use_tone_amps = use_tone_amps
        self.save_tone_freq_lo = save_tone_freq_lo
        self.reorder_data = reorder_data

        # pick out only what is required for the pseudo VNA and define as global to this class
        self.lo_frequencies = self.nc.variables['Data.Toltec.LoFreq'][:].data
        self.lo_freq_center = self.lo_frequencies[int(len(self.lo_frequencies)/2)]
        # some datasets have two columns of tone frequencies, [0,:] and [1,:]
        self.tone_frequencies = self.nc.variables['Header.Toltec.ToneFreq'][:].data[0,:]
        self.i_raw = self.nc.variables['Data.Toltec.Is'][:].data # 4910 x 1000 array of int32
        self.q_raw = self.nc.variables['Data.Toltec.Qs'][:].data # 4910 x 1000 array of int32
        self.f_array_hz_raw = self.make_frequency_array() # new, 12/8/22

        if self.use_drive_atten:
                self.drive_atten = self.nc.variables['Header.Toltec.DriveAtten'][:].data
        if self.use_sense_atten:
                self.sense_atten = self.nc.variables['Header.Toltec.SenseAtten'][:].data
        if self.use_tone_amps:
                self.tone_amps = self.nc.variables['Header.Toltec.ToneAmps'][:].data
        if self.save_tone_freq_lo:
                self.tone_freq_lo = self.tone_frequencies + self.lo_freq_center

        # derived globals from these inputs
        self.num_tones = len(self.tone_frequencies)
        self.num_lo_frequencies = len(self.lo_frequencies)
        self.num_repeat_measurements = self.get_number_of_repeat_lo_frequencies()
        self.data = self.create_data_structure()

    def get_number_of_repeat_lo_frequencies(self):
        # note method assumes the same number of repeat measurements for all unique LO frequencies
        return len(np.where(self.lo_frequencies==self.lo_frequencies[0])[0])

    def make_frequency_array(self):
        ''' The absolute frequency is combination of the LO frequency (which is stepped) and the tone frequencies '''
        f_array_hz_raw = np.empty((len(self.lo_frequencies),len(self.tone_frequencies)))
        for ii in range(len(self.tone_frequencies)):
            f_array_hz_raw[:,ii] = self.lo_frequencies + self.tone_frequencies[ii]
        return f_array_hz_raw

    def create_data_structure(self):
        ''' Returns a num_pts_per_tone x num_tones x 3 data structure.
            The last index is 0: frequency, 1: i, 2:q.
            Toltec pseudo VNA sweep records multiple data points for fixed LO frequency to
            average down the noise.  This method returns a data structure that is the average of
            these multiple repeat measurements.
        '''
        f_array_hz = self.make_frequency_array()
        arr_out = np.empty((self.num_lo_frequencies//self.num_repeat_measurements,self.num_tones,3))
        for jj,arr in enumerate([f_array_hz,self.i_raw,self.q_raw]): # count, value
            for ii in range(self.num_tones):
                arr_out[:,ii,jj] = np.mean(arr[:,ii].reshape(-1,self.num_repeat_measurements), axis=1)

        # now re-order in ascenting tone frequency.
        # prior to this step, packaged as index 0-499 as above LO (from f_mid to f_high) and
        # 500-999 from f_low to f_mid.
        # below is a stupid way to do this.
        arr_out_cp = np.copy(arr_out)
        if self.reorder_data: # need to fix
            for ii in range(self.num_tones//2):
                arr_out_cp[:,ii,:] = arr_out[:,ii+self.num_tones//2,:]
            for ii in range(self.num_tones//2):
                arr_out_cp[:,ii+self.num_tones//2,:] = arr_out[:,ii,:]

        return arr_out_cp

    def plot_tone_for_index(self,ii,fig=None,ax=None):
        fig,ax = plot_summary_for_index(self.data,ii,fig=fig,ax=ax)
        return fig,ax

    def plot_all_mag(self):
        plt.figure()
        for ii in range(self.num_tones):
            y=magS21(self.data[:,ii,1],self.data[:,ii,2])
            plt.plot(self.data[:,ii,0],y)


class Pvna():
    ''' Class to construct a pseudo VNA sweep from a short LO sweep with multiple tones '''
    def __init__(self,tone_data):
        ''' data is a data structure of dimensions (num_freq_per_tone x num_tones x 3).
            The last index is (frequency,i,q) for index (0,1,2)
        '''
        self.tone_data = tone_data
        self.num_pts_per_tone,self.num_tones,foo = np.shape(self.tone_data)
        self.f,self.mags21 = self.make_pseudo_vna_sweep()

    def linear_weight(self,a,b):
        assert len(a) == len(b),'Vectors have two different lengths!'
        w = np.linspace(1,0,len(a))
        return a*w+b*(1-w)
    
    # new, 12/8/22
    def flatten(self,f): # makes array of lists of lists one-dimensional or flattens
        return [item for sublist in f for item in sublist]

    def __stitch_two(self,f0,y0,f1,y1,force_dc_alignment=False,showplot=False):
        assert f1[0]-f0[0] > 0, 'f1 must extend to higher frequencies than f0'

        lo_deltaf = f0[1]-f0[0]
        n0 = len(f0)

        # shift the ii+1 tone frequency array to align on same frequency spacing
        f_shift = (f1[0] - f0[0])%lo_deltaf
        f1p = f1 - f_shift # shifted tone1 frequency array
        y1p = np.interp(f1p,f1,y1)

        if f0[-1]<f1[0]: # case of no overlap frequencies
            f_stitch = np.append(f0,f1p)
            y_stitch = np.append(y0,y1p)
        else: # case with an overlap in frequencies between tone0 and tone1
            dex = np.where(f0-f1p[0]==0)[0][0] # index of tone 0 1st overlap frequency
            if force_dc_alignment:
                # because offset value is arbitrary, and can be different between neighboring tones,
                # force to same value at 1st overlap frequency
                y_offset = y0[dex] - y1p[0]
                y1p = y1p+y_offset

            # stitch three parts with linear weighting in overlap region
            a = y0[0:dex]
            b = self.linear_weight(y0[dex:],y1p[0:n0-dex])
            c = y1p[n0-dex:]

            y_stitch = np.hstack((a,b,c))
            f_stitch = np.append(f0,f1p[n0-dex:])

        if showplot:
            plt.figure()
            plt.plot(f0,y0,'b.')
            plt.plot(f0[dex],y0[dex],'bo')
            plt.plot(f1[0],y1[0],'ro')
            plt.plot(f1,y1,'r.')
            plt.plot(f_stitch,y_stitch,'k.')
        return f_stitch, y_stitch

    def stitch_two_complex_w_phase(self,f0,y0,f1,y1,normalize=True,showplot=False): # y0 and y1 have I + jQ data
        assert f1[0]-f0[0] > 0, 'f1 must extend to higher frequencies than f0'
        lo_deltaf = f0[1]-f0[0]
        n0 = len(f0)
    
        # shift the ii+1 tone frequency array to align on same frequency spacing as the ii tone
        f_shift = (f1[0] - f0[0])%lo_deltaf
        f1_shifted = f1 - f_shift
    
        # now interpolate complex S21 data of tone 1
        y1_interp = np.interp(f1_shifted,f1,y1)
        # alternative method
        y1_interp_fit_2 = interpolate.interp1d(f1,y1,fill_value='extrapolate')
        y1_interp_2 = y1_interp_fit_2(f1_shifted)
    
        # normalize data using them maximum mag S21 value
        if normalize:
            norm_factor_y0 = np.max(magS21(y0.real,y0.imag))
            norm_factor_y1 = np.max(magS21(y1_interp.real,y1_interp.imag))
        else:
            norm_factor_y0 = 1.
            norm_factor_y1 = 1.

        # complex data normalized
        y0_norm = y0/norm_factor_y0
        y1_norm = y1_interp/norm_factor_y1
    
        # determine overlap index
        dex = np.where(f0-f1_shifted[0]==0)[0][0] # index of tone 0 1st overlap frequency
        #print(np.shape(f0)) # shape is updated if iterating through multiple tones
        #print(f0[0]) # same for each iteration
        #print(f0[-1]) # different for each iteration
    
        # calculate theta for tone 0 and tone 1, just overlap
        theta_y0_overlap = phase2(y0_norm[dex:]) #phaseS21(y0_norm[dex:].real,y0_norm[dex:].imag)
        theta_y1_overlap = phase2(y1_norm[0:n0-dex]) #phaseS21(y1_norm[0:n0-dex].real,y1_norm[0:n0-dex].imag)

        # make duplicate array of theta for each tone, just overlap
        theta_y0_overlap_cp = np.copy(theta_y0_overlap)
        theta_y1_overlap_cp = np.copy(theta_y1_overlap)

        # unwrap phase with numpy, just overlap # already unwrap in phase2 function
        #theta_y0_overlap_cp = np.unwrap(theta_y0_overlap_cp)
        #theta_y1_overlap_cp = np.unwrap(theta_y1_overlap_cp)
    
        # make duplicate array of the complex data for each tone, just overlap
        y0_norm_overlap_cp = np.copy(y0_norm[dex:])
        y1_norm_overlap_cp = np.copy(y1_norm[0:n0-dex])
    
        # for overlap
        for i in range(len(y1_norm_overlap_cp)):
            theta_diff = theta_y1_overlap_cp[i]-theta_y0_overlap_cp[i] # with respect to tone0
            y1_norm_overlap_cp[i] = y1_norm_overlap_cp[i]*np.exp(-1.j*theta_diff)
    
        # stitch three parts with linear weighting in overlap region
        a = y0_norm[0:dex] # y0 # tone 0 before overlap
        # overlap between tone 0 and tone 1 with linear weighting
        b = self.linear_weight(y0_norm[dex:],y1_norm_overlap_cp) # y1p # y1_norm[0:n0-dex] 
        c = y1_norm[n0-dex:] # tone 1 after overlap
    
        # make a copy of different regions
        a_cp = np.copy(a)
        b_cp = np.copy(b)
        c_cp = np.copy(c)
    
        # calculate theta for a, b, and c
        theta_a = phase2(a) #phaseS21(a.real,a.imag)
        theta_b = phase2(b) #phaseS21(b.real,b.imag)
        theta_c = phase2(c) #phaseS21(c.real,c.imag)
    
        # make duplicate array of each theta
        theta_a_cp = np.copy(theta_a)
        theta_b_cp = np.copy(theta_b)
        theta_c_cp = np.copy(theta_c)
    
        # unwrap phase with numpy # already unwrap in phase2 function
        #theta_a_cp = np.unwrap(theta_a_cp)
        #theta_b_cp = np.unwrap(theta_b_cp)
        #theta_c_cp = np.unwrap(theta_c_cp)
    
        # find difference between phase level of b and a and b and c
        theta_a_level = theta_b_cp[0] - theta_a_cp[-1]
        theta_c_level = theta_b_cp[-1] - theta_c_cp[0]
    
        # for c level correction
        for i in range(len(c_cp)):
            c_cp[i] = c_cp[i]*np.exp(1.j*theta_c_level) # very effective

        # for a level correction
        for i in range(len(a_cp)):
            a_cp[i] = a_cp[i]*np.exp(1.j*theta_a_level) # similar to above
    
        y1_b_c = np.concatenate((b_cp,c_cp)) # for just b and c
    
        y_stitch = np.hstack((a_cp,b_cp,c_cp)) # I + jQ # a,b,c
        f_stitch = np.append(f0,f1_shifted[n0-dex:])
    
        if showplot:
            plt.figure()
            plt.plot(f0,magS21(y0_norm.real,y0_norm.imag),'bo')
            plt.plot(f1,magS21(y1_norm.real,y1_norm.imag),'ro')
            plt.plot(f_stitch,magS21(y_stitch.real,y_stitch.imag),'k.')

        return f_stitch, y_stitch

    def make_pseudo_vna_sweep(self):
        fs=self.tone_data[:,0,0]
        ys=magS21(self.tone_data[:,0,1],self.tone_data[:,0,2])
        for ii in range(self.num_tones-2):
            f1=self.tone_data[:,ii+1,0]
            y1=magS21(self.tone_data[:,ii+1,1],self.tone_data[:,ii+1,2])
            fs,ys = self.__stitch_two(fs,ys,f1,y1,showplot=False)
        return fs,ys
    
    def make_pseudo_vna_sweep2(self): # need to do this in two halves?
        fs=self.tone_data[:,0,0]
        ys_i=self.tone_data[:,0,1]
        ys_q = self.tone_data[:,0,2]
        ys = ys_i + 1.j*ys_q
        for ii in range(self.num_tones-2):
            f1=self.tone_data[:,ii+1,0]
            y1=self.tone_data[:,ii+1,1] + 1.j*self.tone_data[:,ii+1,2] # I + jQ
            fs,ys = self.stitch_two_complex_w_phase(fs,ys,f1,y1,showplot=False)
        return fs,ys

    def plot(self):
        plt.figure()
        plt.plot(self.f*1e-6,self.mags21,'k-',linewidth=1,label='stitched')
        #plt.plot(self.f2*1e-6,mags21(self.y2.real,self.y2.imag),'k-',linewidth=1,label='stitched 2')
        for ii in range(self.num_tones):
            y=magS21(self.tone_data[:,ii,1],self.tone_data[:,ii,2])
            plt.plot(self.tone_data[:,ii,0]*1e-6,y,label='_nolegend_')

        plt.xlabel('Frequency (MHz)')
        plt.ylabel('mag S21 (lin)')
        plt.legend()


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('config',help='<Required> yaml file with config settings')
        args = parser.parse_args()
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
	
        # for data
        sweep_dir_1 = config['load']['home_dir']
        save_dir = config['load']['fig_dir']
        sweep_name_1 = config['load']['sweep_name']
        split_sweep = sweep_name_1.split('_')
        network_num = int(''.join(filter(str.isdigit, split_sweep[0])))
        print(split_sweep)
        print(network_num)
        
        use_save_fig = config['save']['use_save_fig'] #False
        use_save_pdf = config['save']['use_save_pdf'] #False
        use_save_file = config['save']['use_save_file'] #False

        # cable delay, ns # from measurements at commissioning site
        tau_per_network = [-36.2,-34.4,-34.8,-34.7,-34.6,-41.3,-39.2,\
                           -32.3,-31.,-31.,-37.4,-39.3] # no network 10
        tau = tau_per_network[network_num] #-58.5
        
        # window to applying weighting to in fit
        if config['weight']['window_Qr'] == None:
            print('Using default window_Qr in fit weighting')
            window_Qr = 12000
        else:
            window_Qr = config['weight']['window_Qr']

        # if use_weight_type == None, fit weighting is turned off
        use_weight_type = config['weight']['weight_type']

        start = timer()
        
        files = glob.glob(sweep_dir_1+sweep_name_1+'_*')
        files.sort()
        #files.reverse()
        print(files)

        #other_file = sweep_dir_1 + 'toltec0_017069_000_0000_2022_12_11_06_08_41_tune.nc'
        #lo_sweep_check = ToltecLOsweep(other_file,use_drive_atten=True,use_sense_atten=True,use_tone_amps=True,reorder_data=False)
        #print(lo_sweep_check.drive_atten)
        ##print(lo_sweep_check.tone_amps[:])

        data_pow_sweep = []
        drive_atten = []
        sense_atten = []
        tone_amps_all = []
        tone_freq_lo_all = [] # new, 1/23/23
        for ii in range(len(files)): 
                print('Working on file: ' + files[ii])
                lo_sweep_ii = ToltecLOsweep(files[ii],use_drive_atten=True,use_sense_atten=True,use_tone_amps=True,save_tone_freq_lo=True,reorder_data=False)
                data_pow_sweep.append(lo_sweep_ii.data)
                drive_atten.append(lo_sweep_ii.drive_atten)
                sense_atten.append(lo_sweep_ii.sense_atten)
                tone_amps_all.append(lo_sweep_ii.tone_amps)
                tone_freq_lo_all.append(lo_sweep_ii.tone_freq_lo)
        drive_atten = np.asarray(drive_atten)
        sense_atten = np.asarray(sense_atten)
        tone_amps_all = np.asarray(tone_amps_all)
        tone_freq_lo_all = np.asarray(tone_freq_lo_all)
        print(drive_atten)
        print(sense_atten)
        print(np.shape(data_pow_sweep))
        print(np.shape(tone_amps_all))
        print(np.shape(tone_amps_all)[1])
        
        powlist_full = np.asarray(-1.*drive_atten)
        powlist_start = config['fit_settings']['powlist_start']
        powlist_end = config['fit_settings']['powlist_end']
        
        print('powlist range:', powlist_start, powlist_end)
        if isinstance(powlist_end, (int or float)):
            if powlist_end > 0:
                print('powlist_end needs to be negative, multiplying by -1')
                powlist_end = int(-1*powlist_end)
        
        if (powlist_start == None) and (powlist_end == None):
            powlist = np.asarray(-1.*drive_atten)
        elif (powlist_start != None) and (powlist_end != None):
            powlist = np.asarray(-1.*drive_atten[int(powlist_start):int(powlist_end)])
        elif (powlist_start != None):
            powlist = np.asarray(-1.*drive_atten[int(powlist_start):])
        elif (powlist_end != None):
            powlist = np.asarray(-1.*drive_atten[:int(powlist_end)])
        else:
            print('Invalid powlist range given')
        
        #use_window_width = tone_freq_lo_all[0]/window_Qr

        intersect_powlist,indices_powlist,indices_powlist_full = np.intersect1d(powlist,powlist_full,return_indices=True)
        print(indices_powlist_full)

        num_tones = np.shape(tone_amps_all)[1]

        all_fits = {}
        all_fits_2 = {}
        all_flags = {}
        all_bif = {}

        # desired nonlinearity parameter
        if config['flag_settings']['a_predict_guess'] == None:
            print('Using default a_predict_guess')
            use_a_predict_guess = 0.2
        else:
            use_a_predict_guess = config['flag_settings']['a_predict_guess']
            
        # percent that a_predict_guess is off from data, lower will be more stringent
        if config['flag_settings']['a_predict_threshold'] == None:
            print('Using default a_predict_threshold')
            use_a_predict_threshold = 0.2 
        else:
            use_a_predict_threshold = config['flag_settings']['a_predict_threshold']
            
        # percent that the phase fit is off from data
        if config['flag_settings']['pherr_threshold'] == None:
            print('Using default pherr_threshold')
            use_pherr_threshold = 0.2
        else:
            use_pherr_threshold = config['flag_settings']['pherr_threshold']
            
        # number of points allowed above pherr_threshold
        if config['flag_settings']['pherr_threshold_num'] == None:
            print('Using default pherr_threshold_num')
            use_pherr_threshold_num = 10 # doesn't seem to change things much
        else:
            use_pherr_threshold_num = int(config['flag_settings']['pherr_threshold_num'])
            
        # number of linewidths to fit
        if config['fit_settings']['numspan'] == None:
            print('Using default numspan')
            use_numspan = 2
        else:
            use_numspan = int(config['fit_settings']['numspan'])

        
        print(np.shape(data_pow_sweep))
        if config['fit_settings']['tone_range'] == 'all':
            tone_range = range(0,num_tones)
        elif isinstance(config['fit_settings']['tone_range'], list):
            tone_range = config['fit_settings']['tone_range']
        elif isinstance(config['fit_settings']['tone_range'], (int,float)):
            tone_range = [int(config['fit_settings']['tone_range'])]
        else:
            print('Invalid tone_range specified in config')

        a_predict_flag_all = []
        Pro_guess_dBm_list = []

        f0_corr_all_from_mean = []
        f0_corr_all_from_median = []
        flag_list_all = []
        
        # need to think of better fix than setting filedir to 0
        for ii in tone_range:
                print('Fitting res' + str(ii))
                
                if use_weight_type == None:
                    all_fits[ii] = ResonanceFitterSingleTonePowSweep(0,data_pow_sweep,ii,tone_freq_lo_all[0][ii],powlist_full,powlist,\
                                                                     tau,numspan=use_numspan,window_width=0,\
                                                                     a_predict_guess=use_a_predict_guess,a_predict_threshold=use_a_predict_threshold,\
                                                                     pherr_threshold=use_pherr_threshold,pherr_threshold_num=use_pherr_threshold_num,\
                                                                     save_fig=use_save_fig,save_pdf=use_save_pdf,filename=save_dir+sweep_name_1)
                else:
                    use_window_width = tone_freq_lo_all[0]/window_Qr
                    all_fits[ii] = ResonanceFitterSingleTonePowSweep(0,data_pow_sweep,ii,tone_freq_lo_all[0][ii],powlist_full,powlist,\
                                                                     tau,numspan=use_numspan,window_width=use_window_width[ii],\
                                                                     a_predict_guess=use_a_predict_guess,a_predict_threshold=use_a_predict_threshold,\
                                                                     pherr_threshold=use_pherr_threshold,pherr_threshold_num=use_pherr_threshold_num,\
                                                                     save_fig=use_save_fig,save_pdf=use_save_pdf,filename=save_dir+sweep_name_1,\
                                                                     weight_type=use_weight_type)
                results_all = {}
                for jj in powlist:
                    try:
                        results_all[jj] = all_fits[ii].result[jj].result
                    except:
                        pass
                all_fits_2[ii] = results_all
                
                #all_fits[ii].plot()
                #all_fits[ii].plot_fit_results()
                #all_fits[ii].plot_iq_rect([-15,-11])
                
                # avoid using negative bif values
                print(all_fits[ii].a_predict,all_fits[ii].a_predict_pow2,all_fits[ii].Pro_test_low_dBm)
                print(all_fits[ii].a_predict_off)
                a_predict_flag_all.append(all_fits[ii].a_predict_flag)
                Pro_guess_dBm_list.append(all_fits[ii].Pro_test_low_guess_dBm)
                flag_list_all.append(all_fits[ii].flag_list)
                #fit_flag_list # what is the difference between flag_list and fit_flag_list (looks like flag_list is used more)
        
        if use_save_fig:
            plt.ioff()
            for ii in tone_range:
                print('saving figures for res' + str(ii))
                ctx = all_fits[ii].plot()
                plt.close(ctx)
			
                ctx2 = all_fits[ii].plot_fit_results()
                plt.close(ctx2)

        if use_save_pdf:
            plt.ioff()
            with PdfPages('fits_' + sweep_name_1 + '_' + config['save']['save_name'] + '.pdf') as pdf:
                for ii in tone_range:
                    print('saving figures in pdf for res' + str(ii))
                    ctx = all_fits[ii].plot()
                    pdf.savefig(ctx,dpi=100)
                    plt.close(ctx)
                
                    ctx2 = all_fits[ii].plot_fit_results()
                    pdf.savefig(ctx2,dpi=100)
                    plt.close(ctx2)

        print(a_predict_flag_all)
        a_predict_flag_good_indices = np.where(np.asarray(a_predict_flag_all) == 0)
        Pro_guess_dBm_good = np.asarray(Pro_guess_dBm_list)[a_predict_flag_good_indices] 
        Pro_guess_dBm_good_original = Pro_guess_dBm_good
        Pro_guess_dBm_good = Pro_guess_dBm_good[~np.isnan(Pro_guess_dBm_good)]
        Pro_guess_dBm_good_nonan = Pro_guess_dBm_good
        print('before removing a few outliers', len(Pro_guess_dBm_good))
        Pro_guess_dBm_good = Pro_guess_dBm_good[Pro_guess_dBm_good < -5.] # 0, -5 is temporary until I'm able to improve the fits
        Pro_guess_dBm_good_less_outliers = Pro_guess_dBm_good
        print('after removing a few outliers', len(Pro_guess_dBm_good))
        print('# of good resonator fits: ' + str(len(a_predict_flag_good_indices[0])) + ' out of ' + str(num_tones))
        print(a_predict_flag_good_indices)
        print(Pro_guess_dBm_good)
        print(np.median(Pro_guess_dBm_good))
        print(np.mean(Pro_guess_dBm_good))
        print(np.std(Pro_guess_dBm_good))
        drive_atten_best_index_mean = find_nearest_indice(powlist,np.mean(Pro_guess_dBm_good))
        drive_atten_best_index_median = find_nearest_indice(powlist,np.median(Pro_guess_dBm_good))
        
        for ii in tone_range:
			#print('res ' + str(ii))
            if len(all_fits[ii].result) == 0:
                f0_corr_all_from_mean.append(0)
                f0_corr_all_from_median.append(0)
            else:
                f0_corr_all_from_mean.append(all_fits[ii].result[powlist[drive_atten_best_index_mean]].result['f0_corr'])
                f0_corr_all_from_median.append(all_fits[ii].result[powlist[drive_atten_best_index_median]].result['f0_corr'])
        indices_mean_sorted = np.argsort(f0_corr_all_from_mean)
        indices_median_sorted = np.argsort(f0_corr_all_from_median)
        
        Pro_guess_dBm_list_pos = np.asarray(Pro_guess_dBm_list)
        Pro_guess_dBm_list_neg_indices = np.where(Pro_guess_dBm_list_pos < 0)
        Pro_guess_dBm_list_pos[Pro_guess_dBm_list_neg_indices] = -1.*Pro_guess_dBm_list_pos[Pro_guess_dBm_list_neg_indices]
        
        csv_columns = ['tone_num','drive_atten','drive_atten_flag','fit_flags'] # 0 is good, 1 is bag
        rows = zip(tone_range, Pro_guess_dBm_list_pos, a_predict_flag_all, flag_list_all)
        with open('drive_atten_' + sweep_name_1 + '_' + config['save']['save_name'] + '.csv', 'w') as f: #csvfile:
            writer = csv.writer(f)
            writer.writerow(csv_columns)
            for row in rows:
                writer.writerow(row)
        
        a_save = {'tau': tau, 'files': files, 'drive_atten': drive_atten, 'sense_atten': sense_atten, 'tone_amps': tone_amps_all, 'data': data_pow_sweep,\
                   'num_tones':num_tones, 'powlist': powlist, 'powlist_full': powlist_full, 'numspan': use_numspan, 'a_predict_guess': use_a_predict_guess,\
                   'a_predict_threshold': use_a_predict_threshold, 'a_predict_flag': a_predict_flag_all, 'Pro_guess_dBm': Pro_guess_dBm_list,\
                   'a_predict_good': a_predict_flag_good_indices, 'Pro_guess_dBm_good': Pro_guess_dBm_good_original, 'Pro_guess_dBm_good_nonan': Pro_guess_dBm_good_nonan,\
                   'Pro_guess_dBm_good_less_outliers': Pro_guess_dBm_good_less_outliers, 'Pro_guess_dBm_good_final': Pro_guess_dBm_good, 'f0_mean': f0_corr_all_from_mean,\
                   'f0_median': f0_corr_all_from_median, 'f0_mean_sorted': np.sort(f0_corr_all_from_mean),'f0_median_sorted': np.sort(f0_corr_all_from_median),\
                   'indices_mean_sorted': indices_mean_sorted, 'indices_median_sorted': indices_median_sorted, 'Pro_guess_dBm_mean': np.mean(Pro_guess_dBm_good),\
                   'Pro_guess_dBm_median': np.median(Pro_guess_dBm_good),'Pro_guess_dBm_std': np.std(Pro_guess_dBm_good),'drive_atten_mean': drive_atten_best_index_mean,\
                   'drive_atten_median': drive_atten_best_index_median, 'fits': all_fits_2
                   }
        # is all_fits the problem? # # 'fits': all_fits, (yes, saving data in a weird format) # just saving result from class which contains fit and some other info, other variables?
        if use_save_file:
            with open('fits_' + sweep_name_1 + '_' + config['save']['save_name'] + '.pkl', 'wb') as handle:
                pickle.dump(a_save, handle)
        elapsed_time = timer() - start
        print('elapsed time',elapsed_time)
        print('num_tones', num_tones)
# d['fits'][55].result[-17].result['f0_corr']
# example: all_fits[1].result[-17].result['flag']
