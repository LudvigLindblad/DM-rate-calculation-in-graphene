import matplotlib.pyplot as plt
from mpmath import mp
from mpmath import fp
import scipy.integrate as spint
import numpy as np
import time
import scipy.special as spc
from multiprocessing import Pool
import pathos.pools as pp
from datetime import datetime
import sys
import scipy.interpolate as interp

import Rate_calculation
import constants as ct
import dataplotter
import integration_methods as mint



xl=np.linspace(-np.sqrt(3)*ct.c/2, np.sqrt(3)*ct.c/2, 100);
yl=[];
for x in xl:
    yl.append(ct.firstBZ(x));

def g(v):
    return 1/(ct.Nesc*(np.pi**(3/2))*ct.v0**3)*mp.exp(-(v+ct.vobs)**2/ct.v0**2);


xl=np.linspace(0, 1.5*ct.vesc+ct.vobs);
yl=[];
for x in xl:
    yl.append(g(x));

    
#fill*1BZ tabulate over lx, ly 
num=17;    #grid size
fill=1.1;  
lxl=np.linspace(-3**(1/2)*ct.c*fill/2, 3**(1/2)*ct.c*fill/2, num);
lyl=np.linspace(-ct.c*fill, ct.c*fill, num);


d2l=2*mp.sqrt(3)*(ct.c/num)**2;  #differential element used in summation

lmx, lmy=np.meshgrid(lxl, lyl);
inBZ=np.zeros(np.shape(lmx));
nevals=0;

E_pi=np.zeros(np.shape(lmx));

for i in range(0, len(lmx)):
    row=[];
    for j in range(0, len(lmy)):
        E_pi[i][j]=eq5.E_pi_minus(lmx[i][j], lmy[i][j])/ct.q;
        if abs(ct.firstBZ(lmx[i][j])) > abs(lmy[i][j]):
            inBZ[i][j]=1;
        else:
            nevals+=1;


            

method="mp-gl";

calc=eq5.calculatormk3(method);

def lcalc(i, j, band, checkBZ=True, fdm=1):
    if checkBZ:
        if inBZ[i][j]==1:
            return lambda kf, ind: d2l*(ct.rho_chi*ct.Nc*ct.Auc*2/(ct.gen_mu(ind)**2*ct.mchi_list[ind]))*calc.dRdEe(lmx[i][j], lmy[i][j], band, fdm, ind)(kf);
        else:
            return lambda kf, ind: 0;
    else:
        return lambda kf, ind: (ct.rho_chi*ct.Nc*ct.Auc*2/(ct.gen_mu(ind)**2*ct.mchi_list[ind]))*calc.dRdEe(lmx[i][j], lmy[i][j], band, fdm, ind)(kf); 



#not used during interpolation
Rfuncs_pi=[];
Rfuncs_sigma1=[];
Rfuncs_sigma2=[];
Rfuncs_sigma3=[];


#change this if you want to run differnt q:s for non-interpolation
#default value 1,  2=q,  3=q**2
fdm=1;

for i in range(0, len(lmx)):
    for j in range(0, len(lmy)):
        Rfuncs_pi.append(lcalc(i, j, "pi", True, fdm));
        Rfuncs_sigma1.append(lcalc(i, j, "sigma1", True, fdm));
        Rfuncs_sigma2.append(lcalc(i, j, "sigma2", True, fdm));
        Rfuncs_sigma3.append(lcalc(i, j, "sigma3", True, fdm));


       
 
#upper bound for pi: ~345 eV
kl=np.asarray(np.geomspace(3e-25, np.sqrt(2*ct.me*270*ct.q), 16)); #Interval, and #of k_f values

def keval(k, band, parts, index, verbose, mchi_index=0):
    Rk=0;
 
    if parts==1:
        start=0;
        stop=num**2;
    else:
        start=int((index-1)*num**2/parts);
        stop=int(index*num**2/parts);

    if band=="pi":
        for i in range(start, stop):
            if verbose:
                print("Evaluating Rfunc: "+str(i+1)+"/"+str(num**2)+" for kf "+str(k)+" in band pi", file=sys.stderr);
            Rk+=mp.re(Rfuncs_pi[i](k, mchi_index));
    elif band=="sigma1":
        for i in range(start, stop):
            if verbose:
                print("Evaluating Rfunc: "+str(i+1)+"/"+str(num**2)+" for kf "+str(k)+" in band sigma1", file=sys.stderr);
            Rk+=mp.re(Rfuncs_sigma1[i](k, mchi_index)); 
    elif band=="sigma2":
        for i in range(start, stop):
            if verbose:
                print("Evaluating Rfunc: "+str(i+1)+"/"+str(num**2)+" for kf "+str(k)+" in band sigma2", file=sys.stderr);
            Rk+=mp.re(Rfuncs_sigma2[i](k, mchi_index));
    elif band=="sigma3":
        for i in range(start, stop):
            if verbose:
                print("Evaluating Rfunc: "+str(i+1)+"/"+str(num**2)+" for kf "+str(k)+" in band sigma3", file=sys.stderr);
            Rk+=mp.re(Rfuncs_sigma3[i](k, mchi_index));
    else:
        print("band error");
        return 0;

    return Rk;




def keval_L(k, band, verbose, fdm):
    iL=[];
    
    for i in range(0, len(lmx)):
        iLr=[];
        for j in range(0, len(lmy)):
            if verbose:
                print("Evaluating Rfunc: "+str(i*len(lmx)+j+1)+"/"+str(num**2)+" for kf "+str(k)+" in band "+band, file=sys.stderr);
            #iLr.append(1);
            iLr.append(float(mp.re(lcalc(i,j, band, False, fdm)(k))));
        iL.append(iLr);

    plt.figure();
    plt.imshow(iL);
    plt.show();
    
    fL=interp.interp2d(lmx, lmy, iL, kind='linear');
    return fL;



def integrate_L(fL):
    return mp.quad(lambda x: mp.quad(lambda y: fL(float(x), float(y))[0], [-1*ct.firstBZ(x), ct.firstBZ(x)], method="gauss-legendre"), [-3**(1/2)*ct.c/2, 3**(1/2)*ct.c/2], method="gauss-legendre"); 

#expecting all 1:s
def plot_C_pi():
    C=[];
    for m in [0, 1]:
        Cm=[];
        for i in range(0, len(lmx)):
            Crow=[];
            for j in range(0, len(lmy)):
                if m==0:
                    Crow.append(1);
                if m==1:
                    Crow.append(float(mp.norm(-mp.exp(1j*(mp.atan(mp.im(eq5.fl(lmx[i][j], lmy[i][j], 0))/mp.re(eq5.fl(lmx[i][j], lmx[i][j], 0))))))));
            Cm.append(Crow);
        C.append(Cm);

    for m in range(0, len(C)):
        plt.figure();
        plt.title("C matrix for band pi, element number "+str(m));
        plt.contourf(lmx, lmy, C[m], cmap=plt.get_cmap("coolwarm"));
        plt.colorbar();
        plt.xlabel(r'$l_x$ $[m^{-1}]$');
        plt.ylabel(r'$l_y$ $[m^{-1}]$');

#check if remotely executed
if __name__=="__main__":
    if len(sys.argv)==1:
        #start parallellisation
        nnodes=16;                    #cores
        calc_band="sigma1";           #Bands: pi,sigma1,sigma2,sigma3 
        interpolation=False;
        fdm=1;         #Not used,  Fabian!!
        do_mchi_loop=False;

        p=pp.ProcessPool(nodes=nnodes);
        bands=["pi", "sigma1", "sigma2", "sigma3"];
        if do_mchi_loop==False:
            if interpolation==True:
                est=107.14*num**2*len(kl)/nnodes;          #Estimates time 
                t0=time.time();
                print("Calculation started with method: "+method+", "+str(num**2)+" total l evaluations. Estimated time: "+str(est), file=sys.stderr); 
                fLl=p.map(lambda k: keval_L(k, calc_band, True, fdm), kl);
                krates=[];

                for i in range(0, len(kl)):        
                    krates.append(integrate_L(fLl[i]));
                    
                print("Total elapsed time: "+str(time.time()-t0), file=sys.stderr);

            else:
                est=107.14*nevals**2*len(kl)/nnodes;
                t0=time.time();
                print("Calculation started with method: "+method+", "+str(nevals)+" total l evaluations. Estimated time: "+str(est), file=sys.stderr); 
                krates=p.map(lambda k: keval(k, calc_band, 1, 0, True), kl);

                print("Total elapsed time: "+str(time.time()-t0), file=sys.stderr);
       
            print(kl, file=sys.stderr)
            print(krates, file=sys.stderr)

            plotter=dataplotter.DataPlotter(kl, krates, calc_band);   
        
            plotter.printout();
            #plotter.plot();

        else:
            est=107.14*nevals**2*len(kl)*len(ct.mchi_list)/nnodes;
            t0=time.time();
            print("m_chi-sigma_e evaluation started. "+str(len(ct.mchi_list)*nevals)+" total l evaluations, with F_DM code: "+str(fdm)+" Estimated time: "+str(est), file=sys.stderr);
            
            krate=list(map(lambda i: sum(p.map(lambda k: float(keval(k, calc_band, 1, 0, True, i)), kl)), range(0, len(ct.mchi_list))));

            print("Total elapsed time: "+str(time.time()-t0), file=sys.stderr);

            print("F_DM code: "+str(fdm));
            print("Mchi_list: "+str(ct.mchi_list));
            print("Corresponding rates:"+str(krate));

    else:
        if len(sys.argv)==2:
            if sys.argv[1]=="help":
                print("Usage: integration_station [band] [nodes] [F_DM] ([parts]=1 [index]=1)");
                print("F_DM: \t 1 - 1");
                print("\t 2 - q");
                print("\t 3 - q**2");
        else:
            try:
                band=sys.argv[1];
                nnodes=int(sys.argv[2]);
                fdm=int(sys.argv[3]);
                try:
                    parts=int(sys.argv[4]);
                    index=int(sys.argv[5]);
                except:
                    parts=1;
                    index=1;
            except:
                raise Exception("Too few input arguments");

            p=pp.ProcessPool(nodes=nnodes);
            
            est=107.14*num**2*len(kl)/nnodes;
            t0=time.time();
            print("Calculation started with method: "+method+", "+str(num**2)+" total l evaluations. Estimated time: "+str(est), file=sys.stderr); 
            fLl=p.map(lambda k: keval_L(k, band, True, fdm), kl);
            krates=[];
            #print(fLl);
            for i in range(0, len(kl)):        
                krates.append(integrate_L(fLl[i]));
                #plot_L(kl[i], fL[i]);
            print("Total elapsed time: "+str(time.time()-t0), file=sys.stderr);  
            
            plotter=dataplotter.DataPlotter(kl, krates);
            plotter.datasets.name=band+" part "+str(index)+"/"+str(parts)+" for F_DM code"+str(fdm);

            print(kl);
            print(krates);

            plotter.printout();
