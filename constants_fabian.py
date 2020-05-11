import numpy as np
import scipy.special as spc
import matplotlib.pyplot as plt
from mpmath import mp

#Note: all constants given in SI units.

#general constants
pi=np.pi;
c0=299792458;
q=1.60217622e-19;
m_e=9.10938356e-31;
me=m_e;
hbar=1.054718e-34;
eps_0=8.854187e-12;
a0=4*pi*eps_0*hbar**2/(m_e*q**2);

#graphene constants
a=0.142e-9;
c=4*np.pi/(a*(27**(1/2)));
fi=4.3*q;

#effective charges for carbon wavefunctions
Zeff2s=4.84;
Zeff2pz=4.03;
Zeff2pxy=5.49;

#graphene wavefunction norms
nrm_pz=(Zeff2pz**7)**(1/2)/mp.pi;
nrm_px=(Zeff2pxy**7)**(1/2)/mp.pi;
nrm_py=nrm_px;
nrm_s=(Zeff2s**5)**(1/2)/mp.pi;

#DM parameters
Auc=(27**(1/2)*a**2)/2;
Nc=5e25;
rho_chi=0.4e15*q/(c0**2);

v0=220e3;
vesc=550e3;
vobs=244e3;
vmax=vesc+vobs;

K=v0**3*mp.pi*(mp.pi**(1/2)*mp.erf(vesc/v0)-2*(vesc/v0)*mp.exp(-(vesc/v0)**2));

Nesc=spc.erf(vesc/v0)-2*(vesc/v0)*np.exp(-vesc**2/v0**2)/(np.pi**(1/2));

#DM model parameters
sigma_e=1e-37*1e-4;
m_chi=q*1e8/(c0**2); # DM mass, 100 MeV
mchi=m_chi;

m_chi_list=np.multiply([100, 1, 5, 10, 30, 50, 300, 500, 1000], q*1e6/(c0**2));
mchi_list=m_chi_list;

mu=mchi*me/(mchi+me);

#first BZ of graphene
def firstBZ(x):
    if x<-3**(1/2)*c/2:
        return 0;
    elif x<0:
        return c+x/(3**(1/2));
    elif x<3**(1/2)*c/2:
        return c-x/(3**(1/2));
    else:
        return 0;

def gen_mu(mchi_index):
    if mchi_index==0:
        return mu;
    else:
        return mchi_list[mchi_index-1]*me/(mchi_list[mchi_index-1]+me);



if __name__=="__main__":
    r=np.linspace(-2*c, 2*c, 100);
    R=map(lambda x: firstBZ(x), r);
    R1=list(R);
    plt.plot(r, R1);
    plt.show();
