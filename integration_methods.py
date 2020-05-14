import Rate_calculation
#import constants as ct

from mpmath import mp
from mpmath import fp
import numpy as np
import scipy.integrate as spint
import time

methods=["mp-gl", "mp-ts", "sp-quad", "sp-gauss", "monte-carlo", "w-cumsum", "sp-simps", "romberg"];

#cumtrapz relative error tolerance
err_rel=1e-1;
def_nodes=1e2; #default number of nodes
maxloop=100;

# general integration method
# syntax is mpmath like, set the limits you dont want to integrate over to 0
#
def integrate(f, limx, limy, limz, method):
    if method=="mp-gl":
        if limz!=0:
            return mp.quad(f, limx, limy, limz, method="gauss-legendre");
        else:
            if limy!=0:
                return mp.quad(f, limx, limy, method="gauss-legendre");
            else:
                return mp.quad(f, limx, method="gauss-legendre");
    elif method=="mp-ts":
        if limz!=0:
            return mp.quad(f, limx, limy, limz, method="tanh-sinh");
        else:
            if limy!=0:
                return mp.quad(f, limx, limy, method="tanh-sinh");
            else:
                return mp.quad(f, limx, method="tanh-sinh");
    elif method=="fp-gl":
        if limz!=0:
            return fp.quad(f, limx, limy, limz, method="gauss-legendre");
        else:
            if limy!=0:
                return fp.quad(f, limx, limy, method="gauss-legendre");
            else:
                return fp.quad(f, limx, method="gauss-legendre");
    elif method=="fp-ts":
        if limz!=0:
            return fp.quad(f, limx, limy, limz, method="tanh-sinh");
        else:
            if limy!=0:
                return fp.quad(f, limx, limy, method="tanh-sinh");
            else:
                return fp.quad(f, limx, method="tanh-sinh");
    elif method=="sp-quad":
        if limz!=0:
            return spint.tplquad(f, limz[0], limz[1], limy[0], limy[1], limx[0], limx[1])[0];
        else:
            if limy!=0:
                return spint.dblquad(f, limy[0], limy[1], limx[0], limx[1])[0];
            else:
                return spint.quad(f, limx[0], limx[1])[0];

    elif method=="romberg":
        if not np.ndim(limx)==0:
            limx=[float(limx[0]), float(limx[1])];
        if not np.ndim(limy)==0:
            limy=[float(limy[0]), float(limy[1])];
        if not np.ndim(limz)==0:
            limz=[float(limz[0]), float(limz[1])];

        reltol=1e-16;
        abstol=1e-16;

        if limz!=0:
            return spint.romberg(lambda z: spint.romberg(lambda y: spint.romberg(lambda x: float(f(x,y,z)), limx[0], limx[1], tol=abstol, rtol=reltol), limy[0], limy[1], tol=abstol, rtol=reltol), limz[0], limz[1], tol=abstol, rtol=reltol);
        else:
            if limy!=0:
                return spint.romberg(lambda y: spint.romberg(lambda x: float(f(x,y)), limx[0], limy[1], tol=abstol, rtol=reltol), limy[0], limy[1], tol=abstol, rtol=reltol);
            else:
                return spint.romberg(lambda x: float(f(x)), limx[0], limx[1], tol=abstol, rtol=reltol);

    #currently broken, but slow so unused
    elif method=="sp-gauss":
        if not np.ndim(limx)==0:
            limx=[float(limx[0]), float(limx[1])];
        if not np.ndim(limy)==0:
            limy=[float(limy[0]), float(limy[1])];
        if not np.ndim(limz)==0:
            limz=[float(limz[0]), float(limz[1])];

        order=7;

        if limz!=0:
            return spint.fixed_quad(lambda z: spint.fixed_quad(lambda y: spint.fixed_quad(lambda x: f(x,y,z), limx[0], limx[1], n=order)[0], limy[0], limy[1], n=order)[0], limz[0], limz[1], n=order)[0];
        else:
            if limy!=0:
                return spint.fixed_quad(lambda y: spint.romberg(lambda x: f(x,y), limx[0], limy[1], n=order)[0], limy[0], limy[1], n=order)[0];
            else:
                return spint.fixed_quad(lambda x: f(x), limx[0], limx[1], n=order)[0];

    elif method=="w-cumsum":
        if not np.ndim(limx)==0:
            limx=[float(limx[0]), float(limx[1])];
        if not np.ndim(limy)==0:
            limy=[float(limy[0]), float(limy[1])];
        if not np.ndim(limz)==0:
            limz=[float(limz[0]), float(limz[1])];

        if limz!=0:
            dx=(limx[1]-limx[0])/def_nodes;
            dy=(limy[1]-limy[0])/def_nodes;
            dz=(limz[1]-limz[0])/def_nodes;

            loop=0;
            lastres=0;
            while True:
                xl=np.arange(limx[0], limx[1], dx);
                yl=np.arange(limy[0], limy[1], dy);
                zl=np.arange(limz[0], limz[1], dz);

                X, Y, Z=np.meshgrid(xl, yl, zl);

                fx=[];
                for i in range(0, len(X)):
                    fy=[];
                    for j in range(0, len(Y)):
                        fz=[];
                        for k in range(0, len(Z)):
                            fz.append(f(X[i][j][k], Y[i][j][k], zl[k]));
                        fy.append(spint.simps(fz, dx=dz));
                    fx.append(spint.simps(fy, dx=dy));
                res=spint.simps(fx, dx=dx);
                if loop!=0:
                    if np.abs(res-lastres)/res < err_rel:
                        return res;
                    else:
                        ad=(1/2)**loop; #linear to begin with
                        dx=dx*ad;
                        dy=dy*ad;
                        dz=dz*ad;
                        lastres=res;
                if loop > maxloop:
                    break;
                loop+=1;
            else:
                if limy!=0:
                    dx=def_dx;
                    dy=def_dx;

                    loop=0;
                    lastres=0;
                    while True:
                        xl=np.arange(limx[0], limx[1], dx);
                        yl=np.arange(limy[0], limy[1], dy);
                    
                        X, Y=np.meshgrid(xl, yl);

                        fx=[];
                        for i in range(0, len(X)):
                            fy=[];
                            for j in range(0, len(Y)):
                                fy.append(f(X[i][j], yl[j]));
                            fx.append(spint.simps(fy, dx=dy));
                        res=spint.simps(fx, dx=dx);
                        if loop!=0:
                            if np.abs(res-lastres)/res < err_rel:
                                return res;
                            else:
                                ad=(1/2)**loop; #linear to begin with
                                dx=dx*ad;
                                dy=dy*ad;
                                lastres=res;
                        if loop > maxloop:
                            break;
                        loop+=1;
                else:
                    dx=def_dx;

                    loop=0;
                    lastres=0;
                    while True:
                        xl=np.arange(limx[0], limx[1], dx);

                        fx=[];
                        for i in range(0, len(X)):
                            fx.append(f(xl[i]));
                        
                        res=spint.simps(fx, dx=dx);
                        if loop!=0:
                            if np.abs(res-lastres)/res < err_rel:
                                return res;
                        else:
                            ad=(1/2)**loop; #linear to begin with
                            dx=dx*ad;
                            lastres=res;
                        if loop > maxloop:
                            break;
                        loop+=1;
                        
    #still a bit broken but proved slower than mp-gl
    elif method=="monte-carlo":
        N=int(1e6);
        
        if limz!=0:
            N=int(round(N**(1/3)));
            x=np.random.rand(N)*(limx[1]-limx[0])+limx[0];
            y=np.random.rand(N)*(limy[1]-limy[0])+limy[0];
            z=np.random.rand(N)*(limz[1]-limy[0])+limz[0];

            X,Y,Z=np.meshgrid(x,y,z);

            fxyz=[];
            for i in range(0, len(X)):
                fxy=[];
                for j in range(0, len(Y)):
                    fx=[];
                    for k in range(0, len(Z)):
                        fx.append(f(X[i][j][k], Y[i][j][k], Z[i][j][k]));
                    fxy.append(fx);
                fxyz.append(fxy);
            
            wmax=np.max(fxyz);
            wmin=np.min(fxyz);

            W=np.random.rand(N, N, N)*(wmax-wmin)+wmin;

            est=0;
            for i in range(0, len(fxyz)):
                for j in range(0, len(fxyz[i])):
                    for k in range(0, len(fxyz[i][j])):
                        if W[i][j][k] > 0 and W[i][j][k] < fxyz[i][j][k]:
                            est=est+fxyz[i][j][k];
                        elif W[i][j][k] < 0 and W[i][j][k] > fxyz[i][j][k]:
                            est=est+fxyz[i][j][k];

                return (est/(N**3))*(limx[1]-limx[0])*(limy[1]-limy[0])*(limz[1]-limz[0])*(wmax-wmin); 
        else:
            if limy!=0:
                N=int(round(N**(1/2)));
                x=np.random.rand(N)*(limx[1]-limx[0])+limx[0];
                y=np.random.rand(N)*(limy[1]-limy[0])+limy[0];
                
                X,Y=np.meshgrid(x,y);

                fxy=[];
                for i in range(0, len(X)):
                    fx=[];
                    for j in range(0, len(Y)):
                        fx.append(f(X[i][j], Y[i][j]));
                    fxy.append(fx);

                zmax=np.max(fxy);
                zmin=np.min(fxy);

                Z=np.random.rand(N, N)*(zmax-zmin)+zmin;

                est=0;
                for i in range(0, len(fxy)):
                    for j in range(0, len(fxy[i])):
                        if Z[i][j] > 0 and Z[i][j] < fxy[i][j]:
                            est=est+fxy[i][j];
                        elif Z[i][j] < 0 and Z[i][j] > fxy[i][j]:
                            est=est+fxy[i][j];

                return (est/(N**2))*(limx[1]-limx[0])*(limy[1]-limy[0])*(zmax-zmin);
            else:
                X=np.random.rand(N)*(limx[1]-limx[0])+limx[0];

                fx=[];
                for i in range(0, len(X)):
                    fx.append(f(X[i]));

                ymax=np.max(fx);
                ymin=np.min(fx);

                Y=np.random.rand(N)*(ymax-ymin)+ymin;

                est=0;
                for i in range(0, len(fx)):
                    if Y[i] > 0 and Y[i] < fx[i]:
                        est=est+fx[i];
                    elif Y[i] < 0 and Y[i] > fx[i]:
                        est=est+fx[i];

                return (est/N)*(limx[1]-limx[0])*(ymax-ymin);


    #preallocated, expected to be slow
    elif method=="sp-simps":
        if limz!=0:
            dx=(limx[1]-limx[0])/def_nodes;
            dy=(limy[1]-limy[0])/def_nodes;
            dz=(limz[1]-limz[0])/def_nodes;

            loop=0;
            lastres=0;
            while True:
                xl=np.arange(limx[0], limx[1], dx);
                yl=np.arange(limy[0], limy[1], dy);
                zl=np.arange(limz[0], limz[1], dz);

                X, Y, Z=np.meshgrid(xl, yl, zl);

                fx=[];
                for i in range(0, len(X)):
                    fy=[];
                    for j in range(0, len(Y)):
                        fz=[];
                        for k in range(0, len(Z)):
                            fz.append(f(X[i][j][k], Y[i][j][k], zl[k]));
                        fy.append(spint.simps(fz, dx=dz));
                    fx.append(spint.simps(fy, dx=dy));
                res=spint.simps(fx, dx=dx);
                if loop!=0:
                    if np.abs(res-lastres)/res < err_rel:
                        return res;
                    else:
                        ad=(1/2)**loop; #linear to begin with
                        dx=dx*ad;
                        dy=dy*ad;
                        dz=dz*ad;
                        lastres=res;
                if loop > maxloop:
                    break;
                loop+=1;
            else:
                if limy!=0:
                    dx=def_dx;
                    dy=def_dx;

                    loop=0;
                    lastres=0;
                    while True:
                        xl=np.arange(limx[0], limx[1], dx);
                        yl=np.arange(limy[0], limy[1], dy);
                    
                        X, Y=np.meshgrid(xl, yl);

                        fx=[];
                        for i in range(0, len(X)):
                            fy=[];
                            for j in range(0, len(Y)):
                                fy.append(f(X[i][j], yl[j]));
                            fx.append(spint.simps(fy, dx=dy));
                        res=spint.simps(fx, dx=dx);
                        if loop!=0:
                            if np.abs(res-lastres)/res < err_rel:
                                return res;
                            else:
                                ad=(1/2)**loop; #linear to begin with
                                dx=dx*ad;
                                dy=dy*ad;
                                lastres=res;
                        if loop > maxloop:
                            break;
                        loop+=1;
                else:
                    dx=def_dx;

                    loop=0;
                    lastres=0;
                    while True:
                        xl=np.arange(limx[0], limx[1], dx);

                        fx=[];
                        for i in range(0, len(X)):
                            fx.append(f(xl[i]));
                        
                        res=spint.simps(fx, dx=dx);
                        if loop!=0:
                            if np.abs(res-lastres)/res < err_rel:
                                return res;
                        else:
                            ad=(1/2)**loop; #linear to begin with
                            dx=dx*ad;
                            lastres=res;
                        if loop > maxloop:
                            break;
                        loop+=1;

            return res;

#benchmarking
def benchmark():
    f=lambda x,y,z: x**2*y**2-x**2*z;

    for m in methods:
        t=time.time();
        r=integrate(f, [0, 1], 0, 0, m);
        print("Method: "+m+" Time: "+str(time.time()-t)+" Result: "+str(r));

if __name__=="__main__":
    benchmark();
