import matplotlib.pyplot as plt
import numpy as np
from mpmath import mpf
import sys
import csv
from datetime import datetime

import constants as ct

plt.rc('text', usetex=True);
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

#PLOTTING CONSTANTS
#pi, sigma1, sigma2, sigma3
colors=["#008b8b", "#00cf00", "#0000cf", "#ff1493"];
colors=[(0, 0x8b/0xff, 0x8b/0xff), (0, 0xcf/0xff, 0), (0, 0, 0xcf/0xff), (1, 0x14/0xff, 0x93/0xff)];

#1, q, q^2
linestyles=["solid", "dashdot", "dashed"];

scale=2;
thicken=1.75;
figure_size=[1.3*4.8, 4.8]; #inches
margin=0.15;

#overall rate normalization to kg^-1 year^-1 + (2*pi)^3 factor
factor=365.25*3600*24*((2*ct.pi)**3);

class DataPlotter:
    datasets=[];
    filename="0";

    def __init__(self, kl=0, Rl=0, filename="0", mode="k"):
        if mode=="k":
            if filename=="0":
                if not np.ndim(kl)==0 and not np.ndim(Rl)==0:
                    ds=[];
                    if not np.ndim(kl)==1:
                        for i in range(0, len(kl)):
                            ds.append(DataSet("noname", kl[i] , Rl[i]));
                    else:
                        ds=DataSet("noname", kl, Rl);
                    self.datasets=ds;
            else:
                self.filename=filename;
                if np.ndim(kl)==0 and np.ndim(Rl)==0:
                    self.datasets=self.load(filename);
                else:
                    self.filename=filename;

                    ds=[];
                    if not np.ndim(kl)==1:
                        for i in range(0, len(kl)):
                            ds.append(DataSet("noname", kl[i] , Rl[i]));
                    else:
                        ds=DataSet("noname", kl, Rl);
                    self.datasets=ds;
        elif mode=="m":
            self.filename=filename;
            self.datasets=self.load_m(filename); 
 

    def plot(self, plot_sum=False, dist_fdm=False, only_sum=False):
        fig=plt.gcf();
        fig.set_size_inches(figure_size[0], figure_size[1]);
        fig.add_axes([margin, margin, 1-2*margin, 1-2*margin]);

        if hasattr(self.datasets, "__len__"):
            if not only_sum==True:
                for i in range(0, len(self.datasets)):
                    El=[];
                    for j in range(0, len(self.datasets[i].kl)):
                        El.append(self.datasets[i].kl[j]**2/(2*ct.me)/ct.q);

                    Rl=np.multiply(self.datasets[i].Rl, factor);

                    lbl, clr, style=self.build(self.datasets[i].name);
                    
                    plt.plot(El, Rl, label="\Large{"+lbl+"}", color=clr, linestyle=style, linewidth=scale);

            if plot_sum==True:
                if only_sum==True:
                    really_thicken=1;
                else:
                    really_thicken=thicken;
                
                if dist_fdm==False:
                    Rlsum=np.zeros(16);

                    El=[];
                    for j in range(0, len(self.datasets[i].kl)):
                        El.append(self.datasets[i].kl[j]**2/(2*ct.me)/ct.q);

                    for d in self.datasets:
                        Rlsum=np.add(Rlsum, d.Rl);
                    
                    plt.plot(El, np.multiply(Rlsum, factor), label=r"\Large{$\sum_{\text{bands}}$}", color="#000000", linestyle="solid", linewidth=scale*really_thicken);
                    
                else:
                    Rlsum=[np.zeros(16), np.zeros(16), np.zeros(16)];
                    for d in self.datasets:
                        if "F_DM=1" in d.name:
                            Rlsum[0]=np.add(Rlsum[0], d.Rl);
                        elif "F_DM=q" in d.name:
                            Rlsum[1]=np.add(Rlsum[1], d.Rl);
                        elif "F_DM=2q" in d.name:
                            Rlsum[2]=np.add(Rlsum[2], d.Rl);
                    El=[];
                    for j in range(0, len(self.datasets[0].kl)):
                        El.append(self.datasets[0].kl[j]**2/(2*ct.me)/ct.q);
 
                    plt.plot(El, np.multiply(Rlsum[0], factor), label=r"\Large{$\sum_{\text{bands}}$ $\left|\mathcal{M}\right|^2 \propto 1$}", color="#ff4500", linestyle=linestyles[0], linewidth=scale*really_thicken);
                    plt.plot(El, np.multiply(Rlsum[1], factor), label=r"\Large{$\sum_{\text{bands}}$ $\left|\mathcal{M}\right|^2 \propto {|\mathbf{q}|}^2$}", color="#00cfcf", linestyle=linestyles[1], linewidth=scale*really_thicken);
                    plt.plot(El, np.multiply(Rlsum[2], factor), label=r"\Large{$\sum_{\text{bands}}$ $\left|\mathcal{M}\right|^2 \propto {|\mathbf{q}|}^4$}", color="#cf00cf", linestyle=linestyles[2], linewidth=scale*really_thicken);

        else:
            El=[];
            for j in range(0, len(self.datasets.kl)):
                El.append(self.datasets.kl[j]**2/(2*ct.me)/ct.q);

            lbl, style, clr=self.build(self.datasets.name);

            Rl=np.multiply(self.datasets.Rl, factor);

            plt.plot(El, Rl, label=lbl, color=clr, linewidth=scale);

        plt.xscale('log');
        plt.xlim(El[0], El[len(El)-1]);
        plt.xlabel(r"\huge{$E_{er}$ [eV]}");
        plt.yscale('log');
        plt.ylabel(r"\huge{$dR/d\ln{E_{er}}$ [$\text{kg}^{-1}\cdot\text{year}^{-1}$]}"); 
        plt.legend(loc=3);

        fig=plt.gcf();
        fig.set_size_inches(figure_size[0], figure_size[1]);
        
        ax=plt.gca();
        ax.tick_params(axis='both', which='major', labelsize=16);

        plt.show();
    
    def plot_m(self):
        fig=plt.gcf();
        fig.set_size_inches(figure_size[0], figure_size[1]);
        fig.add_axes([margin, margin, 1-2*margin, 1-2*margin]);

        ml=list(map(lambda m: m*9e16/1.6e-13, self.datasets[0].kl));

        if hasattr(self.datasets, "__len__"):
            sigma_e_l=[np.zeros(8), np.zeros(8), np.zeros(8)];
            for d in self.datasets:
                if "F_DM=1" in d.name:
                    sigma_e_l[0]=self.build_sigma(d);
                elif "F_DM=q" in d.name:
                    sigma_e_l[1]=self.build_sigma(d);
                elif "F_DM=2q" in d.name:
                    sigma_e_l[2]=self.build_sigma(d);

            plt.plot(ml, sigma_e_l[0], label=r"\Large{$\pi$ $\left|\mathcal{M}\right|^2 \propto 1$}", color="#ff4500", linestyle=linestyles[0], linewidth=scale);
            plt.plot(ml, sigma_e_l[1], label=r"\Large{$\pi$ $\left|\mathcal{M}\right|^2 \propto {|\mathbf{q}|}^2$}", color="#00cfcf", linestyle=linestyles[1], linewidth=scale);
            plt.plot(ml, sigma_e_l[2], label=r"\Large{$\pi$ $\left|\mathcal{M}\right|^2 \propto {|\mathbf{q}|}^4$}", color="#cf00cf", linestyle=linestyles[2], linewidth=scale);

        else:
            sigma_e_l=self.build_sigma(self.datasets);
            lbl, style, clr=self.build(self.datasets.name);

            plt.plot(ml, sigma_e_l, label=lbl, color=clr, linewidth=scale);

        plt.xscale('log');
        plt.xlim(ml[0], ml[len(ml)-1]);
        plt.xlabel(r"\huge{$m_\chi$ [$\text{MeV}/c^2$]}");
        plt.yscale('log');
        plt.ylabel(r"\huge{$\overline{\sigma_e}$ [$\text{cm}^2$]}"); 
        plt.legend(loc=3);

        fig=plt.gcf();
        fig.set_size_inches(figure_size[0], figure_size[1]);
        
        ax=plt.gca();
        ax.tick_params(axis='both', which='major', labelsize=16);

        plt.show();

    def build(self, text):
        builtlbl="";
        color="#000000";
        style="solid";

        if "pi" in text:
            builtlbl+="$\pi$ "
            color=colors[0];
        elif "sigma_1" in text:
            builtlbl+="$\sigma_1$ ";
            color=colors[1];
        elif "sigma_2" in text:
            builtlbl+="$\sigma_2$ ";
            color=colors[2];
        elif "sigma_3" in text:
            builtlbl+="$\sigma_3$ ";
            color=colors[3];

        if "F_DM=1" in text:
            builtlbl+=r"$\left|\mathcal{M}\right|^2\propto 1$ ";
            style=linestyles[0];
        elif "F_DM=q" in text:
            builtlbl+=r"$\left|\mathcal{M}\right|^2\propto |\mathbf{q}|^2$";
            style=linestyles[1];
            color=np.multiply(0.7, color);
        elif "F_DM=2q" in text or "fdm=q^2" in text:
            builtlbl+=r"$\left|\mathcal{M}\right|^2\propto {|\mathbf{q}|}^4$";
            style=linestyles[2];
            color=np.multiply(0.5, color);
        if builtlbl=="":
            return text, color, style;
        else:
            return builtlbl, color, style;

    def build_sigma(self, dataset):
        sigmas=[];
        for R in dataset.Rl:
            sigmas.append(1e-37*3/(R*factor));
        return sigmas;


    def printout(self):
        if hasattr(self.datasets, "__len__"):
            for i in range(0, len(self.datasets)):
                print(":"+self.datasets[i].name);
                for k in self.datasets[i].kl:
                    print(k, end="\t");
                print("", end="\n");
                for R in self.datasets[i].Rl:
                    print(R, end="\t");
                print("", end="\n");
        else:
            print(":"+self.datasets.name);
            for k in self.datasets.kl:
                print(k, end="\t");
            print("", end="\n");
            for R in self.datasets.Rl:
                print(R, end="\t");
            print("", end="\n");

    def load(self, file):
        with open(file) as tsvfile:
            data=[];
            
            reader=csv.reader(tsvfile, delimiter="\t");
            
            names=[];
            kls=[];
            Rls=[];
            
            carriage=0;
            for row in reader:
                if carriage==0:
                    if ":" in row[0]:
                        names.append(row[0].split(":", 1)[1]);
                        carriage=1;
                elif carriage==1:
                    kls.append(list(map(lambda k: float(k), row)));
                    carriage=2;
                elif carriage==2:
                    Rls.append(list(map(lambda R: float(R), row)));
                    carriage=0;
            
            sets=[];
            for i in range(0, len(names)):
                sets.append(DataSet(names[i], kls[i], Rls[i]));
         
        return sets;


    def load_m(self, file):
        with open(file) as tsvfile:
            data=[];
            
            reader=csv.reader(tsvfile, delimiter="\t");
            
            names=[];
            mchis=[];
            Rls=[];
            
            carriage=0;
            for row in reader:
                if carriage==0:
                    if ":" in row[0]:
                        names.append(row[0].split(":", 1)[1]);
                        carriage=1;
                elif carriage==1:
                    mchis.append(list(map(lambda m: float(m), row)));
                    carriage=2;
                elif carriage==2:
                    Rls.append(list(map(lambda R: float(R), row)));
                    carriage=0;
            
            sets=[];
            for i in range(0, len(names)):
                sets.append(DataSet(names[i], mchis[i], Rls[i]));
         
        return sets;

class DataSet():
    name="";
    kl=[];
    Rl=[];

    def __init__(self, name, kl, Rl):
        self.name=name;
        self.kl=kl;
        self.Rl=Rl;

#normalize rates for different F_DM:s so that the values for the lowest k_f:s end up in the same position
def norm_rates(ds, mode="k"):
    if mode=="k":
        num=len(ds);
        #0-pi, 1-sigma1, 2-sigma2, 3-sigma3
        maxes=[0, 0, 0, 0];
        for i in range(0, len(ds)):
            if "pi" in ds[i].name and ds[i].Rl[0] > maxes[0]:
                maxes[0]=ds[i].Rl[0];
            elif "sigma_1" in ds[i].name and ds[i].Rl[0] > maxes[1]:
                maxes[1]=ds[i].Rl[0];
            elif "sigma_2" in ds[i].name and ds[i].Rl[0] > maxes[2]:
                maxes[2]=ds[i].Rl[0];
            elif "sigma_3" in ds[i].name and ds[i].Rl[0] > maxes[3]:
                maxes[3]=ds[i].Rl[0];
        
        constants=[];
        normed_sets=[];
        for d in ds:
            if "pi" in d.name:
                maxor=maxes[0];
            elif "sigma_1" in d.name:
                maxor=maxes[1];
            elif "sigma_2" in d.name:
                maxor=maxes[2];
            elif "sigma_3" in d.name:
                maxor=maxes[3]
            else:
                maxor=d.Rl[0];
        
            print(maxor/d.Rl[0]);
            normed_sets.append(DataSet(d.name, d.kl, np.multiply(d.Rl, maxor/d.Rl[0])));
            constants.append(maxor/d.Rl[0]);    

        return normed_sets, constants;
    elif mode=="m":
        num=len(ds);
        #0-pi, 1-sigma1, 2-sigma2, 3-sigma3
        maxes=[0, 0, 0, 0];
        for i in range(0, len(ds)):
            if "pi" in ds[i].name and ds[i].Rl[0] > maxes[0]:
                maxes[0]=ds[i].Rl[round(len(ds[i].Rl)/2)];
            elif "sigma_1" in ds[i].name and ds[i].Rl[0] > maxes[1]:
                maxes[1]=ds[i].Rl[round(len(ds[i].Rl)/2)];
            elif "sigma_2" in ds[i].name and ds[i].Rl[0] > maxes[2]:
                maxes[2]=ds[i].Rl[round(len(ds[i].Rl)/2)];
            elif "sigma_3" in ds[i].name and ds[i].Rl[0] > maxes[3]:
                maxes[3]=ds[i].Rl[round(len(ds[i].Rl)/2)];
        
        constants=[];
        normed_sets=[];
        for d in ds:
            if "pi" in d.name:
                maxor=maxes[0];
            elif "sigma_1" in d.name:
                maxor=maxes[1];
            elif "sigma_2" in d.name:
                maxor=maxes[2];
            elif "sigma_3" in d.name:
                maxor=maxes[3]
            else:
                maxor=d.Rl[0];
        
            print(maxor/d.Rl[0]);
            normed_sets.append(DataSet(d.name, d.kl, np.multiply(d.Rl, maxor/d.Rl[0])));
            constants.append(maxor/d.Rl[0]);    

        return normed_sets, constants;  
    else:
        raise Exception("Mode error");

if __name__=="__main__":
    if len(sys.argv)==1:
        plt.plot(El, Rl, label="noname");
        plt.xscale('log');
        plt.yscale('log');
        plt.legend();

        plt.show();
    else:
        if sys.argv[1]=="help" or sys.argv[1]=="-h":
            print("usage: dataplotter.py [filename] {-m --mass-mode}");
            print("in differential rate mode:  [normalize different F_DM (true/false)] {[plot sum (true/false)] [distinguish F_DM in summed rates (true/false)] [only sums (true/false)]}");
            print("in mass mode: [normalize different F_DM (true/false)]");
        else:
            fn=sys.argv[1];
            
            if sys.argv[2]=="-m" or sys.argv[2]=="--mass-mode":
                dp=DataPlotter(0, 0, fn, "m");
                if sys.argv[3]=="true":
                    normed_sets, constants=norm_rates(dp.datasets, "m");
                else:
                    normed_sets=dp.datasets;

                dp.datasets=normed_sets;
                dp.plot_m();
            else:
                if sys.argv[2]=="true":
                    normalize=True;
                else:
                    normalize=False;
                try:
                    if sys.argv[3]=="true":
                        plot_sum=True;
                    else:
                        plot_sum=False;
                except:
                    plot_sum=False;

                try:
                    if sys.argv[4]=="true":
                        dist_fdm=True;
                    else:
                        dist_fdm=False;
                except:
                    dist_fdm=False;

                try:
                    if sys.argv[5]=="true":
                        only_sum=True;
                    else:
                        only_sum=False;
                except:
                    only_sum=False; 

                dp=DataPlotter(0, 0, fn);
                if normalize==True:
                    normed_sets, constants=norm_rates(dp.datasets);
                else:
                    normed_sets=dp.datasets;

                dp.datasets=normed_sets;
                dp.plot(plot_sum, dist_fdm, only_sum);  
