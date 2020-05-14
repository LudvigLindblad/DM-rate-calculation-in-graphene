import numpy as np
import scipy as sp
from mpmath import mp
import csv
import sys
import re
import matplotlib.pyplot as plt

import constants as ct

lead=__file__.split("/loadtables.py", 1)[0];

#file names
Es1=lead+"/tables/Esigma1.txt";
Es2=lead+"/tables/Esigma2.txt";
Es3=lead+"/tables/Esigma3.txt";

C1=lead+"/tables/C_Sigma1_500.txt";
C2=lead+"/tables/C_Sigma2_500.txt";
C3=lead+"/tables/C_Sigma3_500.txt";

#generating lx, ly table constants
a=ct.a;
c=4*np.pi/(a*(27**(1/2)));
num=100;
num500=500;
fill=1.1;
lxl=np.linspace(-fill*3**(1/2)*c/2,fill*3**(1/2)*c/2, num);  #Ematrix expecting 100x100
lyl=np.linspace(-fill*c, fill*c, num);
lmx, lmy=np.meshgrid(lxl, lyl);
lxl5=np.linspace(-fill*3**(1/2)*c/2,fill*3**(1/2)*c/2, num500);  #Cmatrix expecting 500x500
lyl5=np.linspace(-fill*c, fill*c, num500);
lmx5, lmy5=np.meshgrid(lxl5, lyl5);

class Table():
    Es1mat=[];
    Es2mat=[];
    Es3mat=[];

    C1mat=[];
    C2mat=[];
    C3mat=[];
    
    def __init__(self):
        self.Es1mat=get_Ematrix(Es1);
        self.Es2mat=get_Ematrix(Es2);
        self.Es3mat=get_Ematrix(Es3);

        self.C1mat=get_C_Sigma(C1);
        self.C2mat=get_C_Sigma(C2);
        self.C3mat=get_C_Sigma(C3);
 
    def C_Sigma(self, lx, ly, band):
        i=np.argmin(np.abs(np.subtract(lxl5, lx)));
        j=np.argmin(np.abs(np.subtract(lyl5, ly)));

        if band=="sigma1":
            return self.C1mat[i][j];
        elif band=="sigma2":
            return self.C2mat[i][j];
        elif band=="sigma3":
            return self.C3mat[i][j];
        else:
            print("band error");
            return 0;

    def Ematrix(self, lx, ly, band):
        i=np.argmin(np.abs(np.subtract(lxl, lx)));
        j=np.argmin(np.abs(np.subtract(lyl, ly)));
 
        if band=="sigma1":
            return self.Es1mat[i][j];
        elif band=="sigma2":
            return self.Es2mat[i][j];
        elif band=="sigma3":
            return self.Es3mat[i][j];
        else:
            print("band error");
            return 0;

#load E values
def get_Ematrix(file):
    data=[]

    with open(file) as tsvfile:
        reader=csv.reader(tsvfile, delimiter=",");
        for row in reader:
            data.append(row);

        Em=[];
        for row in data:
            Er=[];
            for i in range(0, len(row)):
                if i==0:
                    Er.append(float(row[i].split("{")[1].strip()));
                elif i==len(row)-1:
                    Er.append(float(row[i].split("}")[0].strip()));
                else:
                    Er.append(float(row[i].strip()));
            Em.append(Er);

    return Em;

def get_C_Sigma(file):
    data=[];

    with open(file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter=",");
        for row in reader:
            data.append(row);
    Cmat=[];
    for y in range(0, num500):
        Crow=[];
        for x in range(0, num500):
            sm=[];
            real=data[y*num500+x][0].split(" ")[0]; #Read data from string
            sign=data[y*num500+x][0].split(" ")[1]; #Split into Real and Imaginary parts
            imag=data[y*num500+x][0].split(" ")[2].split("i")[0];
            if sign=="+":
                sm.append(float(real)+1j*float(imag));
            elif sign=="-":
                sm.append(float(real)-1j*float(imag));
            
            for s in range(1, 6):
                real=data[y*num500+x][s].split(" ")[1];
                sign=data[y*num500+x][s].split(" ")[2];
                imag=data[y*num500+x][s].split(" ")[3].split("i")[0];
                if sign=="+":
                    sm.append(float(real)+1j*float(imag));
                elif sign=="-":
                    sm.append(float(real)-1j*float(imag));
                
            Crow.append(sm);   
        Cmat.append(Crow);
    return Cmat;
