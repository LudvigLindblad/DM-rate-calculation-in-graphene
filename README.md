Python tool fo calculating differential ejection rate of electrons from DM-electon scattering in graphene.

To run the code please download the files listed. The tabel of eigenvectors C (3 in total) are to large to be stored at Github but can be generated from Graphene_wavefunctions.nb or be sent by request. Do note that the eigenvectors are only needed for the sigma bands.

The calculations are run through integration_station.py 

The integrals over q and k are performed in rate_calc.py, as well as the normalization, bandenergies for pi and sigma. 

The integration over lattice momentum is done in integration_station.py as a sum on a grid. To alter the scattering amplitude please change Fdm from 1 to 2 or 3. The band and gridsize is also set in Integration_station.
Code to run an interpolation over lattice momentum can be found in Integration_station.py and give comparative results. Note however the interpolation were found to be somewhat unstables and needs to be expanded upon using it.

A benchmark method can be found in integration_method.py to determine optimal integration method, default is "Gauss-Legendre", but can be changed in integration_station.py

The constants used are given in SI units and are listed in constants.py 

Band energies and eigenvectors for sigma are gathered in loadtables.py 

The author of these tools are Julia Andersson, Ebba Grönfors, Christoffer Hellekant, Ludvig Lindblad and Fabian Resare.

For questions, bug reports or other suggestions please contact Ludli@student.chalmers.se, Resaref@student.chalmers.se 
or chrhelle@student.chalmers.se
