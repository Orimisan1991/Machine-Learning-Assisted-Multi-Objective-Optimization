import numpy as np
import matplotlib.pyplot as plt

# IEEE style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

properties = ['Density', 'Cp', 'k', 'Viscosity']

Al2200 = [1322.28, 1.286, 0.142, 1.48e-3]
Al3690 = [1329.60, 1.271, 0.155, 1.62e-3]
TiO2   = [1334.23, 1.268, 0.149, 1.71e-3]
CuO    = [1341.61, 1.255, 0.173, 1.85e-3]
R134a  = [1317.56, 1.323, 0.095, 1.00e-3]

x = np.arange(len(properties))

plt.figure(figsize=(8,5))

plt.plot(x,R134a,'ko-',linewidth=2,label='R134a')
plt.plot(x,Al2200,'bs--',linewidth=2,label='Al$_2$O$_3$ (2200)')
plt.plot(x,Al3690,'r^-.',linewidth=2,label='Al$_2$O$_3$ (3690)')
plt.plot(x,TiO2,'md:',linewidth=2,label='TiO$_2$')
plt.plot(x,CuO,'gv-',linewidth=2,label='CuO')

plt.xticks(x,properties)
plt.ylabel('Property Value')
plt.title('Thermophysical Properties of Nanorefrigerants at φ = 0.6%')
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig('Thermophysical_Properties.png',dpi=600,bbox_inches='tight')
plt.show()