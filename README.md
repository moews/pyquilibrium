# PyQuilibrium

### Python implementation of the equilibrium model of galaxy evolution

The equilibrium model is an analytic formalism to describe the evolution
of a set of baryonic galaxy properties, specifically the star formation
rate, gas fraction and metallicity [1]. After subsequent work on the
equilibrium model in [2] and [3], this code now provides a pure-Python
implementation of the formalism, complete with comments, adherence to
the PIP-8 coding conventions and references to research results used in
calculations at the locations of use. In addition, the relationship
between cosmological age and redshift is splined outside the primary
loop to avoid computing the redshift for each separate iteration.

### Quickstart
If you want to start quickly, 'equilibrium' only NEEDS four inputs:
(1) The final redshift to which the galaxy is evolved
(2) The initial halo mass as an exponential of base 10
(3) The hubble constant that is to be used by the code
(4) The matter density parameter (commonly as Omega_m)
In addition, thera are two optional parameters that can be set:
(5) The exponent used for calculating the ejective feedback parameter
    'eta' in the function 'eta_function()'
(6) The exponent used for calculating the gas fraction in 'gas_frac()'
An example with the four required parameters would look like this:
    
```python
from equilibrium import equilibrium_model

output = equilibrium_model(final_redshift = 0.0,
                           halo_mass = 8.2,
                           hubble_const = 70.0,
                            omega_matter = 0.3)
```

### Libraries

Python 3.4.5
NumPy 1.11.3
SciPy 0.18.1
Astropy 1.3

### References

[1] Dave, R. et al. (2011), "The neutral hydrogen content of galaxies in
    cosmological hydrodynamic simulations", Monthly Notices of the Royal
    Astronomical Society, Vol. 434(3), pp. 2645-2663
[2] Asplund, M. et al. (2009), "The chemical composition of the sun",
    Annual Review of Astronomy & Astrophysics, Vol. 47(1), pp. 481-522
[3] Chabrier, G. (2003), "Galactic stellar and substellar initial mass
    function", The Publications of the Astronomical Society of the
    Pacific, Vol. 115(809), pp. 763-795
[4] Trujillo, I. et al. (2006), "The size evolution of galaxies since
    z~3: Combining SDSS, GEMS, and FIRES", The Astrophysical Journal,
    Vol. 650(1), pp. 18-41
[5] Erb, D. K. et al. (2006), "The mass-metallicity relation at z>~2",
    The Astrophysical Journal", Vol. 644(2), pp. 813-828
[6] Behroozi, P. S. et al. (2012), "The average star formation histories
    of galaxies in dark matter halos from z=0-8", The Astrophysical
    Journal, Vol. 770(1), Art. 57
[7] Faucher-Giguere, C. A. et al. (2011), "The baryonic assembly of dark
    matter haloes", Monthly Notices of the Royal Astronomical Society,
    Vol. 417(4), pp. 2982-2999
