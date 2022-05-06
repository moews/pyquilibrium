# PyQuilibrium

### A pure-Python equilibrium model code with speedy t-z splining.

The equilibrium model is an analytic formalism to describe the evolution of a set of baryonic galaxy properties, specifically the star formation rate, gas fraction and metallicity ([Davé et al., 2019](https://arxiv.org/abs/1302.3631)). After subsequent work on the equilibrium model in [Asplund et al. (2009)](https://arxiv.org/abs/0909.0948), [Chabrier (2003)](https://arxiv.org/abs/astro-ph/0304382) and [Moews et al. (2020)](https://arxiv.org/abs/2012.05820), this code now provides a Python implementation of the formalism. In addition, the relationship between cosmological age and redshift is splined outside the primary loop to avoid computing the redshift for each separate iteration.

### Quickstart

If you want to start quickly, 'equilibrium' only NEEDS four inputs:

1. The final redshift to which the galaxy is evolved
2. The initial halo mass as an exponential of base 10
3. The hubble constant that is to be used by the code
4. The matter density parameter (commonly as Omega_m)

In addition, thera are two optional parameters that can be set:

5. The exponent used for calculating the ejective feedback parameter
    'eta' in the function 'eta_function()'
6. The exponent used for calculating the gas fraction in 'gas_frac()'

An example with the four required parameters would look like this:
    
```python
from equilibrium import equilibrium_model

output = equilibrium_model(final_redshift = 0.0,
                           halo_mass = 8.2,
                           hubble_const = 70.0,
                           omega_matter = 0.3)
```

### Libraries

- Python 3.4.5
- NumPy 1.11.3
- SciPy 0.18.1
- Astropy 1.3
