'''
A pure-Python equilibrium model code with speedy t-z splining.

Introduction:
-------------

The equilibrium model is an analytic formalism to describe the evolution
of a set of baryonic galaxy properties, specifically the star formation
rate, gas fraction and metallicity [1]. After subsequent work on the
equilibrium model in [2], [3] and [4], this code now provides a Python
implementation of the formalism. In addition, the relationship between
cosmological age and redshift is splined outside the primary loop to 
avoid computing the redshift for each separate iteration.

Quickstart:
-----------
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

    ----------------------------------------------------------------
    |  from pyquilibrium import equilibrium_model                  |
    |                                                              |
    |  output = equilibrium_model(final_redshift = 0.0,            |
    |                             halo_mass = 8.2,                 |
    |                             hubble_const = 70.0,             |
    |                             omega_matter = 0.3)              |
    |                                                              |
    ----------------------------------------------------------------
    
Author:
--------
Ben Moews
Institute for Astronomy (IfA)
School of Physics & Astronomy
The University of Edinburgh

Libraries:
----------
Python 3.4.5
NumPy 1.11.3
SciPy 0.18.1
Astropy 1.3

References:
-----------
[1] Dave, R. et al. (2011), "The neutral hydrogen content of galaxies in
    cosmological hydrodynamic simulations", Monthly Notices of the Royal
    Astronomical Society, Vol. 434(3), pp. 2645-2663
[2] Asplund, M. et al. (2009), "The chemical composition of the sun",
    Annual Review of Astronomy & Astrophysics, Vol. 47(1), pp. 481-522
[3] Chabrier, G. (2003), "Galactic stellar and substellar initial mass
    function", The Publications of the Astronomical Society of the
    Pacific, Vol. 115(809), pp. 763-795
[4] Moews, B. et al. (2020), "Hybrid analytic and machine-learned baryonic 
    property insertion into galactic dark matter haloes", Monthly Notices
    of the Royal Asstronomical Society, Vol. 504(3), pp. 4024-4038
[5] Trujillo, I. et al. (2006), "The size evolution of galaxies since
    z~3: Combining SDSS, GEMS, and FIRES", The Astrophysical Journal,
    Vol. 650(1), pp. 18-41
[6] Erb, D. K. et al. (2006), "The mass-metallicity relation at z>~2",
    The Astrophysical Journal", Vol. 644(2), pp. 813-828
[7] Behroozi, P. S. et al. (2012), "The average star formation histories
    of galaxies in dark matter halos from z=0-8", The Astrophysical
    Journal, Vol. 770(1), Art. 57
[8] Faucher-Giguere, C. A. et al. (2011), "The baryonic assembly of dark
    matter haloes", Monthly Notices of the Royal Astronomical Society,
    Vol. 417(4), pp. 2982-2999
'''

# Import the necessary libraries
import sys
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy.interpolate import interp1d as splining

def equilibrium_model(final_redshift,
                      halo_mass,
                      hubble_const,
                      omega_matter,
                      eta_slope = None,
                      time_slope = None):
    # Check whether optional parameters were provided
    if eta_slope == None:
        eta_slope = -(1 / 3)
    if time_slope == None:
        time_slope = 0
    # Assign the necessary basic parameters to the respective variables
    baryon_frac, yield_factor, rec_rate, eta_old, mass_stars = basic_params()
    # Calculate the initial redshift, starting roughly at photon mass
    initial_redshift = 9 * np.power(10, (halo_mass - 8) / 2.72) - 1
    # Set Lambda assuming LCDM cosmology
    lambda_value = 1 - omega_matter
    # Set the halo mass based on the power-of-ten input
    halo_mass = np.power(10, halo_mass)
    time_difference = 0.01
    # Establish the flat LCDM cosmology that is used
    cosmology = FlatLambdaCDM(H0 = hubble_const, Om0 = omega_matter)
    # Get the final cosmological age depending on the final redshift
    if final_redshift >= 0:
        final_time = cosmology.age(final_redshift).value
    else:
        final_time = cosmology.age(0).value
    # Get the cosmological age depending on the initial redshift
    time = cosmology.age(initial_redshift)
    # If required via the inputs, print the full evolution
    if final_redshift == -1:
        print('equil: %f %f %f %f %f' % (time.value,
                                         final_redshift,
                                         final_time,
                                         halo_mass,
                                         hubble_const))
    # Begin the galaxy out of a state of equilibrium
    redshift_equilibrium_flag = 0
    # Determine a redshift range for the subsequent splining
    redshift_range = np.arange(0, 10, 0.1)
    # Get the cosmological age for the above redshift values
    time_range = list(map(cosmology.age, redshift_range))
    time_range = [time_range[i] / u.Gyr for i in range(len(time_range))]
    # Spline the relationship between cosmological age and redshift
    cosmology_spline = splining(x = time_range,
                                y = redshift_range,
                                fill_value = "extrapolate")
    # While not reaching the final time, evolve the galaxy
    while(time.value < final_time):
        # Get the current redshift from the time-redshift spline
        redshift = cosmology_spline(time)
        # Calculate the growth of the halo mass
        inflow = baryonic_inflow(halo_mass = halo_mass,
                                 redshift = redshift)
        halo_growth = time_difference * 1e9 * inflow
        # If the halo mass grows too fast, halve the time step
        if halo_growth > 0.01 * halo_mass:
            time_difference = time_difference * 0.5
        # If the halo mass grows too slow, double the time step
        if halo_growth < 0.001 * halo_mass:
            time_difference = time_difference * 2
        # Get the equilibrium relations from their respective functions
        star_formation_rate = equilibrium_condition(halo_mass = halo_mass,
                                                    redshift = redshift,
                                                    baryon_frac = baryon_frac,
                                                    eta_slope = eta_slope)
        redshift_gas = redshift_gas_function(halo_mass = halo_mass,
                                             redshift = redshift,
                                             yield_factor = yield_factor,
                                             eta_slope = eta_slope)
        gas_fraction = gas_frac(redshift = redshift,
                                time = time.value,
                                time_slope = time_slope,
                                mass_stars = mass_stars,
                                star_formation_rate = star_formation_rate,
                                redshift_gas = redshift_gas)
        # Update the halo mass with the halo mass growth
        halo_mass = halo_mass + halo_growth
        # Check whether equilibrium is reached
        check = equilibrium_check(gas_fraction = gas_fraction,
                                  halo_mass = halo_mass,
                                  eta_slope = eta_slope)
        # If in equilibrium, grow the stellar mass
        if (redshift_equilibrium_flag == 0) and (redshift < check):
            mass_stars = (mass_stars + time_difference * 1e9
                          * star_formation_rate * (1 - rec_rate))
        # If required via the inputs, print the full evolution
        if final_redshift == -1:
            halo_growth_exit = baryonic_inflow(halo_mass = halo_mass,
                                               redshift = redshift)
            eta_exit = eta_function(halo_mass = halo_mass,
                                    eta_slope = eta_slope)
            baryon_halo_exit = baryon_frac * halo_growth_exit
            print('%f %f %f %f %f %f %f %f %f' % (redshift,
                                                  time.value,
                                                  star_formation_rate,
                                                  baryon_halo_exit,
                                                  mass_stars,
                                                  halo_mass,
                                                  gas_fraction,
                                                  redshift_gas,
                                                  eta_exit))
        # Update the time for the next redshift computation
        time = time + (time_difference * u.Gyr)
    # If not printing the full evolution, return the final results
    if final_redshift >= 0:
        halo_growth_exit = baryonic_inflow(halo_mass = halo_mass,
                                           redshift = redshift)
        eta_exit = eta_function(halo_mass = halo_mass,
                                eta_slope = eta_slope)
        baryon_halo_exit = baryon_frac * halo_growth_exit
        log_star_formation_rate = np.log10(star_formation_rate)
        result = [redshift,
                  time.value,
                  log_star_formation_rate,
                  baryon_halo_exit,
                  mass_stars,
                  halo_mass,
                  gas_fraction,
                  redshift_gas,
                  eta_exit]
        return result

def basic_params():
    # The baryon fraction used in thesimulations in source [1]
    baryon_frac = 0.164286
    # The solar metal fraction from source [2]
    yield_factor = 0.0126
    # The recycled mass fraction (from SNe) from source [3]
    rec_rate  = 0.18
    # For computing redshift to time, not the mass load
    eta_old = 0.5
    # Set star mass to avoid zero at the beginning
    mass_stars = 1e-10
    # Return the parameters as the function output
    parameters = [baryon_frac,
                  yield_factor,
                  rec_rate,
                  eta_old,
                  mass_stars]
    return parameters

def baryonic_inflow(halo_mass,
                    redshift):
    # Calculate the halo mass part of the growth equation
    halo_power = np.power(halo_mass / 1e12, 0.15)
    # Calculate the redshift part of the growth equation
    redshift_power = np.power((1 + redshift) / 3, 2.25)
    # Calculate the complete halo growth as the final result
    halo_growth = 0.47e-9 * halo_mass * halo_power * redshift_power
    # Return the halo growth as the function output
    return halo_growth


def equilibrium_condition(halo_mass,
                          redshift,
                          baryon_frac,
                          eta_slope):
    # Get the baryonic inflow into the halo
    mass_grav = baryonic_inflow(halo_mass = halo_mass,
                                redshift = redshift)
    # Get the preventive feedback parameter
    zeta = zeta_function(halo_mass = halo_mass,
                         redshift = redshift)
    # Get the ejective feedback parameter
    eta = eta_function(halo_mass = halo_mass,
                       eta_slope = eta_slope)
    # Get the ratio of inflowing and ambient ISM gas metallicity
    alpha = alpha_metal(halo_mass = halo_mass,
                        redshift = redshift)
    # Calculate the star formation rate
    star_formation_rate = mass_grav * baryon_frac * zeta / (1 + eta) * alpha
    # Return the star formation rate as the function output
    return star_formation_rate

def gas_frac(redshift,
             time,
             time_slope,
             mass_stars,
             star_formation_rate,
             redshift_gas):
    # Initialize Sigma and disk radius as zero
    sigma, r_disk = np.zeros(2)
    # Calculate the time dependence
    time_dep = time *1e9 * 0.4 * np.power(mass_stars / 1e10, -0.3)
    # Get the specific SFR from the SFR and the star mass
    ssfr = specific_star_formation(star_formation_rate = star_formation_rate,
                                   mass_stars = mass_stars)
    # Check if a positive time slope is present
    if time_slope > 0:
        # Calculate disk radius according to source [5]
        r_disk = (3 * np.power(1 + redshift, -0.4)
                  * np.power(mass_stars / 5e10, 0.3333))
        # Calculate Sigma according to source [6]
        sigma = 183 * np.power((mass_stars * ssfr) / np.square(r_disk), 0.71)
        # Update the time dependence
        time_dep = time_dep * (sigma + 0.2 / redshift_gas) / sigma
    # Check if a negative time slope for small gas redshifts is present
    if redshift_gas < 0.0189  and time_slope < 0:
        # Calculate the time dependence
        time_dep = (time * 1e9 * np.power(redshift_gas
                           / 0.0189, time_slope) * 0.4
                           * np.power(mass_stars / 1e10, -0.3))
    # Calculate the gas fraction
    gas_fraction = 1 / (1 + 1 / (time_dep * ssfr))
    # Return the gas fraction as the function output
    return gas_fraction

def equilibrium_check(gas_fraction,
                      halo_mass,
                      eta_slope):
    # Get the ejective feedback parameter
    eta = eta_function(halo_mass = halo_mass,
                       eta_slope = eta_slope)
    # Calculate the equilibrium status check value
    check = (np.maximum(np.power(5 * gas_fraction * (1 + eta), 4 / 3)
             * np.power(halo_mass / 1e12, -0.2), 2))
    # Return the check result as the function output
    return check


def zeta_function(halo_mass,
                  redshift):
    # Initialize the quenching redshift as zero
    redshift_quench = 0
    # Calculate the photon mass
    mass_photo = (8 - 2.72 * np.power((1 + redshift) / 9, 0.2)
                  * np.log10((1 + redshift) / 9))
    # Calculate the photon redshift
    redshift_photo = (np.power(1 + 0.587 * np.power(halo_mass
                      / np.power(10, mass_photo), -2), -1.5))
    # Source [7]: SFR ~Mh^-(4/3) above quenching mass;-(4/3)=-0.25-1.083
    mass_quench = np.power(10, 11.7 + 0.25 * redshift)
    # Check if the halo mass is above quenching mass
    if halo_mass > mass_quench:
        redshift_quench = np.power(halo_mass / mass_quench, -1.083)
    # If not, set the quenching redshift to one
    else:
        redshift_quench = 1
    # Calculate the gravitational redshift according to source [8]
    redshift_grav = (0.47 * np.power((1 + redshift) / 4, 0.38)
                     * np.power(halo_mass / 1e12, -0.25))
    # Truncate the gravitational redshift with an upper limit of one
    if redshift_grav > 1:
        redshift_grav = 1
    # Calculate the wind mass
    mass_wind = 10 - 0.25 * redshift
    # Calculate the wind redshift
    intermediate = -np.sqrt((halo_mass / np.power(10, mass_wind)))
    redshift_wind = 1 - np.exp(intermediate)
    # Calculate the preventive feedback parameter
    zeta = redshift_photo * redshift_quench * redshift_grav * redshift_wind
    # Fudge dactor to match source [7]
    zeta = zeta * 0.5
    # Return the parameter as the function output
    return zeta

def eta_function(halo_mass,
                 eta_slope):
    # Calculate the ejective feedback parameter
    eta = np.power(halo_mass / 1e12, eta_slope)
    # Return the parameter as the function output
    return eta

def redshift_gas_function(halo_mass,
                          redshift,
                          yield_factor,
                          eta_slope):
    # Get the ejective feedback parameter
    eta = eta_function(halo_mass = halo_mass,
                       eta_slope = eta_slope)
    # Get the ratio of inflowing and ambient ISM gas metallicity
    alpha = alpha_metal(halo_mass = halo_mass,
                        redshift = redshift)
    # Calculate the gas redshift
    redshift_gas = yield_factor * alpha / (1 + eta)
    # Return the gas redshift as the function output
    return redshift_gas

def alpha_metal(halo_mass,
                redshift):
    # Calculate the ratio of inflowing and ambient ISM gas metallicity
    alpha = np.exp(-redshift) * np.power(halo_mass / 1e11, 2 / 3)
    # Tweaks for high and low values to accommodate source [7]
    if halo_mass > 1e12:
        alpha = alpha * np.power(halo_mass / 1e12, -(2 / 3))
    if alpha < 2:
        alpha = 1 + alpha / 2
        # Return alpha as the function output
    return alpha

def specific_star_formation(star_formation_rate,
                            mass_stars):
    # Calculate the specific SFR
    ssfr = star_formation_rate / mass_stars
    # Return the specific SFR as the function output
    return ssfr
