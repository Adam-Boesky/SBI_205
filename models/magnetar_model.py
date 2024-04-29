import numpy as np
from scipy.integrate import cumtrapz
import astropy.constants as c
import astropy.units as u
from scipy import interpolate
from astropy.cosmology import WMAP9 as cosmo
import extinction

ext_law = extinction.fitzpatrick99(np.linspace(1000,10000,100), 1.0, 3.1)



DAY_CGS = 86400.0
M_SUN_CGS = c.M_sun.cgs.value
C_CGS = c.c.cgs.value
beta = 13.7
KM_CGS = u.km.cgs.scale
RAD_CONST = KM_CGS * DAY_CGS
STEF_CONST = 4. * np.pi * c.sigma_sb.cgs.value
ANG_CGS = u.Angstrom.cgs.scale
MPC_CGS = u.Mpc.cgs.scale

DIFF_CONST = 2.0 * M_SUN_CGS / (beta * C_CGS * KM_CGS)
TRAP_CONST = 3.0 * M_SUN_CGS / (4. * np.pi * KM_CGS ** 2)
FLUX_CONST = 4.0 * np.pi * (
        2.0 * c.h * c.c ** 2 * np.pi).cgs.value * u.Angstrom.cgs.scale
X_CONST = (c.h * c.c / c.k_B).cgs.value


# Central wavelengths for LSST...
wv_central = np.asarray([3751.36, 4741.64, 6173.23, 7501.62, 8679.19, 9711.53])
frequencies = C_CGS / (wv_central * u.Angstrom.cgs.scale)

# Build SLSN Model...including improved Arnett model
def gen_magnetar_model(t, theta, filt=None, dist_const=None, ebv=None, redshift=None):
    pspin, bfield, mns, \
    thetapb, texp, kappa, \
    kappagamma, mej, vej, tfloor = np.array(theta)

    # Send stuff to np
    t = np.array(t)
    redshift = np.array(redshift)
    dist_const = np.array(dist_const)
    filt = np.array(filt)


    Ep = 2.6 * (mns / 1.4) ** (3. / 2.) * pspin ** (-2)
    # ^ E_rot = 1/2 I (2pi/P)^2, unit = erg
    tp = 1.3e5 * bfield ** (-2) * pspin ** 2 * (
        mns / 1.4) ** (3. / 2.) * (np.sin(thetapb)) ** (-2)
    tau_diff = np.sqrt(DIFF_CONST * kappa *
                                mej / vej) / DAY_CGS

    A = (TRAP_CONST * kappagamma * mej / (vej ** 2)) / DAY_CGS ** 2
    td2 =  tau_diff ** 2

    test_t = np.linspace(0,t.max(), 100)
    lum_inp = 2.0 * Ep / tp / (1. + 2.0 * test_t * DAY_CGS / tp) ** 2

    # print('HERE:', np.exp((test_t/tau_diff)**2), 'HERE DONE')
    integrand = 2* lum_inp * test_t/tau_diff * np.exp((test_t/tau_diff)**2)  * 1e52
    # print(test_t/tau_diff)
    # print('integrand = ', integrand)

    multiplier =  (1.0 - np.exp(-A*test_t**-2)) * np.exp(-(test_t/tau_diff)**2) 
    l_out = multiplier * cumtrapz(integrand, test_t, initial = 0)
    # print('lout:', l_out)
    lum_function = interpolate.interp1d(test_t, l_out)
    # print('TEST_T: ', test_t)
    # print('HERE:  ', t.min())
    luminosities = lum_function(t)
    # print('Ls: ', luminosities)

    #Do BB calculation
    radius = RAD_CONST * vej * ((t - texp) * ((t-texp)>0))
    temperature = (luminosities / (STEF_CONST * radius**2))**0.25# * (1e52)**0.25
    gind = (temperature < tfloor) | np.isnan(temperature)
    temperature = np.nan_to_num(temperature)
    notgind = np.invert(gind)
    temperature = (0. * temperature) + (temperature * notgind) + (tfloor * gind)

    radius = np.sqrt(luminosities / (STEF_CONST * temperature**4))
    sed = FLUX_CONST * radius**2 / (ANG_CGS * wv_central[filt]/(1.+redshift))**5 /  \
                np.expm1(X_CONST / (ANG_CGS * wv_central[filt]/(1.+redshift)) / temperature)
    fluxes = sed / ANG_CGS * (C_CGS / ((frequencies[filt] * (1.+redshift)) ** 2))
    #gind = np.where(wv_central[filt]<3000)
    #sed[gind] = FLUX_CONST * radius**2 / (ANG_CGS * wv_central[filt[gind]]/3000.0)**5 / \
    #                np.expm1(X_CONST / (ANG_CGS * wv_central[filt[gind]]/3000.0) / temperature)
    fluxes = sed / ANG_CGS * (
                    C_CGS / (frequencies[filt] ** 2))
    # print('flux',fluxes)
    mags = - 2.5 * (np.log10(fluxes) - dist_const) - 48.60
    dist_const2 = np.log10(4. * np.pi * (3.086*10**19) ** 2)
    # print(dist_const2)
    mag_other = - 2.5 * (np.log10(fluxes) - dist_const2) - 48.60 + 38.33857468411047
    # print(mags, mag_other)
    return mags
# bfield = 1.6783
# pspin = 6.3101
# mns = 2.0
# thetapb = np.pi/2.0
# texp = 0.0
# kappa = 0.1126
# kappagamma = 28.7036
# mej = 3.2037
# vej = 6394.03
# tfloor = 5247.38
# redshift = 0.307
# lumdist = cosmo.luminosity_distance(redshift)
# print(cosmo.distmod(redshift))
# dist_const =  np.log10(4. * np.pi * (lumdist.cgs.value) ** 2)

# theta = [pspin, bfield, mns, \
# thetapb, texp, kappa, \
# kappagamma, mej, vej, tfloor]
# filters = np.asarray(np.zeros(100)+2.0, dtype=int)
# print(gen_magnetar_model(np.linspace(0,100,100),theta,filt=filters,redshift=0.1,ebv=0,dist_const=dist_const))
