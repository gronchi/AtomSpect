"""Every public name of the original single-file module must stay importable."""
import os
os.environ.setdefault("MPMATH_NOGMPY", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
import matplotlib
matplotlib.use("Agg")
import pytest
import AtomSpect

NAMES = """
c_light hbar hh muB muBcm g_L g_S kboltz m_prot m_elec q_charge hhev
lande_gi lande_gj lande_gj2 lande_gf Delta_E_HFS
Hz_to_nm nm_to_Hz ev_to_MHz cm_to_nm cm_to_Hz cm_to_ev ev_to_cm Vac_to_air air_to_vac
A_Ein A_Ein2 plotarray parse_plot_data get_target_axis consolidate_legend PlotFunction
Zslider Zslider_Pol enable_manual_input MakeSlider
Gaussian GaussianInstrum Voigt Lorrentzian Normalize
make_Jmj polarization CGCoeff Zeemanlow ZeemanHigh ZeemanhighHFS CGw3 dipolestr stateprocess
Zeeman_signal Zeeman_sig_norad Zeeman_func HZeeman stateprocess_HFS dipolestr_HFS
Zeeman_signal_HFS Zeeman_func_HFS HZeeman_HFS FastZfan ZeemanFan plotZfan AtomSpect_Main
MultiSpec Convol_Sticks make_stickbins Convol_Spect
read_Spectra read_dict read_HFSInput Savenpy_Dict Loadnpy_Dict
""".split()


@pytest.mark.parametrize("name", NAMES)
def test_name_available(name):
    assert hasattr(AtomSpect, name)
