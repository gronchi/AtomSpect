# Algorithmic Calculation of Transition Strengths and Einstein Coefficients in the Zeeman–Paschen–Back Regime

## An external-data implementation specification for laboratory plasmas at 0.1–10 T

**Revision:** 2.0 — 13 October 2026  
**Primary output:** field-dependent atomic energies, E1 component strengths, transition frequencies, and spontaneous-emission rates.  
**Atomic scope:** atoms and ions, initially low-to-mid nuclear charge, with an explicit route for intermediate-coupling and high-$Z$ species.  
**Input philosophy:** the downstream program does not evaluate radial wavefunctions. It uses measured or calculated atomic data supplied externally. This does **not** mean that energies and unsigned transition rates determine every possible magnetic-field spectrum.

This revision retains the original sequence—external inputs, magnetic Hamiltonian, dipole amplitudes, strengths, Einstein coefficients, verification, and workflow—but separates three physically different data models. It corrects the SI prefactors, Hamiltonian/dipole phase consistency, statistical-weight treatment, and Paschen–Back limiting statements. The original isolated-LS-term construction remains available as Model B; it is no longer presented as a general intermediate-coupling calculation or an exact reproduction of ADAS603.

**How to read the evidence:** references `[R1]`–`[R12]` identify external atomic-data definitions, documentation, and research. The matrix construction, data contract, decision rules, algebraic deductions, and synthetic benchmark in this guide are an implementation specification derived under the stated assumptions. They are not a claim that any particular ion has already been validated over the full field range. Numerical screening thresholds must be selected for the intended observable and accuracy.

---

## Contents

1. Physical foundations and choice of model
2. Required external inputs and data provenance
3. Notation, units, and phase conventions
4. Convert external atomic data into zero-field line strengths
5. Model A: isolated-$J$ Zeeman components from individual line data
6. Model B: isolated-LS-term Zeeman–Paschen–Back calculation
7. Model C: general magnetic mixing using external reduced matrix elements
8. Common calculation of amplitudes, strengths, and Einstein coefficients
9. Hyperfine structure and isotopes
10. Applicability at 0.1–10 T and extension to high $Z$
11. From atomic components to laboratory-plasma spectra
12. Implementation data contract and algorithmic workflow
13. Verification, invariants, and numerical benchmark
14. Uncertainty, incomplete data, and reporting rules
15. References and revision notes

---

## 1. Physical foundations and choice of model

### 1.1 The problem being solved

Choose the magnetic field as the quantization axis,

$$
\mathbf B=B\hat{\mathbf z}.
$$

The basic electronic model is

$$
H(B)=H_{\rm atom}^{(0)}+H_{\rm mag}(B),
$$

where $H_{\rm atom}^{(0)}$ already includes the zero-field structure represented by the input levels. In a nonrelativistic LS model,

$$
H_{\rm mag}(B)=\frac{\mu_B B}{\hbar}(L_z+g_sS_z),
\qquad
\mu_B=\frac{e\hbar}{2m_e}>0,
$$

with $e$ the positive elementary charge and $g_s\simeq2.0023193$ the **positive magnitude** of the free-electron spin factor. A source may instead quote a negative electron $g$-factor; do not substitute that signed number into this Hamiltonian. The positive form above already accounts for the negative electronic magnetic moment through $H=-\boldsymbol\mu\cdot\mathbf B$. [R2, R8]

**Do not add a second spin–orbit term to experimental fine-structure energies.** That would count the same splitting twice. An explicit spin–orbit model is needed only when constructing or parametrizing the zero-field Hamiltonian itself.

Electric dipole radiation is described by

$$
\mathbf D=-e\sum_k\mathbf r_k.
$$

A field-independent dipole matrix in a fixed basis is an input-model assumption: the relevant zero-field atomic states and their operator matrix elements are assumed to span the field-dressed states adequately. It is **not** a consequence of $\mathbf D$ commuting with the magnetic interaction. In fact,

$$
[L_z,D_q]=q\hbar D_q,
\qquad
[S_z,D_q]=0,
\qquad
[H_{\rm mag},D_q]=q\mu_BB D_q.
$$

The Wigner–Eckart theorem determines angular dependence. It does not determine unknown orbital matrix elements or guarantee unchanged atomic structure at every field.

### 1.2 The three supported data models

| Model | External atomic information | What can be computed | Principal restriction |
|---|---|---|---|
| **A — isolated $J$ levels** | Energies, $J$, parity, individual E1 $A$ or $f$ or line strength, and reliable level $g_J$ values | Zeeman component strengths and linear magnetic shifts for each independent line | Magnetic mixing with other electronic levels must be negligible for the required accuracy |
| **B — isolated LS terms** | Complete $J$-resolved energies of each term, credible $L,S$ assignments, and a normalization for each orbital term-to-term channel | Fine-structure Zeeman and incomplete/complete Paschen–Back mixing within each retained term | LS angular wavefunctions and the common reduced orbital norm must be adequate; omitted interactions must be small |
| **C — general matrix-element model** | Zero-field levels and phase-consistent reduced E1 and electronic magnetic-interaction matrix elements across a sufficient basis | Magnetic mixing of arbitrary electronic coupling schemes; field-induced E1 components | Required matrix elements and a converged basis must actually be available |

Hyperfine structure is an extension of these models, not a substitute for selecting the electronic model. Electric fields, diamagnetism, and plasma dynamics require additional operators or a broader line-shape calculation.

**Model A is not restricted to pure LS coupling.** For an isolated exact zero-field level with definite $J$, rotational symmetry fixes the magnetic-sublevel E1 pattern even when its internal wavefunction is relativistic or strongly mixed. An experimentally established $A$-value and $g_J$ can therefore be enough for a high-$Z$ line in the isolated-$J$ regime. Conversely, an LS term label alone does not justify Model B. NIST describes labels as assignments to potentially mixed eigenvectors, not guarantees of purity. [R2, R3]

### 1.3 Why 0.1–10 T is not one universal coupling regime

The useful magnetic energy scale is

$$
\frac{\mu_BB}{h}\simeq13.99624\,B[\mathrm T]\ \mathrm{GHz},
$$

$$
\frac{\mu_BB}{hc}\simeq0.4668645\,B[\mathrm T]\ \mathrm{cm}^{-1}.
$$

Here and below, quoting an energy divided by $hc$ in $\mathrm{cm}^{-1}$ includes the appropriate metre-to-centimetre conversion. [R2, R8]

| $B$ | $\mu_BB/h$ | $\mu_BB/(hc)$ | $\mu_BB$ |
|---:|---:|---:|---:|
| 0.1 T | 1.3996 GHz | 0.046686 $\mathrm{cm}^{-1}$ | $5.7884\times10^{-6}$ eV |
| 1 T | 13.996 GHz | 0.466864 $\mathrm{cm}^{-1}$ | $5.7884\times10^{-5}$ eV |
| 10 T | 139.962 GHz | 4.66864 $\mathrm{cm}^{-1}$ | $5.7884\times10^{-4}$ eV |

These are scale factors, not maximum shifts: actual matrix elements contain angular factors. Compare the coupling of each pair of relevant states with **their energy separation**, not with the optical transition energy or ionization potential.

For a first screening of Models A and C, let

$$
\eta_{ab}^{(M)}(B)=
\frac{\mu_B|B|\,|T_{ab}^{(M)}|}
{|E_a^{(0)}-E_b^{(0)}+\mu_BB(g_a-g_b)M|},
$$

where $T_{ab}^{(M)}$ is the dimensionless off-diagonal magnetic matrix element. This denominator uses diagonal magnetic shifts as a screening approximation. Near a crossing, or when it becomes small, use full diagonalization rather than this expression.

Small $\eta$ suggests weak state admixture. It does **not** by itself guarantee a small relative error in a weak or cancelling transition amplitude. If $T_{ab}$ is unavailable, an energy-gap comparison is only a heuristic and cannot certify the approximation.

---

## 2. Required external inputs and data provenance

### 2.1 Level records

For every zero-field level retain:

- A unique level ID, element, charge state, and isotope specification where relevant.
- Energy and uncertainty; energy unit and common reference zero.
- Exact $J$ and parity; configuration and term label as metadata.
- $L,S$ and parentage identifiers when Model B is used.
- Experimental or calculated $g_J$, with source and uncertainty when available.

Store distinct states with the same apparent term symbol separately. The identifier $\gamma$ includes configuration and any additional parentage needed to distinguish terms. A string such as `3P` is not a unique state identifier.

Represent angular momenta internally using integers $2J,2M,2S,2I$; convert to exact half-integers for angular algebra. Do not infer quantum numbers by rounding approximate floating-point values.

### 2.2 Radiative records

Each transition needs upper/lower level IDs, multipole type, and at least one of:

- An individual zero-field transition probability $A_{ul}$.
- An absorption oscillator strength $f_{lu}$ or $\log_{10}(w_lf_{lu})$.
- A reduced line strength with an explicit operator and unit convention.
- A lifetime **and** a radiative branching fraction for the desired branch.
- A signed or complex reduced transition matrix element.

Here $w_l=2J_l+1$ is a statistical weight, not a magnetic $g$-factor. Relative measured line intensities are not substitutes for $A$-values: they also depend on upper-state populations, excitation conditions, and measurement geometry. [R4]

Keep the source frequency or wavelength used with the radiative datum, whether it is vacuum or air, and whether it is observed or derived from levels. When updating energies, record whether the published rate has already been rescaled to those energies.

### 2.3 Practical external sources

**NIST Atomic Spectra Database.** Use critically evaluated energies and radiative data where available. Preserve bibliographic references and uncertainty categories. Request explicit units and distinguish $A$ from $w_uA$, and $f$ from $\log(gf)$. A prime attached to an accuracy code can indicate that a multiplet was split into lines using an LS assumption; those lines do not independently validate LS strength ratios. [R4]

**OPEN-ADAS `adf04`.** These datasets provide level energies, spontaneous-emission coefficients, and collisional data for a specified ion model. Inspect the file comments and resolution: an LS-term-resolved entry is not a measured $J$-resolved fine-structure level. `adf04` process codes are not a universal E1/M1/E2 classifier. Do not treat every radiative coefficient as an E1 datum or every displayed statistical-weight field as $2J+1$ without checking the dataset definition. [R5]

**Atomic-structure providers or published matrix-element tables.** Model C requires more than a typical line list. Request electronic magnetic matrix elements and signed E1 reduced matrix elements in a shared state-phase convention. HFSZEEMAN95 is an example of a structure-based workflow that constructs magnetic mixing and transition rates from such quantities. Its upstream GRASP calculation uses atomic state functions; a downstream spectrum program can instead consume exported matrices. This guide does not assume that all required exports are available in a public database. [R7]

### 2.4 What is, and is not, recoverable from unsigned rates

An E1 $A$-value fixes the magnitude squared of one reduced dipole matrix element. It does not determine its phase, any other independent orbital channel, or an off-diagonal magnetic matrix element.

In Model A, a single isolated line does not require relative reduced-amplitude phases. In Model B, angular algebra fixes relative amplitudes within one orbital channel, leaving one irrelevant common phase. In Model C, different paths can interfere after mixing; unspecified relative phases then make the prediction underdetermined.

An overall rephasing of any basis state is harmless **only when every operator and coefficient involving that state is transformed consistently**. Setting every imported reduced amplitude to its positive square root while retaining an independently chosen magnetic matrix is not such a consistent rephasing.

---

## 3. Notation, units, and phase conventions

### 3.1 Definitions that must remain distinct

| Symbol | Definition |
|---|---|
| $S$ | Electronic spin quantum number |
| $\mathscr S_{ab}^{(0)}$ | Zero-field reduced E1 line strength between electronic levels |
| $s_{ul}(B)$ | Strength between specified field-dressed magnetic states |
| $R$ | Orbital reduced dipole matrix element for one LS term-to-term channel |
| $r_{ab}$ | Electronic reduced dipole matrix element in Model C |
| $\mathcal G_{ul}$ | Dimensionless angular amplitude in Model B |
| $g_J$ | Magnetic Landé factor |
| $w_J=2J+1$ | Statistical weight of an electronic level |
| $A_{ul}$ | Spontaneous-emission rate, integrated over photon directions and polarizations |
| $A_{\rm hfs},B_{\rm hfs}$ | Hyperfine constants, not radiative rates or magnetic-field strengths |

### 3.2 Recommended unit conventions

Use one explicit internal energy convention. SI joules are straightforward. Wavenumbers in $\mathrm{cm}^{-1}$ are also convenient, provided **every Hamiltonian term** is converted to those units before diagonalization:

$$
E[\mathrm J]=hc\,(100\,\widetilde E[\mathrm{cm}^{-1}]),
$$

$$
\nu[\mathrm{Hz}]=100c\,\Delta\widetilde E[\mathrm{cm}^{-1}],
\qquad
\lambda_{\rm vac}[\mathrm{nm}]=\frac{10^7}{\Delta\widetilde E[\mathrm{cm}^{-1}]}.
$$

For dipoles, use either $\mathrm{C\,m}$ or atomic units throughout:

$$
1\ \mathrm{a.u.\ of\ dipole}=ea_0,
\qquad
1\ \mathrm{a.u.\ of\ E1\ strength}=(ea_0)^2.
$$

Numerically, $(ea_0)^2\simeq7.18825\times10^{-59}\ \mathrm{C^2m^2}$. Obtain a consistent constants set and record its version rather than mixing rounded constants from different tables. [R8]

Only vacuum frequencies or vacuum wavelengths belong in radiative formulas. Convert an air wavelength using its specified refractive-index convention or use consistent level differences. [R4]

### 3.3 Tensor and reduced-matrix-element convention

Adopt

$$
D_0=D_z,
\qquad
D_{\pm1}=\mp\frac{D_x\pm iD_y}{\sqrt2}.
$$

Use the following Wigner–Eckart convention everywhere:

$$
\boxed{
\langle aJ_aM_a|D_q|bJ_bM_b\rangle
=
(-1)^{J_a-M_a}
\begin{pmatrix}J_a&1&J_b\\-M_a&q&M_b\end{pmatrix}
r_{ab},
\quad
r_{ab}=\langle aJ_a\|\mathbf D\|bJ_b\rangle.
}
$$

The $3j$ symbol enforces $q=M_a-M_b$. Hermiticity requires

$$
D_q^\dagger=(-1)^qD_{-q},
\qquad
r_{ba}=(-1)^{J_b-J_a}r_{ab}^{*}.
$$

Some structure programs use differently normalized reduced elements. Convert their definitions; do not infer compatibility merely because the output is called a “reduced matrix element.” Recovery of a published $A$-value is an essential normalization check.

For Model B use Condon–Shortley Clebsch–Gordan coefficients with **orbital angular momentum first**:

$$
|LSJM\rangle
=
\sum_{m_L,m_S}
\langle Lm_L,Sm_S|JM\rangle
|Lm_L\rangle|Sm_S\rangle.
$$

### 3.4 Eigenvector storage

In the following formulas, the columns of $C$ are eigenvectors:

$$
|u\rangle=\sum_a C_{au}|a\rangle,
\qquad C^\dagger C=I.
$$

An operator with row index in the upper basis and column index in the lower basis transforms as

$$
D_q^{\rm dressed}=C_u^\dagger D_q^{(0)}C_l.
$$

Do not use a transpose instead of a conjugate transpose in a general implementation. Real arithmetic is a valid optimization only when the complete chosen representation is real.

---

## 4. Convert external atomic data into zero-field line strengths

### 4.1 Definition of a reduced E1 line strength

For an individual zero-field electronic line,

$$
\mathscr S_{ul}^{(0)}
=
|\langle uJ_u\|\mathbf D\|lJ_l\rangle|^2.
$$

It equals the sum of squared magnetic-sublevel matrix elements over all upper and lower $M$ values and all $q$ values. It is **not** the strength of one magnetic component.

### 4.2 From an Einstein $A$-value

In SI units,

$$
\boxed{
A_{ul}^{(0)}
=
\frac{\omega_0^3}{3\pi\epsilon_0\hbar c^3}
\frac{\mathscr S_{ul}^{(0)}}{2J_u+1}
=
\frac{16\pi^3\nu_0^3}{3\epsilon_0hc^3}
\frac{\mathscr S_{ul}^{(0)}}{2J_u+1}.
}
$$

Thus

$$
\boxed{
\mathscr S_{ul}^{(0)}
=
\frac{3\epsilon_0hc^3}{16\pi^3\nu_0^3}
(2J_u+1)A_{ul}^{(0)}.
}
$$

These are the SI E1 relations tabulated by NIST. A Gaussian-unit prefactor without $\epsilon_0$ must not be used with SI dipoles. [R1]

A convenient equivalent conversion is

$$
A_{ul}^{(0)}[\mathrm{s}^{-1}]
=
\frac{C_{\rm E1}}{(2J_u+1)\lambda_{\rm vac}[\text{\AA}]^3}
\mathscr S_{ul}^{(0)}[\mathrm{a.u.}],
$$

$$
C_{\rm E1}
=\frac{16\pi^3(ea_0)^2}{3\epsilon_0h}\,10^{30}
\simeq2.0261269\times10^{18}.
$$

Use the defining expression when testing at high numerical precision. Replacing ångströms with nanometres without changing this coefficient produces a factor-$10^3$ error.

### 4.3 From $f$, $gf$, or $\log(gf)$

For the absorption oscillator strength $f_{lu}$,

$$
A_{ul}^{(0)}
=
\frac{2\pi e^2}{m_ec\epsilon_0\lambda_{\rm vac}^2}
\frac{w_l}{w_u}f_{lu},
\qquad
w_l=2J_l+1,\quad w_u=2J_u+1.
$$

Alternatively,

$$
\boxed{
\mathscr S_{ul}^{(0)}[\mathrm{a.u.}]
=
\frac{3}{2}\frac{w_l f_{lu}}{\Delta E[\mathrm{Hartree}]}.
}
$$

For $x=\log_{10}(w_lf_{lu})$, first set $w_lf_{lu}=10^x$; do not use $w_u$ in its place. [R1, R4]

Restrict this conversion route to explicitly identified E1 data. Generalized oscillator-strength fields or blended multipole rates in a database require their documented definitions; they do not automatically define an electric dipole amplitude.

### 4.4 From a lifetime and branching fraction

For a radiative lifetime $\tau_u$ and an identified radiative branch fraction $b_{ul}$,

$$
A_{ul}=\frac{b_{ul}}{\tau_u}.
$$

A lifetime alone fixes a sum of rates, not one branch. Verify that the quoted lifetime is radiative rather than collisionally quenched, and that autoionization or other nonradiative loss is not included in the inferred radiative sum. Where several multipoles contribute to one branch, separate the E1 contribution before this guide's E1 inversion is used.

### 4.5 From a multiplet strength or multiplet-averaged rate

For one complete pure-LS orbital channel, define

$$
\mathscr S_{\rm mult}
=\sum_{J_u,J_l}\mathscr S_{J_uJ_l}^{(0)}.
$$

With the reduced-element convention of Section 3,

$$
\boxed{\mathscr S_{\rm mult}=(2S+1)|R|^2.}
$$

A tabulated multiplet quantity is usable only after its averaging convention is established. For example, a statistically averaged multiplet rate would be

$$
\overline A
=\frac{\sum_{J_u}(2J_u+1)\sum_{J_l}A_{J_uJ_l}}
{(2L_u+1)(2S+1)}.
$$

Converting this single number to $\mathscr S_{\rm mult}$ using a common frequency is an approximation unless the source explicitly defines an effective frequency and normalization. Prefer summing individually converted line strengths when component frequencies are available.

For a multiplet absorption oscillator strength defined with lower-term weight $w_{l,\rm term}=(2L_l+1)(2S+1)$, the common-frequency approximation gives

$$
\mathscr S_{\rm mult}[\mathrm{a.u.}]
\simeq\frac{3}{2}
\frac{w_{l,\rm term}f_{\rm mult}}{\Delta E_{\rm representative}[\mathrm{Hartree}]}.
$$

Never substitute a term weight for a level weight solely because a line is experimentally unresolved.

---

## 5. Model A: isolated-$J$ Zeeman components from individual line data

### 5.1 Applicability

Use Model A when each participating electronic level remains effectively isolated from magnetic mixing with other levels. Its internal LS composition need not be known. Hyperfine structure must either be demonstrably negligible for the observable or treated using Section 9.

Required inputs for one E1 line are $J_u,J_l$, opposite parities, energies, a zero-field line strength or rate, and reliable $g_u,g_l$ values.

An empirical $g_J$ is preferred when available. The LS Landé formula is a fallback only when the LS assumption is supported. Knowledge of a diagonal $g_J$ does not supply off-diagonal magnetic matrix elements.

### 5.2 Energies and component frequencies

For each $M=-J,\ldots,J$,

$$
E_{JM}(B)=E_J^{(0)}+\mu_BBg_JM.
$$

For $q=M_u-M_l\in\{-1,0,1\}$,

$$
\nu_{ul}(B)=\nu_0+
\frac{\mu_BB}{h}(g_uM_u-g_lM_l).
$$

This is a linear magnetic approximation. Small but resolvable quadratic shifts can require a more complete energy model even before intensity changes are important.

### 5.3 Component strengths and rates

The strength between specified magnetic states is

$$
\boxed{
s_{J_uM_u\to J_lM_l}
=
\mathscr S_{ul}^{(0)}
\begin{pmatrix}J_u&1&J_l\\-M_u&q&M_l\end{pmatrix}^{\!2}.
}
$$

No LS $6j$ factor is required because the individual reduced line strength is already supplied.

The corresponding rate is

$$
\boxed{
A_{J_uM_u\to J_lM_l}(B)
=
A_{ul}^{(0)}
\left(\frac{\nu_{ul}(B)}{\nu_0}\right)^3
(2J_u+1)
\begin{pmatrix}J_u&1&J_l\\-M_u&q&M_l\end{pmatrix}^{\!2}.
}
$$

At zero field, for **each** upper $M_u$,

$$
\sum_{M_l,q}A_{J_uM_u\to J_lM_l}(0)=A_{ul}^{(0)}.
$$

This shows why one must not divide each component rate by $2J_u+1$ again. The statistical averaging already entered when the reduced strength was inferred from the tabulated level-to-level rate.

### 5.4 When this is the best high-$Z$ starting point

For a strongly mixed high-$Z$ level that nevertheless has a well-isolated $J$, an independent measured line rate and measured $g_J$ can be more defensible than forcing the line into a pure-LS Paschen–Back model. The criterion is isolation and accuracy of the inputs, not a chosen atomic-number cutoff.

If mixing becomes significant anywhere between 0.1 and 10 T, upgrade the whole affected connected level set to Model B or C. Do not independently split lines whose parent states are actually mixing with one another.

---

## 6. Model B: isolated-LS-term Zeeman–Paschen–Back calculation

### 6.1 Physical assumptions and basis completeness

Retain a complete upper term $\gamma_uL_uS$ and a complete lower term $\gamma_lL_lS$. The same spin $S$ is required for this pure-LS E1 construction, together with opposite parity and the orbital triangle rule. The case $L_u=L_l=0$ is forbidden for E1.

For each term include every

$$
J=|L-S|,|L-S|+1,\ldots,L+S,
$$

and every $M=-J,\ldots,J$. Missing $J$ energies mean the Model B input is incomplete. An interpolated or theoretical replacement can be used only when explicitly identified and assigned an uncertainty.

Here $M=m_L+m_S$ is conserved because the retained Hamiltonian is axially symmetric. It is not conserved in the same way after adding transverse electric fields, and electronic $M$ alone is generally not conserved after adding hyperfine interactions.

The diagonalization handles arbitrary ratios of the retained linear magnetic interaction to the retained fine-structure splitting. Its validity still depends on the LS wavefunctions, orbital-channel factorization, and neglect of additional interactions. This restricted fixed-$M$ approach is related to the formulation described in the ADAS603 manual, without asserting identical implementation or scope. [R6]

### 6.2 Preferred Hamiltonian construction: use Clebsch–Gordan transformations

For each $M$, form a matrix $U^{(M)}$ whose rows are uncoupled states $(m_L,m_S)$ with $m_L+m_S=M$, and whose columns are coupled states $J$ with $J\geq|M|$:

$$
U^{(M)}_{(m_L,m_S),J}
=\langle Lm_L,Sm_S|JM\rangle.
$$

With the convention in Section 3, $U$ is real and orthogonal. Construct

$$
T_{\rm unc}^{(M)}
=\operatorname{diag}(m_L+g_sm_S).
$$

Then the Hamiltonian in the coupled basis is

$$
\boxed{
H_{\rm coupled}^{(M)}(B)
=\operatorname{diag}(E_0(J))
+\mu_BB\,U^{(M)\dagger}T_{\rm unc}^{(M)}U^{(M)}.
}
$$

An equally valid independent implementation uses the uncoupled basis:

$$
\boxed{
H_{\rm unc}^{(M)}(B)
=U^{(M)}\operatorname{diag}(E_0(J))U^{(M)\dagger}
+\mu_BB\,T_{\rm unc}^{(M)}.
}
$$

These two implementations must yield identical energies and transition strengths after the dipoles are transformed consistently. This is the preferred protection against phase-convention mistakes.

### 6.3 Equivalent analytic matrix entries

For $J>0$,

$$
H_{JJ}^{(M)}
=E_0(J)+\mu_BBg_JM,
$$

$$
g_J
=1+(g_s-1)
\frac{J(J+1)-L(L+1)+S(S+1)}{2J(J+1)}.
$$

For $J=0$, the diagonal magnetic term is zero; **do not evaluate the singular Landé expression**.

With the orbital-first Condon–Shortley convention used here,

$$
\boxed{
H_{J-1,J}^{(M)}
=-\mu_BB(g_s-1)
\sqrt{
\frac{(J^2-M^2)\,[J^2-(L-S)^2]\,[(L+S+1)^2-J^2]}
{4J^2(4J^2-1)}
}.
}
$$

The transpose entry is the same, and $|\Delta J|>1$ entries vanish. Evaluate an off-diagonal entry only when both basis states actually exist. This avoids spurious singular cases at the ends of a term.

The minus sign is tied to the stated basis phases, not an independently observable sign. A positive off-diagonal convention is permissible only after the relevant basis rephasing is also applied to the dipole matrices. Combining the positive sign with the unchanged dipole recoupling formula below gives incorrect component strengths, even though the energy eigenvalues and some strength sums can still agree.

Do not replace the diagonal LS $g_J$ values by unrelated experimental values while silently keeping LS off-diagonals and calling the result exact. Such a hybrid may be a useful phenomenological model, but it needs its own validation and metadata.

### 6.4 Extract one reduced orbital norm per channel

Define

$$
R=\langle\gamma_uL_u\|\mathbf D\|\gamma_lL_l\rangle.
$$

Within this model,

$$
\boxed{
r_{J_uJ_l}
=(-1)^{L_u+S+J_l+1}
\sqrt{(2J_u+1)(2J_l+1)}
\begin{Bmatrix}L_u&J_u&S\\J_l&L_l&1\end{Bmatrix}R.
}
$$

Define

$$
\mathcal F_{J_uJ_l}
=(2J_u+1)(2J_l+1)
\begin{Bmatrix}L_u&J_u&S\\J_l&L_l&1\end{Bmatrix}^{\!2}.
$$

For a valid reference branch with $\mathcal F_{\rm ref}>0$,

$$
\boxed{
|R|^2=\frac{\mathscr S_{\rm ref}^{(0)}}{\mathcal F_{\rm ref}}.
}
$$

One may choose $R$ real and positive for an isolated channel because its common phase cancels from its strengths. This does not provide relative phases among several channels that will later interfere in Model C.

Use a well-determined reference rate; a very weak or cancellation-sensitive line is usually a poor sole normalization. A single reference fits one norm but does not establish the LS approximation.

### 6.5 Use all available branches to test, not merely fit, the model

Convert each independently available branch into $\mathscr S_i^{(0)}$ and compute

$$
x_i=\frac{\mathscr S_i^{(0)}}{\mathcal F_i},
\qquad \mathcal F_i>0.
$$

Strict Model B predicts a common $x_i=|R|^2$. The equivalent rate test is constancy of

$$
Q_i=\frac{A_i^{(0)}(2J_{u,i}+1)}{\nu_i^3\mathcal F_i}.
$$

Disagreement beyond credible uncertainties indicates that the one-norm model does not reproduce those data. Possible causes include mixed atomic coupling, channel-dependent structure, data errors, wrong assignments, or inconsistent units. Agreement is supportive, not conclusive—especially when the tabulated branches were themselves generated by LS splitting. [R4]

For independent line-strength errors, a weighted estimate is

$$
\widehat x
=\frac{\sum_i\mathcal F_i\mathscr S_i^{(0)}/\sigma_i^2}
{\sum_i\mathcal F_i^2/\sigma_i^2}.
$$

With a reliable covariance matrix $\Sigma$, use

$$
\widehat x
=\frac{\mathcal F^{\mathsf T}\Sigma^{-1}\mathscr S}
{\mathcal F^{\mathsf T}\Sigma^{-1}\mathcal F}.
$$

Do not treat shared systematic uncertainties or LS-generated branch values as independent measurements. Do not rescale every branch separately and continue to apply strict-LS invariants without qualification; see Section 7.5.

### 6.6 Construct the dipole matrices and diagonalize

In the coupled basis, for $q=M_u-M_l$,

$$
(D_q^{(0)})_{J_uJ_l}
=(-1)^{J_u-M_u}
\begin{pmatrix}J_u&1&J_l\\-M_u&q&M_l\end{pmatrix}
r_{J_uJ_l}.
$$

For an independent uncoupled construction,

$$
\boxed{
\langle L_um_{L,u},Sm_{S,u}|D_q|
L_lm_{L,l},Sm_{S,l}\rangle
=
\delta_{m_{S,u},m_{S,l}}
(-1)^{L_u-m_{L,u}}
\begin{pmatrix}L_u&1&L_l\\-m_{L,u}&q&m_{L,l}\end{pmatrix}R.
}
$$

Diagonalize each Hamiltonian block and use $C_u^\dagger D_q^{(0)}C_l$. Equivalently, factor out $R$ and compute

$$
\begin{aligned}
\mathcal G_{ul}(B)
={}&\sum_{J_u,J_l}C_{J_u,u}^{*}C_{J_l,l}
(-1)^{J_u-M_u+L_u+S+J_l+1}
\sqrt{(2J_u+1)(2J_l+1)}\\
&\times
\begin{pmatrix}J_u&1&J_l\\-M_u&q&M_l\end{pmatrix}
\begin{Bmatrix}L_u&J_u&S\\J_l&L_l&1\end{Bmatrix}.
\end{aligned}
$$

Then

$$
s_{ul}(B)=|R|^2|\mathcal G_{ul}(B)|^2.
$$

Always sum amplitudes before taking the squared magnitude. Summing $|C|^2$-weighted branch strengths drops interference terms and is not equivalent.

### 6.7 Direct rate formula using the reference line

The constants and reduced norm cancel to give

$$
\boxed{
A_{ul}(B)
=A_{\rm ref}
\left[\frac{\nu_{ul}(B)}{\nu_{\rm ref}}\right]^3
\frac{(2J_{u,\rm ref}+1)|\mathcal G_{ul}(B)|^2}
{\mathcal F_{\rm ref}}.
}
$$

This formula gives the rate from **one specified upper magnetic eigenstate** into **one specified lower magnetic eigenstate**, integrated over emission direction. It is not a population-weighted intensity.

### 6.8 Multiple orbital channels and hydrogenic features

One reduced orbital norm is required for each independent $\gamma_uL_u\to\gamma_lL_l$ channel. A complete spectral feature can contain several such channels.

For example, applying E1 orbital selection rules to hydrogenic $n=3\to2$ gives

$$
3s\to2p,\qquad3p\to2s,\qquad3d\to2p.
$$

The three channels do not share one norm merely because they contribute to the same Balmer feature. Retrieve separate resolved data, or externally supplied hydrogenic matrix elements, for each channel. Sum intensities over distinct initial/final states after accounting for their populations. If added interactions mix these channels coherently, the needed relative matrix-element phases must also be supplied or derived consistently.

### 6.9 Correct Paschen–Back limit

When $\mu_BB$ dominates the retained fine-structure intervals, the eigenvectors approach uncoupled states and the E1 amplitudes approach

$$
\Delta m_S=0,\qquad\Delta m_L=q=0,\pm1.
$$

However, neglecting off-diagonal fine-structure mixing at large $B$ does not eliminate the diagonal fine-structure expectation value:

$$
E(m_L,m_S;B)
\simeq\mu_BB(m_L+g_sm_S)
+\sum_J|\langle Lm_L,Sm_S|JM\rangle|^2E_0(J).
$$

Thus the magnetic slopes approach three transition groups, but residual internal frequency structure can remain within those groups. An exact three-frequency pattern requires extra approximations or unresolved residual splitting.

For a simple $H_{\rm fs}=\zeta\mathbf L\cdot\mathbf S/\hbar^2$ model, the residual diagonal energy is $\zeta m_Lm_S$. A $^2P\to{}^2S$ line then has two spin-conserving $\pi$ transitions at one frequency and two frequencies within each $\sigma$ group: generally five limiting frequencies, not three.

This is a formal asymptote of the truncated linear-magnetic model. Diamagnetic and other omitted effects must be checked before interpreting an arbitrarily large numerical field physically.

---

## 7. Model C: general magnetic mixing using external reduced matrix elements

### 7.1 Use exact zero-field electronic levels as the basis

Write

$$
|aJ_aM\rangle,
$$

where $a$ identifies a zero-field electronic eigenlevel. It may contain arbitrary LS, configuration, or relativistic mixing. Its $L,S$ label is not used to reconstruct unknown matrix elements.

Define a dimensionless electronic magnetic tensor by

$$
H_{\rm mag}=\mu_BB\,T_0^{(1)},
\qquad
\mathbf T^{(1)}=-\boldsymbol\mu_{\rm electronic}/\mu_B.
$$

The required reduced inputs are

$$
t_{ab}=\langle aJ_a\|\mathbf T^{(1)}\|bJ_b\rangle,
\qquad
r_{ab}=\langle aJ_a\|\mathbf D\|bJ_b\rangle.
$$

They must share level identities, normalization conventions, and state phases. A physical magnetic-moment matrix in $\mathrm{J/T}$ must be converted with the minus sign and the factor $\mu_B$ in the definition above.

Model C is the general finite-basis version of the magnetic-state expansion used in structure-based Zeeman calculations. External producers can compute the radial/relativistic physics once; the downstream field sweep then operates only on level lists and matrices. [R7]

### 7.2 Magnetic Hamiltonian

For each parity and conserved $M$, build

$$
\boxed{
H_{ab}^{(M)}(B)
=E_a^{(0)}\delta_{ab}
+\mu_BB(-1)^{J_a-M}
\begin{pmatrix}J_a&1&J_b\\-M&0&M\end{pmatrix}t_{ab}.
}
$$

The electronic magnetic interaction preserves parity. It may connect different zero-field levels with $\Delta J=0,\pm1$, except $J_a=J_b=0$; the full matrix is not generally tridiagonal because several states can share a given $J$.

Useful consistency relations are

$$
t_{ba}=(-1)^{J_b-J_a}t_{ab}^{*},
$$

and, for $J_a>0$,

$$
\boxed{
t_{aa}=g_a\sqrt{J_a(J_a+1)(2J_a+1)}.
}
$$

The latter recovers the diagonal shift $\mu_BBg_aM$ and tests sign/normalization conventions. It does not determine $t_{ab}$ for $a\ne b$.

A tabulated radiative M1 rate can sometimes constrain the magnitude of a related magnetic-moment matrix element in the appropriate long-wavelength convention. It does not supply its sign, all diagonal moments, or a complete static interaction matrix. Do not use radiative M1 tables as a drop-in replacement without deriving the operator conversion.

### 7.3 Dipole matrices and basis selection

Construct the E1 matrices using the Wigner–Eckart convention in Section 3 and the supplied $r_{ab}$. Include all same-parity states needed for magnetic mixing, not merely levels whose lines lie inside the final wavelength window. Include all dipole links between the retained manifolds that can contribute to the requested transitions.

Use exact zero-field selection rules to identify structural zeros. A missing database entry for a transition that is not symmetry-forbidden is **unknown**, not automatically zero.

After diagonalization, the E1 selection rule is on field-dressed projections, $|M_u-M_l|\leq1$, and opposite parity. Zero-field $J$ labels no longer provide an exact transition rule. The pure-LS rule $\Delta S=0$ is also not a general constraint on mixed eigenlevels.

### 7.4 Why magnitudes alone can be insufficient

Suppose a dressed upper state has the form

$$
|u\rangle=\cos\theta\,|a\rangle+\sin\theta\,|b\rangle.
$$

Its amplitude to one lower state is

$$
d_{ul}=\cos\theta\,d_{al}+\sin\theta\,d_{bl}.
$$

The same two zero-field strengths $|d_{al}|^2$ and $|d_{bl}|^2$ allow constructive or destructive interference, depending on the phase relation to the mixing Hamiltonian. Those alternatives can produce very different weak components.

Without phase information, the elementary bounds are

$$
\left(|\cos\theta\,d_{al}|-|\sin\theta\,d_{bl}|\right)^2
\leq |d_{ul}|^2
\leq
\left(|\cos\theta\,d_{al}|+|\sin\theta\,d_{bl}|\right)^2.
$$

Such bounds can be useful diagnostics, but arbitrary phases may not satisfy all constraints of a multilevel atomic model. They are not a replacement for a phase-consistent dataset.

If only one path contributes, or all relevant paths have independently known phase relations, the missing-phase problem can disappear for that particular observable. The general limitation should not be interpreted as a claim that every rate-only problem is insoluble.

### 7.5 Optional semiempirical amplitude calibration

A consciously approximate extension is to take relative signs from a trusted LS or structure calculation while replacing magnitudes using independent measured strengths:

$$
r_{ab}^{\rm calibrated}
=e^{i\arg r_{ab}^{\rm model}}
\sqrt{\mathscr S_{ab}^{\rm measured}}.
$$

Likewise, a producer's phase-consistent theoretical matrices can be combined with improved experimental diagonal energies. These steps may improve selected observables but do not reconstruct unknown wavefunctions or prove the phases/magnetic couplings correct.

Label the phase source as `assumed_from_model`, retain the original matrices, and quantify sensitivity. Do not silently use this option when strict LS branch ratios fail. A calibrated dipole matrix need not satisfy the strict-LS per-upper-state norm in Section 13, although its global trace norm is still invariant under unitary rotations of complete fixed bases.

When several orbital channels are present, relative channel phases must come from a consistent model or experiment; they cannot all be inferred from independent $A$-values.


### 7.6 Alternative external input: signed atomic mixing coefficients

A producer may supply zero-field expansion coefficients rather than final operator matrices:

$$
|aJM\rangle=\sum_{\beta}W_{\beta J,a}\,|\beta LSJM\rangle,
$$

where $\beta$ also identifies configuration/parentage and the underlying LS basis is shared and orthonormal. These are coefficients of zero-field atomic-structure mixing, not the field-dependent eigenvectors $C$ calculated later. NIST's eigenvector-composition discussion describes this distinction; tabulated percentages are squared coefficient magnitudes, not signed amplitudes. [R3]

In the nonrelativistic magnetic approximation, the total generators $\mathbf L$ and $\mathbf S$ preserve the LS term and its orthonormal multiplicity label. Build their LS-basis matrices using Section 6 for every retained term, then transform:

$$
T_0^{\rm electronic}=W^\dagger T_0^{\rm LS}W.
$$

This can provide off-diagonal magnetic couplings between mixed zero-field levels without evaluating radial functions locally. It still assumes the magnetic operator $\mathbf L+g_s\mathbf S$ is adequate; relativistic corrections may require external operator matrices instead.

For the dipole, construct an LS-basis matrix using one reduced orbital amplitude for **each** required channel, then use

$$
D_q^{\rm electronic}=W_u^\dagger D_q^{\rm LS}W_l.
$$

The channel amplitudes and their relative phases remain required inputs. Atomic mixing coefficients alone do not determine those dipole norms. Coefficients from separately optimized, nonorthogonal orbital bases require the producer's overlaps/operator transformation, not this simple shared-basis formula.

Composition percentages or a list of only the largest components generally cannot replace $W$. Arbitrarily assigning positive square roots or renormalizing a truncated composition list can give wrong interference and magnetic couplings. Require a consistent full expansion, verify $W^\dagger W=I$ within the represented space, and test convergence when components are omitted.

---

## 8. Common calculation of amplitudes, strengths, and Einstein coefficients

### 8.1 Field-dressed transition amplitudes

For arbitrary retained electronic basis states,

$$
\boxed{
d_{ul}^{(q)}(B)
=\sum_{a,b}C_{au}^{*}(B)
\langle a|D_q|b\rangle
C_{bl}(B).
}
$$

Define

$$
\boxed{s_{ul}(B)=\sum_{q=-1}^{1}|d_{ul}^{(q)}(B)|^2.}
$$

For states of definite $M$, only $q=M_u-M_l$ can be nonzero. With transverse electric fields or another symmetry-breaking interaction, several $q$ values can contribute to one pair; retain the vector amplitudes rather than forcing a single $\pi/\sigma$ label.

Interference among basis-state contributions occurs **inside** each amplitude. The three orthogonal spherical components are summed as squared magnitudes for the total angle-integrated E1 rate.

### 8.2 Frequencies

For the actual higher-energy state $u$ and lower-energy state $l$,

$$
\nu_{ul}(B)=\frac{E_u(B)-E_l(B)}{h}>0,
\qquad
\lambda_{ul,\rm vac}(B)=\frac{c}{\nu_{ul}(B)}.
$$

If a preassigned pair has a negative energy difference, it is not a downward emission transition in that ordering. Reorder the physical states and matrix elements if appropriate, or exclude the pair. A negative computed frequency is not itself a classification of the physical process as absorption.

All electronic blocks must share a consistent energy reference. Subtracting a convenient scalar before diagonalization is recommended for conditioning, but restore it before forming interblock transition frequencies.

### 8.3 Absolute component $A$-values

For SI E1 strengths,

$$
\boxed{
A_{ul}(B)
=\frac{\omega_{ul}^3}{3\pi\epsilon_0\hbar c^3}\,s_{ul}(B)
=\frac{16\pi^3\nu_{ul}^3}{3\epsilon_0hc^3}\,s_{ul}(B).
}
$$

Equivalently,

$$
\boxed{
A_{ul}(B)[\mathrm{s}^{-1}]
=\frac{C_{\rm E1}}{\lambda_{ul,\rm vac}[\text{\AA}]^3}
\,s_{ul}(B)[\mathrm{a.u.}].
}
$$

There is **no further statistical-weight division** for a specified initial magnetic state. This is true even if another state happens to have the same energy: it follows from specifying the initial state, not from claiming that a field removes every degeneracy.

The total radiative decay rate is

$$
\Gamma_u^{\rm rad}(B)=\sum_l A_{ul}^{\rm E1}(B)+\sum_l A_{ul}^{\rm other\ multipoles}(B),
\qquad
\tau_u^{\rm rad}=1/\Gamma_u^{\rm rad}.
$$

A calculation covering only one lower term or one spectral window yields a **partial** rate, not necessarily a lifetime.

### 8.4 Consistent use of updated transition energies

A reduced matrix element is the structural input; the rate also contains a frequency factor. If theoretical rates are supplied at theoretical frequencies, first invert them with those same frequencies. Then use the desired experimental or field-dressed frequency in the rate formula.

Do not invert a theoretical rate with an unrelated experimental frequency and also apply a separate frequency-rescaling correction. That can apply the correction twice. For exact zero-field recovery tests, either use the original source frequency or compare with the appropriately rescaled reference rate.

For optical transitions the magnetic change in $\nu^3$ may be small, but retaining it costs little. For closely spaced infrared or microwave transitions, inspect it explicitly.

### 8.5 Optional Einstein $B$-coefficients

The symbol $B$ in this subsection denotes an Einstein coefficient, not the magnetic field. Specify the radiation-density convention before exporting it.

For isotropic, unpolarized radiation with spectral energy density $u_\nu$ **per unit ordinary frequency**, define the stimulated rate as $B_{ul}^{(\nu)}u_\nu(\nu_{ul})$. Then

$$
\boxed{
B_{ul}^{(\nu)}=\frac{c^3}{8\pi h\nu_{ul}^3}A_{ul}.
}
$$

For specified magnetic states, $B_{lu}^{(\nu)}=B_{ul}^{(\nu)}$. For level-averaged zero-field coefficients,

$$
w_lB_{lu}^{(\nu)}=w_uB_{ul}^{(\nu)}.
$$

A coefficient defined for energy density per angular frequency differs by a factor $2\pi$. Polarized directional pumping requires the corresponding dipole/polarization contraction, not a scalar isotropic Einstein coefficient alone.

### 8.6 Non-E1 lines, especially for heavier species

Do not use an E1 prefactor or E1 units for M1, E2, M2, or mixed-multipole data. Nevertheless, the isolated-$J$ angular construction generalizes directly.

For one identified multipole of rank $k$, with a fixed long-wavelength multipole operator and known zero-field rate,

$$
\boxed{
A_{J_uM_u\to J_lM_l}^{(k)}(B)
=A_{ul}^{(k)}(0)
\left(\frac{\nu_{ul}(B)}{\nu_0}\right)^{2k+1}
(2J_u+1)
\begin{pmatrix}J_u&k&J_l\\-M_u&q&M_l\end{pmatrix}^{\!2}.
}
$$

Here $q=M_u-M_l$ can range from $-k$ to $k$. Electric rank-$k$ transitions require parity product $(-1)^k$; magnetic rank-$k$ transitions require $(-1)^{k+1}$. [R1]

For general magnetic mixing, contract the appropriate phase-consistent rank-$k$ transition matrices with the same eigenvectors. Obtain that multipole's absolute normalization from its external matrix convention or an identified reference rate. The E1 orbital $6j$ normalization is not a universal normalization for other operators.

If only a combined `M1+E2` rate is given, the separate component patterns cannot in general be recovered without the separate contributions. E1/M1 share a $\nu^3$ long-wavelength dependence; E2/M2 have $\nu^5$. Finite-wavelength relativistic operator definitions need the producer's own conversion rather than an unexamined power-law rescaling.

---

## 9. Hyperfine structure and isotopes

### 9.1 Decide using the isotope and observable

A nonzero nuclear spin introduces hyperfine structure and nuclear magnetic shifts. Over 0.1–10 T, some levels can be in a hyperfine Paschen–Back regime while their electronic $J$ levels remain isolated. This is distinct from fine-structure Paschen–Back mixing. An unresolved hyperfine pattern is not automatically irrelevant: it can still affect component strengths or precision centroids. [R9]

Specify the isotope, nuclear spin $I$, nuclear magnetic moment, and any needed hyperfine constants. For a natural isotopic mixture, compute separate isotope spectra and sum them with the appropriate abundances and isotope-specific populations; do not add isotope amplitudes coherently.

### 9.2 Data-only hyperfine extension for an isolated electronic $J$

This extension requires no additional electronic radial dipole integral beyond the electronic line data in Model A. Work in

$$
|Jm_J\rangle|Im_I\rangle,
\qquad M_F=m_J+m_I.
$$

For a given electronic level, obtain zero-field hyperfine energies $E_F^{(0)}$ directly from experiment or from hyperfine constants. For the magnetic-dipole and electric-quadrupole terms,

$$
K=F(F+1)-I(I+1)-J(J+1),
$$

$$
E_F^{(0)}-E_J^{(0)}
=\frac{A_{\rm hfs}}{2}K
+B_{\rm hfs}
\frac{\tfrac34K(K+1)-I(I+1)J(J+1)}
{2I(2I-1)J(2J-1)}.
$$

Omit the quadrupole term unless $I\geq1$ and $J\geq1$; never evaluate its singular denominator otherwise. The constants in this equation have energy units. A source quoting $A_{\rm hfs}/h$ or $B_{\rm hfs}/h$ in MHz requires multiplication by $h\times10^6$, not by $\hbar\times10^6$. Higher hyperfine multipoles, when needed, require additional data. [R9]

For each $M_F$, form

$$
V_{(m_J,m_I),F}=\langle Jm_J,Im_I|FM_F\rangle.
$$

Define $g_I$ by the physical nuclear moment

$$
\boldsymbol\mu_I=g_I\mu_N\mathbf I/\hbar.
$$

Then construct

$$
\boxed{
H^{(M_F)}
=V\operatorname{diag}(E_F^{(0)})V^\dagger
+\operatorname{diag}\!\left[
B(\mu_Bg_Jm_J-\mu_Ng_Im_I)
\right].
}
$$

Some sources quote nuclear factors in Bohr magnetons or with a Hamiltonian sign already included. Convert them to this definition before use.

The E1 operator is electronic and leaves the nuclear projection unchanged:

$$
\langle J_um_{J,u},Im_{I,u}|D_q|
J_lm_{J,l},Im_{I,l}\rangle
=
\delta_{m_{I,u},m_{I,l}}
\langle J_um_{J,u}|D_q|J_lm_{J,l}\rangle.
$$

Diagonalize and use the same matrix contraction and rate formula as before. Now the exact axial selection rule is $q=M_{F,u}-M_{F,l}$.

When a hyperfine-resolved reference amplitude is needed, the electronic-first coupling convention gives

$$
\langle (J_uI)F_u\|D\|(J_lI)F_l\rangle
=(-1)^{J_u+I+F_l+1}
\sqrt{(2F_u+1)(2F_l+1)}
\begin{Bmatrix}J_u&F_u&I\\F_l&J_l&1\end{Bmatrix}
\langle J_u\|D\|J_l\rangle.
$$

A reference rate between hyperfine levels uses upper statistical weight $2F_u+1$, not $2J_u+1$.

### 9.3 Simultaneous fine-structure and hyperfine mixing

When electronic $J$ mixing is significant, include multiple electronic levels and their nuclear-spin products. Diagonal hyperfine constants for individual $J$ levels do not determine all off-diagonal hyperfine couplings between different electronic levels.

Use externally supplied reduced electronic hyperfine tensors and nuclear moments, or explicitly identify any neglected off-diagonal hyperfine elements. The general magnetic/hyperfine transition workflow is illustrated by HFSZEEMAN95. [R7]

---

## 10. Applicability at 0.1–10 T and extension to high $Z$

### 10.1 No universal atomic-number or density cutoff

Select the model for a specific ion, level set, isotope, field interval, and observable. “Low $Z$” does not guarantee negligible Stark or hyperfine effects, and “high $Z$” does not imply that 10 T strongly mixes every fine-structure level.

For each application record both a frequency tolerance and a strength/intensity tolerance. A model adequate for fitting a Doppler-broadened envelope can be inadequate for predicting a very weak satellite component.

### 10.2 Omitted-state mixing and basis convergence

For a retained state $a$ and omitted state $b$, a perturbative screen is

$$
\epsilon_{ab}(B)
=\frac{|\langle a|V_{\rm omitted}(B)|b\rangle|}
{|E_a(B)-E_b(B)|}.
$$

Also estimate omitted shifts,

$$
\delta E_a^{(2)}
\simeq\sum_{b\notin\mathcal B}
\frac{|\langle b|V|a\rangle|^2}{E_a-E_b},
$$

where $\mathcal B$ is the retained basis and perturbation theory is valid only away from small denominators. A small energy correction does not guarantee an accurate weak transition amplitude.

The preferred test is to enlarge the basis and compare the actual requested frequencies, component strengths, and final convolved spectrum. Use the complete 0.1–10 T interval, because relevant level separations can vary with field. If omitted matrix elements are unknown, label truncation as unassessed rather than reporting a numerical error bound without evidence.

### 10.3 Diamagnetism and two distinct meanings of “quadratic Zeeman”

The linear magnetic Hamiltonian already produces nonlinear, including quadratic, eigenenergy shifts through level mixing. These are present when its matrix is diagonalized.

The nonrelativistic fixed-nucleus Hamiltonian additionally contains the explicit diamagnetic term

$$
\boxed{
H_{\rm dia}=\frac{e^2B^2}{8m_e}\sum_k(x_k^2+y_k^2).
}
$$

That operator is absent from Models A–C as written. It can shift and mix same-parity orbital states. An order-of-magnitude conversion is

$$
\frac{\langle H_{\rm dia}\rangle}{hc}
\simeq
4.9655\times10^{-5}
\left(\frac{B}{10\ \mathrm T}\right)^2
\frac{\langle\sum_k r_{\perp,k}^2\rangle}{a_0^2}
\ \mathrm{cm}^{-1}.
$$

This coefficient is obtained directly from the displayed operator and SI constants; it is not a species-specific estimate. Large excited-state radii and small nearby level separations can amplify its importance. An atomic binding-energy comparison alone is insufficient.

To include it without local wavefunctions, obtain matrix elements of the quadratic magnetic operator, or a compatible effective Hamiltonian, from an external calculation or experiment. A measured quadratic-shift coefficient may already include virtual magnetic mixing represented in your retained basis. Do not add that full coefficient to a Hamiltonian that already includes the same contribution.

For a relativistic high-$Z$ calculation, use a consistently derived field-interaction model rather than appending this nonrelativistic expression indiscriminately.

### 10.4 Relativistic and intermediate-coupling requirements

For high-$Z$ or strongly mixed states, use reliable measured $g_J$ and line data in Model A, or relativistic phase-consistent matrices in Model C. A free-electron $g_s$ and pure-LS off-diagonal formula are not a general relativistic magnetic operator.

Close same-parity levels can still make laboratory fields important. Laboratory measurements of magnetic-field-induced emission in Fe X provide a concrete example of sensitivity associated with nearby fine-structure levels. [R10]

For an intended heavy-element line, the workflow is therefore: identify the actual zero-field levels and nearby couplings; establish the required isotope/multipole information; then choose A, B, or C. Do not select the coupling model from the element name alone.

At sufficiently short wavelengths, check the long-wavelength E1 approximation as well: $kr\ll1$, with $k=2\pi/\lambda$ and an appropriate electronic length scale. When a producer supplies finite-wavelength relativistic multipole amplitudes, retain its operator normalization and frequency dependence. An experimental rate can incorporate structural corrections that a pure-LS redistribution does not reproduce.

### 10.5 Static and motional electric fields

An electric field adds

$$
H_E=-\mathbf D\cdot\mathbf E
=-\sum_{q=-1}^{1}(-1)^q E_{-q}D_q.
$$

It couples opposite parity. A transverse component also breaks conservation of electronic $M$ about $\mathbf B$. The required dipole links include those between nearby levels mixed by the field, not merely the optical transition being observed.

For a slowly moving emitter, the leading rest-frame field includes

$$
\mathbf E_{\rm rest}\simeq\mathbf E_{\rm lab}+\mathbf v\times\mathbf B.
$$

Motional Stark effects can therefore depend on the emitter velocity distribution and viewing geometry. Magnetized-plasma line-shape calculations explicitly treat this contribution. [R11]

For specified static fields, the enlarged Hamiltonian can still be diagonalized with externally supplied matrices. For fluctuating fields, one needs a time-dependent or statistically averaged line-shape treatment with a justified dynamical approximation.

### 10.6 Laboratory-plasma broadening is a separate modeling layer

The output of this guide is initially an isolated-emitter component list. It does not, by itself, predict Stark broadening, collisional redistribution, radiation trapping, or non-LTE populations.

Adding the same zero-field Stark profile to every magnetic component is not generally equivalent to a combined Stark–Zeeman calculation. In particular, published Lyman-$\alpha$ calculations show different magnetic-field responses for the central and lateral components. [R12]

A practical hierarchy is:

1. **Magnetic component list plus instrumental/Doppler convolution:** appropriate when the neglected electric/collisional effects are small for the requested observable.
2. **Component-dependent widths/shifts:** an additional approximation requiring a broadening model applicable in the magnetic field.
3. **Joint Stark–Zeeman dynamics:** needed when microfields mix relevant states, alter polarization, or change the shape nonperturbatively.

No universal electron-density cutoff can be supplied without the species, states, temperatures, perturber composition, and accuracy target. External collision rates may support population modeling, but level-resolved rates do not automatically supply magnetic-sublevel redistribution or coherence relaxation.

The spontaneous-emission formulas also assume ordinary free-space radiative modes and suitably isolated atomic levels. Strongly dispersive electromagnetic environments, cavities, strong level dissolution, or strongly modified plasma atomic structure require further treatment.

---

## 11. From atomic components to laboratory-plasma spectra

### 11.1 Separate three quantities

Keep **atomic strength**, **transition rate**, and **observed intensity** distinct:

$$
s_{ul}=|d_{ul}|^2,
\qquad
A_{ul}\propto\nu_{ul}^3s_{ul},
\qquad
I_{ul}\propto n_u h\nu_{ul}A_{ul}
\times\text{geometry and transport factors}.
$$

The calculation of $A_{ul}$ does not require knowledge of $n_u$. Calculation of a plasma spectrum does.

If magnetic substates of one isolated level are statistically populated, $n_{JM}=n_J/(2J+1)$. Equal populations across all states of an LS term are a stronger assumption. Neither follows solely from the fact that a thermal energy scale exceeds the Zeeman splitting; excitation, cascades, depolarization, quenching, and transport must support the assumption. The ADAS603 formulation explicitly distinguishes such population assumptions from its atomic component calculation. [R6]

### 11.2 Viewing angle and polarization

Let $\theta$ be the angle between the line of sight and $\mathbf B$. For a single definite-$q$ E1 component, the normalized angular emission functions are

$$
W_\pi(\theta)=\frac{3}{8\pi}\sin^2\theta,
$$

$$
W_{\sigma^\pm}(\theta)=\frac{3}{16\pi}(1+\cos^2\theta),
\qquad
\int W_q\,d\Omega=1.
$$

Therefore $dA_{ul}/d\Omega=A_{ul}W_q$. The rate $A_{ul}$ itself is not viewing-angle dependent.

Along the field, $\pi$ emission vanishes and the two $\sigma$ components have circular polarizations. Perpendicular to the field, $\pi$ is linear along the projected field and $\sigma$ is linear perpendicular to it. At intermediate viewing angles the $\sigma$ polarization is generally elliptical. The names “right” and “left” require a stated propagation/observer convention; retain $q=M_u-M_l$ as the unambiguous computational label. [R6]

For a general mixed-polarization transition, use the physical emission vector

$$
\mathbf d_{lu}=\langle l|\mathbf D|u\rangle.
$$

Its spherical components obey $(d_{lu})_{-q}=(-1)^q[d_{ul}^{(q)}]^*$. For a transverse unit analyzer polarization $\boldsymbol\epsilon$,

$$
\boxed{
\frac{dA_{ul,\epsilon}}{d\Omega}
=\frac{\omega_{ul}^3}{8\pi^2\epsilon_0\hbar c^3}
|\boldsymbol\epsilon^{*}\cdot\mathbf d_{lu}|^2.
}
$$

Sum over two orthogonal transverse analyzers for unpolarized detection. This formula avoids ambiguous helicity labels and handles coherent combinations of different spherical components.

### 11.3 Optically thin component synthesis

For an incoherent upper-state population and a normalized frequency profile $\phi_{ul}$,

$$
\int\phi_{ul}(\nu)\,d\nu=1,
$$

$$
j_\nu(\hat{\mathbf n})
=\sum_{u,l}n_u h\nu_{ul}
\frac{dA_{ul}}{d\Omega}
\phi_{ul}(\nu-\nu_{ul}).
$$

Integrate this local emissivity along the sightline, using local field magnitude, direction, and populations. The expression is optically thin; self-absorption and polarized radiative transfer need a separate model.

When converting a normalized profile from frequency to wavelength, include the Jacobian:

$$
\phi_\lambda(\lambda)=\frac{c}{\lambda^2}\phi_\nu(c/\lambda).
$$

For coincident or instrumentally unresolved components, sum their population-weighted intensities, not arbitrary amplitudes associated with different orthogonal states. Instrumental non-resolution does not itself create quantum coherence.

If the excitation produces relevant coherence between upper states, replace the diagonal-population treatment by a density matrix and use the full transition amplitudes. A component list remains useful atomic input but is not then a complete polarization or line-shape solution.

---

## 12. Implementation data contract and algorithmic workflow

### 12.1 Suggested data structures

The following is a proposed software interface, not a database standard.

**Level table**

| Field | Meaning |
|---|---|
| `level_id`, `species_id`, `isotope_id` | Stable identifiers; species includes nuclear charge and ion charge |
| `energy_value`, `energy_unit`, `energy_reference_id` | Zero-field energy and shared reference |
| `energy_uncertainty`, `energy_source_id` | Uncertainty and provenance; preserve precise intervals where available |
| `two_J`, `parity` | Exact angular momentum and parity |
| `term_id`, `L`, `two_S`, `configuration`, `parentage` | Required physical assignments for Model B; metadata otherwise |
| `g_J`, `g_J_source_id`, `g_J_uncertainty` | Magnetic factor, not statistical weight |
| `two_I`, hyperfine/nuclear fields | Isotope-specific extension when used |

**Radiative-link table**

Store `upper_level_id`, `lower_level_id`, `multipole`, the original datum and units, its source frequency/wavelength, uncertainty, and provenance. Store the converted reduced strength separately. For signed matrices also store real/imaginary parts, `phase_convention_id`, `operator_convention_id`, and `matrix_dataset_id`.

A magnitude derived from a rate should carry `phase_known = false` unless Model B or another explicitly identified source determines its relative phase. Distinguish `symmetry_zero`, `known_nonzero`, and `unknown` entries.

**Magnetic-link table**

Store level IDs, complex $t_{ab}$ or an explicitly defined dimensional equivalent, units, phase convention, provenance, and uncertainty. A pair of diagonal $g$-factors is not a replacement for an unknown off-diagonal entry.

**Calculation request**

Specify the field grid, electronic model, isotope/hyperfine treatment, basis selection, omitted operators, requested frequency/strength tolerances, and optional viewing/population/profile model. Include $B=0$ as a verification point even when production output begins at 0.1 T.

### 12.2 End-to-end pseudocode

```text
load atomic levels, radiative records, and available magnetic/hyperfine data
validate IDs, parities, angular momenta, units, energy references, and multipoles
preserve original records and source provenance
convert each radiative datum into its correctly normalized reduced strength

determine the required basis from the desired transitions AND possible mixing
select Model A, B, or C for each magnetically connected level set

if Model A:
    require reliable g_J values and a justified isolation assessment
    form magnetic-sublevel basis states with no interlevel magnetic mixing
    construct each line's dipole magnitudes and angular matrices
    do not invent cross-line phase information

if Model B:
    require complete J-resolved terms and a defined orbital channel per term pair
    test independent zero-field branches against the common-norm assumption
    infer or fit each orbital reduced norm
    construct LS Hamiltonian and E1 matrices in ONE phase convention
    cross-check coupled and uncoupled constructions

if Model C:
    require sufficient phase-consistent dipole AND magnetic matrices
    reject unresolved matrix entries needed by the calculation
    construct zero-field operators in parity and M blocks

if hyperfine structure is included:
    enlarge the basis with nuclear spin
    use M_total = M_F rather than electronic M
    add required hyperfine and nuclear magnetic interactions

precompute all field-independent angular/operator matrices
verify Hermiticity, reduced-element conventions, and zero-field rates

for B in verification and production grid:
    construct each Hamiltonian in one consistent energy unit
    subtract a numerical reference energy if useful
    diagonalize each complete symmetry block
    restore reference energies
    track state labels by overlaps/subspaces, not by array positions alone

    for each required upper/lower block pair:
        for each relevant tensor component q:
            D_dressed[q] = C_upper.conjugate_transpose @ D0[q] @ C_lower

        for each candidate pair of dressed states:
            evaluate all amplitudes before thresholding
            compute strength = sum_q(abs(D_dressed[q])**2)
            determine the actual positive transition frequency
            compute the appropriate multipole rate and vacuum wavelength
            attach exact projection/parity labels and data/model flags

    evaluate complete-basis norm checks before spectral-window filtering
    retain or separately bound weak components discarded from final output

refine the B grid near avoided crossings or rapid strength changes
repeat with larger bases / alternative inputs to assess convergence

optionally add populations, direction/polarization response, and line profiles
export atomic components separately from synthetic spectral intensities
export provenance, uncertainties, assumptions, and validation results
```

The matrix form of this pseudocode is shared by all models, but Model A does not license arbitrary coherent combination of separately normalized lines. Its independent-line construction is sufficient precisely because interlevel mixing is being neglected.

### 12.3 Numerical implementation details

Use exact integer/half-integer arguments for $3j$, $6j$, and Clebsch–Gordan routines; cache their values. Use a Hermitian eigensolver and double precision as a starting point, with convergence checks for very weak amplitudes.

Do not derive $q$ from an eigenvector's largest basis component or from a row number. It is fixed by the exact block projections. At $B=0$, resolve rotational degeneracies using the known $J,M$ basis. At exact degeneracies, individual eigenvectors within a degenerate subspace are not unique; compare subspaces or physically specified ensembles.

State tracking across a field grid can maximize overlaps, with subspace tracking near degeneracy. This is label management for stationary eigenstates, not an assumption of dynamical adiabatic population transport. Physical transport through a field variation may need separate dynamics.

Do not hard-truncate weak amplitudes before normalization tests or before evaluating an intensity budget. Use both absolute and relative cutoffs, record discarded strength/rate, and never turn missing inputs into zeros for numerical convenience.

### 12.4 Output component schema

At minimum export:

```text
field_T
upper_state_id, lower_state_id
upper_energy, lower_energy, energy_unit
upper_two_Mtotal, lower_two_Mtotal, total_projection_definition
upper_parity, lower_parity
q                         # only a single label when axial symmetry permits
nu_Hz, lambda_vac_nm
strength_SI, strength_au   # include operator type and unit definition
A_s1, multipole
complex_dipole_components  # or a linked amplitude table when needed
model_id, basis_id, source_ids
uncertainty_or_sensitivity_bounds
validity_flags
```

Store eigenvectors, parent-level overlaps, and dominant $J$ or LS labels in a separate state table. Dominant labels are descriptive metadata in the mixed regime, not exact selection rules.

Useful flags include `LS_ASSUMED`, `PHASES_ASSUMED_FROM_MODEL`, `TRUNCATION_UNASSESSED`, `HYPERFINE_UNASSESSED`, `STARK_NOT_INCLUDED`, `PARTIAL_DECAY_SUM`, and `MISSING_REQUIRED_INPUT`. A successful eigensolver does not clear any physical-validity flag by itself.

---

## 13. Verification, invariants, and numerical benchmark

### 13.1 Structural and numerical checks

Require Hermitian Hamiltonians, unitary eigenvector matrices, small normalized eigenproblem residuals, and agreement under consistent basis transformations.

For Model B, compare the analytic magnetic entries with direct Clebsch–Gordan construction. Then compare the complete amplitudes computed in coupled and uncoupled bases. Agreement of energies alone cannot detect a Hamiltonian/dipole relative-phase error.

### 13.2 Zero-field checks

Recover the input $E_0(J)$ values. For isolated nondegenerate electronic levels, the initial slope is

$$
\left.\frac{dE_{JM}}{dB}\right|_{B=0}=\mu_Bg_JM.
$$

If different electronic levels are exactly degenerate and coupled, first apply degenerate perturbation theory; the simple parent-level slope need not identify the eigenbranch.

For every supplied E1 reference branch, verify

$$
\sum_{M_l,q}A_{J_uM_u\to J_lM_l}(0)=A_{J_u\to J_l}^{(0)}
$$

for every $M_u$, with consistent reference frequencies. In Model B, test **all available independent branches**, not only the normalization line. In the hyperfine extension, recover the zero-field hyperfine energies and expected electronic/nuclear angular recoupling.

### 13.3 Strict-LS strength sums

For one complete lower LS term and any individual upper eigenstate in strict Model B,

$$
\boxed{\sum_{l,q}s_{ul}(B)=\frac{|R|^2}{2L_u+1}.}
$$

Equivalently,

$$
\sum_{l,q}|\mathcal G_{ul}(B)|^2=\frac{1}{2L_u+1}.
$$

Summing over all upper states gives

$$
\boxed{
\sum_{u,l,q}s_{ul}(B)
=(2S+1)|R|^2=\mathscr S_{\rm mult}.
}
$$

These identities require a complete term basis and the single-channel LS factorization. They are not universal per-state identities for arbitrarily calibrated or mixed configurations.

### 13.4 General unitary norm invariant

For any fixed complete upper and lower bases,

$$
\boxed{
\sum_q\|C_u^\dagger D_qC_l\|_F^2
=\sum_q\|D_q\|_F^2.
}
$$

The Frobenius norm is the sum of squared absolute matrix entries. This global invariant holds in Model C as well, but individual upper-state sums can change.

Evaluate the mathematical full-basis invariant before pruning wavelengths, weak transitions, or states. If an energy-order filter removes matrix entries after a crossing, the remaining emission subset is not necessarily an invariant sum.

A strength invariant does **not** imply an exact rate invariant, because each rate contains its own $\nu_{ul}^3$. Do not renormalize field-dependent rates to force an artificial constant lifetime.

### 13.5 Phase-invariance test

Choose arbitrary diagonal unitary phase matrices $P_u,P_l$. Transform

$$
H_u\to P_u^\dagger H_uP_u,
\quad H_l\to P_l^\dagger H_lP_l,
\quad D_q\to P_u^\dagger D_qP_l,
$$

and transform the eigenvectors consistently. Energies and strengths must be unchanged. Changing only the Hamiltonian or only the dipole phases is not this test.

### 13.6 Paschen–Back tests

Within strict Model B, verify convergence toward $\Delta m_S=0$ and $\Delta m_L=q$ in the formal strong-field limit. Check vanishing spin-changing amplitudes, not merely three visually separated line groups. Retain the residual diagonal fine structure described in Section 6.9.

For a $^2P\to{}^2S$ channel, consider the limiting upper state $|m_L=0,m_S=+1/2\rangle$. In units of $|R|^2$, its strength to lower $m_S=+1/2$ is $1/3$ with $q=0$, and its strength to lower $m_S=-1/2$ is zero. Combining the opposite Hamiltonian phase with the unchanged dipoles can instead give $1/27$ and $8/27$—a wrong result whose sum still equals $1/3$.

### 13.7 Reproducible synthetic numerical benchmark

**These numbers describe an artificial atom, not an identified element or ion.** They test normalization, signs, field mixing, and conversion to rates without relying on uncertain real-atom data.

Use one $^2P\to{}^2S$ orbital channel with

$$
L_u=1,\quad L_l=0,\quad S=\tfrac12,
$$

$$
E_0(^2S_{1/2})=0,
\quad E_0(^2P_{1/2})=20000\ \mathrm{cm}^{-1},
\quad E_0(^2P_{3/2})=20001\ \mathrm{cm}^{-1}.
$$

Take

$$
A_{\rm ref}(^2P_{1/2}\to{}^2S_{1/2})=10^8\ \mathrm{s}^{-1},
\qquad \lambda_{\rm ref}=5000\ \text{\AA}.
$$

For reproducibility, freeze the following numerical constants in this test only:

```text
g_s = 2.00231930436
mu_B_in_wavenumber_units = 0.46686447719298224 cm^-1 T^-1
C_E1 = 2.02612688451534e18 s^-1 angstrom^3 per atomic-unit strength
```

The angular factors and strengths are

$$
\mathcal F_{1/2,1/2}=\frac23,
\qquad
\mathcal F_{3/2,1/2}=\frac43,
$$

$$
\mathscr S_{\rm ref}=12.338812633632335\ \mathrm{a.u.},
\qquad
|R|^2=18.508218950448505\ \mathrm{a.u.}.
$$

The other zero-field electronic branch must satisfy

$$
A(^2P_{3/2}\to{}^2S_{1/2})
=10^8\left(\frac{20001}{20000}\right)^3\ \mathrm{s}^{-1}.
$$

For the upper $M_u=+1/2$ block, the ordered basis is $(J=1/2,J=3/2)$. In wavenumber energy units,

$$
H_u^{(+1/2)}=
\begin{pmatrix}
20000+\beta(4-g_s)/6 & -\beta(g_s-1)\sqrt2/3\\
-\beta(g_s-1)\sqrt2/3 & 20001+\beta(g_s+2)/6
\end{pmatrix},
\quad
\beta=0.46686447719298224 B[\mathrm T].
$$

The lower energies are $\widetilde E_l(\pm1/2)=\pm\beta g_s/2$.

At **1 T**, label the two upper eigenstates in this block by ascending energy, `0` and `1`. The following subset is expected:

| Upper index | $M_l$ | $q$ | Transition wavenumber ($\mathrm{cm}^{-1}$) | $\lvert\mathcal G\rvert^2$ | $A$ ($\mathrm{s}^{-1}$) |
|---:|---:|---:|---:|---:|---:|
| 0 | $+1/2$ | 0 | 19999.64737066 | 0.05873363681 | $1.761915906\times10^7$ |
| 0 | $-1/2$ | +1 | 20000.58218241 | 0.27459969652 | $8.238710319\times10^7$ |
| 1 | $+1/2$ | 0 | 20000.88468206 | 0.27459969652 | $8.239084144\times10^7$ |
| 1 | $-1/2$ | +1 | 20001.81949382 | 0.05873363681 | $1.762490043\times10^7$ |

For either upper row pair, $\sum_l|\mathcal G|^2=1/3$. For this channel's complete six-state upper term, $\sum_{u,l}s_{ul}=2|R|^2$. The table alone covers only one upper $M$ block, so it is not the complete spectrum.

Use tolerances appropriate to the displayed rounding: for example, $10^{-6}\ \mathrm{cm}^{-1}$ in wavenumber and $10^{-7}$ relative in the listed rates. High-precision internal checks should instead use the defining constants and unrounded calculation.

### 13.8 Checks performed for this revision

The core LS equations were checked with independent Clebsch–Gordan/uncoupled dipoles and $3j/6j$ coupled dipoles. Numerical tests covered several integer and half-integer spin/orbital combinations, including singlets and $J=0$ cases, at $B=0,0.1,1,10$ T. They checked Hamiltonian signs, basis transformations, random complex basis rephasings, per-state LS sums, global norm sums, and the synthetic benchmark above.

These are algebraic and numerical consistency checks, not experimental validation of a real ion, an external atomic dataset, or a laboratory-plasma broadening model. Hyperfine, multipole, and general intermediate-coupling extensions require their own implementation tests and external benchmarks when coded.

---

## 14. Uncertainty, incomplete data, and reporting rules

### 14.1 Keep normalization uncertainty distinct from mixing uncertainty

A reference $A$ uncertainty generally propagates as a common normalization uncertainty for one strict-LS channel. Uncertainties in fine-structure intervals, $g$-factors, magnetic matrix elements, or relative dipole phases instead alter mixing and redistribute strength.

To first order,

$$
\frac{\delta A}{A}
\simeq3\frac{\delta\nu}{\nu}
+\frac{\delta s}{s}.
$$

For a nonzero amplitude $d$,

$$
\frac{|\delta s|}{s}\lesssim2\frac{|\delta d|}{|d|}
$$

to first order. Near an amplitude cancellation, relative errors can therefore be large even when absolute matrix-element errors are small. At an exact zero, use an absolute strength or rate bound rather than an undefined relative error.

Preserve uncertainties and correlations in small level separations. Subtracting two heavily rounded absolute energies is particularly dangerous for a near-degenerate pair. When available, use directly determined intervals and a consistent level-energy fit.

### 14.2 Sensitivity and convergence procedure

Vary the physically meaningful uncertain inputs, reconstruct Hermitian operators, rediagonalize, and compare the requested outputs. Do not perturb each matrix entry independently when doing so would violate symmetry, shared phase conventions, or known parameter correlations.

Test at least the ends of the requested field interval and regions of rapid mixing. Compare larger bases, plausible alternative atomic datasets, and both atomic-component and convolved-spectrum errors. Monte Carlo sampling is useful only when the assigned input distributions and correlations are justified; systematic data-quality categories are not automatically Gaussian standard deviations.

For high-dynamic-range spectra, report an absolute detectability/error floor as well as relative uncertainties on retained components.

### 14.3 Fail explicitly when the data do not support the requested model

| Condition | Required response |
|---|---|
| Only unresolved term energies are available, but fine-structure mixing is requested | Obtain resolved energies or label a specified approximate reconstruction; do not invent exact splittings |
| Significant interlevel mixing, but only individual unsigned $A$-values and diagonal $g_J$ are available | Identify the missing magnetic and phase information; use a justified restricted model, bounded sensitivity study, or return insufficient-data status |
| LS branch strengths disagree beyond their uncertainties | Reject strict one-norm reproduction; do not force independent branch normalizations while claiming strict LS physics |
| A required transition or magnetic coupling has no record | Preserve as unknown unless an exact symmetry sets it to zero |
| Only a blended/mixed-multipole rate or a lifetime without branching fractions is known | Do not infer a unique E1 matrix element |
| Hyperfine, electric-field, or truncation effects have not been assessed | Keep an explicit applicability warning rather than treating the omitted effect as established negligible |
| Only a restricted decay subset was calculated | Report partial rates; do not call their inverse a radiative lifetime |

### 14.4 Minimum statement accompanying a calculated line list

A useful report should state:

> Components were calculated with [Model A/B/C and extensions] over [field interval], using [identified data sources and versions] and [basis definition]. The listed strengths use [operator and unit convention]. Rates are spontaneous-emission probabilities from specified magnetic states, without population weighting. [Listed omitted interactions] were neglected with [quantified tests, or an explicit statement that they remain unassessed]. Reported uncertainties cover [specified inputs] and do not imply validation of an additional plasma line-shape model.

This distinguishes a reproducible atomic calculation from an unsupported claim of validity for every laboratory plasma or every species at 0.1–10 T.

---

## 15. References and revision notes

### External references

Sources were consulted for this revision on 13 October 2026. Database content and software interfaces can change; save the actual records and documentation versions used by the eventual code. URLs are given in code spans for unambiguous copying.

**[R1] NIST — Atomic Spectroscopy: Spectral Lines.** Definitions of transition probabilities, oscillator strengths, line strengths, statistical weights, SI conversions, and multipole distinctions.  
`https://www.nist.gov/pml/atomic-spectroscopy-compendium-basic-ideas-notation-data-and-formulas/atomic-spectroscopy`

**[R2] NIST — Atomic Spectroscopy: Zeeman Effect.** Weak-field shifts, Landé factors, and the magnetic energy scale.  
`https://www.nist.gov/pml/atomic-spectroscopy-compendium-basic-ideas-notation-data-and-formulas/atomic-spectroscopy-zeeman`

**[R3] NIST — Atomic Spectroscopy: Eigenvector Composition of Levels.** Atomic coupling mixtures and the interpretation of spectroscopic level labels.  
`https://www.nist.gov/pml/atomic-spectroscopy-compendium-basic-ideas-notation-data-and-formulas/atomic-spectroscopy-3`

**[R4] NIST Atomic Spectra Database — Spectral Lines Help.** Output units, line classification, wavelengths, uncertainty codes, and LS-derived multiplet subdivisions.  
`https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html`

**[R5] OPEN-ADAS — ADF04: Resolved specific ion data collections.** Scope of level, radiative, and collisional data; dataset-selection and process-code documentation.  
`https://open.adas.ac.uk/adf04`

**[R6] ADAS603 manual — Zeeman multiplet line fitting.** The xPaschen-related fixed-projection magnetic construction, population assumptions, and viewing-angle treatment. The present specification is not asserted to duplicate the complete code.  
`https://www.adas.ac.uk/man/chap6-03.pdf`

**[R7] W. Li, J. Grumer, T. Brage, and P. Jönsson (2020).** “HFSZEEMAN95—A program for computing weak and intermediate magnetic-field- and hyperfine-induced transition rates.” *Computer Physics Communications* **253**, 107211. Structure-based magnetic/hyperfine mixing and transition-matrix contraction.  
University record: `https://www.lu.se/publikation/39b7d42d-d174-44f3-8469-40e074891730`  
Publisher record: `https://www.sciencedirect.com/science/article/pii/S0010465520300485`

**[R8] NIST — Fundamental Physical Constants.** Source for a consistent set of SI constants and atomic-unit conversions.  
`https://physics.nist.gov/cuu/Constants/index.html`

**[R9] D. A. Steck — Rubidium 87 D Line Data, revision 2.3.4 (8 August 2025).** Hyperfine energies, magnetic conventions, and angular reduction; used as a formalism reference, not as atomic input for the synthetic example. For critical rubidium numerical data, follow its cited primary measurements.  
Permanent collection page: `https://steck.us/alkalidata/`

**[R10] G. Xu et al. (2022).** “First Laboratory Measurement of Magnetic-field-induced Transition Effect in Fe X at Different Magnetic Fields.” *The Astrophysical Journal* **937**, 48. Experimental example of magnetic-field-induced emission associated with closely spaced fine-structure levels.  
DOI: `10.3847/1538-4357/ac8cfa`

**[R11] K. A. Touati, K. Chenini, and M. T. Meftah (2020).** “Stark-Zeeman Broadening of Spectral Line Shapes in Magnetized Plasmas.” *Atoms* **8**, 91. Example of a line-shape calculation including the motional Stark contribution.  
`https://www.mdpi.com/2218-2004/8/4/91`

**[R12] E. Stambulchik (2023; corrected 9 September 2024).** “Stark Broadening of Lyman-α in the Presence of a Strong Magnetic Field.” *Atoms* **11**, 120. Component-dependent Stark–Zeeman behavior; use the corrected version.  
`https://www.mdpi.com/2218-2004/11/9/120`

### Main changes from the original user-supplied guide

The original file, `zeeman_paschen_back_transition_strength_guide.md`, supplies the starting organization and isolated-term objective. This revision makes the following substantive changes:

- Replaces the invalid dipole-commutation justification with a fixed-basis/operator assumption.
- Separates isolated-$J$, isolated-LS-term, and arbitrary-coupling external-matrix models.
- Makes the Hamiltonian and dipole phase convention explicit and consistent.
- Uses SI Einstein-$A$ prefactors and separates level strengths from magnetic-state strengths.
- Expands external-data inversion to $A$, $f$, $gf$, line strengths, and lifetime/branch data.
- Limits the one-reference normalization to one justified orbital channel and tests all available branches.
- Includes external magnetic matrices and relative-amplitude information in the general model.
- Adds hyperfine/isotope and multipole pathways, a plasma-model interface, basis/uncertainty checks, and a reproducible synthetic benchmark.
- Replaces the exact-triplet assertion by the correct three-slope-group asymptote with possible residual structure.
- Distinguishes a valid atomic component calculation from a validated laboratory-plasma spectrum.

**End of specification.**
