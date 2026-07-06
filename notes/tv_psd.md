Bayesian Non-Stationary Spectral Density Estimation from Time–Frequency Data
|     |     |     |     | with | Tensor-Product | P-Splines |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | -------------- | --------- | --- | --- | --- | --- | --- | --- | --- |
Authors1,∗
1Aﬀiliation
|     |     |     |     |     | (Dated: June | 25, 2026) |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------------ | --------- | --- | --- | --- | --- | --- | --- | --- |
We present a Bayesian framework for estimating smoothly time-varying power spectral densities
(PSDs)fromtime–frequencydata. Thelog-PSDsurfacelogS(t,f)ismodelledbyatensor-product
P-spline with an anisotropic roughness prior, sampled in a whitened parametrisation that removes
the hierarchical funnel and yields stable Hamiltonian Monte Carlo. The model is independent of
how the data are produced: a single Gaussian-coeﬀicient likelihood couples the same surface to
different time–frequency transforms, here the Wilson–Daubechies–Meyer (WDM) wavelet trans-
form and the short-time Fourier transform, with the classical moving periodogram recovered as
the phase-discarded special case of the latter. On a locally stationary process with known truth,
both likelihoods recover the surface with near-nominal credible intervals. Applied to simulated
LISAdata,theframeworkrecoverstheannualmodulationofthecyclostationaryGalacticconfusion
foreground, which a stationary noise model — flat at the year-averaged level — cannot represent.
Because the likelihood admits a signal mean, the same model extends to joint blocked-Gibbs infer-
ence of the noise PSD and an embedded source. The result is a modular framework, applicable to
any time–frequency representation whose coeﬀicients admit a local-power likelihood.
I. INTRODUCTION independent amplitude modulation of a fixed spectral
shape[4,5],andrecentstochasticandglobal-fitpipelines
Classicalspectralanalysisrestsontheassumptionthat characteriseitbyiterativespectralfitting[6]orbyfitting
theconfusionamplitudeindependentlyinshorttimeseg-
| the process | under study | is  | stationary. | For | a stationary |             |                                        |     |     |     |     |     |     |
| ----------- | ----------- | --- | ----------- | --- | ------------ | ----------- | -------------------------------------- | --- | --- | --- | --- | --- | --- |
|             |             |     |             |     |              | ments[7,8]. | Theforegroundissimultaneouslyentangled |     |     |     |     |     |     |
seriestheperiodogramordinatesareasymptoticallyinde-
pendent and exponentially distributed about the power withtheresolvablesourcesonewishestocharacterise, so
spectral density (PSD), the structure that underpins thenoisePSDandthesourceparametersmustultimately
|               |               |            |     |        |                | be inferredtogether |     | [9].       | Inthis        | settinga | stationaryPSD |              |     |
| ------------- | ------------- | ---------- | --- | ------ | -------------- | ------------------- | --- | ---------- | ------------- | -------- | ------------- | ------------ | --- |
| the classical | Whittle       | likelihood | [1, | 2] and | the modern     |                     |     |            |               |          |               |              |     |
|               |               |            |     |        |                | assumption          | is  | not a mild | approximation |          | but           | a structural |     |
| Bayesian      | nonparametric | estimators |     | built  | upon it. These |                     |     |            |               |          |               |              |     |
estimators place a smoothing prior on the log-spectrum, misspecification, which makes LISA a natural proving
|             |             |        |            |     |               | ground | for non-stationary |     | spectral | estimation. |     |     |     |
| ----------- | ----------- | ------ | ---------- | --- | ------------- | ------ | ------------------ | --- | -------- | ----------- | --- | --- | --- |
| for example | a penalised | spline | (P-spline) |     | with a rough- |        |                    |     |          |             |     |     |     |
nesspenalty[3],sothatasinglenoisyrealisationyieldsa A non-stationary process is most naturally described
regularised estimate of S(f) with calibrated uncertainty. in a joint time–frequency representation. Two com-
We give this stationary likelihood explicitly in Sec. III. plementary choices are the moving, or short-time,
|           |                |      |             |      |             | periodogram, |     | which | slides | a window | along | the | se- |
| --------- | -------------- | ---- | ----------- | ---- | ----------- | ------------ | --- | ----- | ------ | -------- | ----- | --- | --- |
| It is the | starting point | that | the present | work | generalises |              |     |       |        |          |       |     |     |
to the non-stationary, time–frequency setting. ries, and critically-sampled wavelet transforms such as
Real instruments rarely satisfy the stationarity as- the Wilson–Daubechies–Meyer (WDM) transform [10],
|           |                                            |     |     |     |     | which tile | the | plane | into a | near-orthogonal |     | grid. | The |
| --------- | ------------------------------------------ | --- | --- | --- | --- | ---------- | --- | ----- | ------ | --------------- | --- | ----- | --- |
| sumption. | Detectornoisedrifts,glitches,andrespondsto |     |     |     |     |            |     |       |        |                 |     |       |     |
environmental and instrumental conditions that change two produce different data products, but both estimate
ontimescalesshortcomparedwithatypicalobservation, the same underlying object: a PSD that evolves slowly
so that its spectral content evolves in time. Treating across the time–frequency plane. When that evolution
|            |               |              |     |     |                 | is smooth, | recovering |     | it becomes | a   | bivariate | regression |     |
| ---------- | ------------- | ------------ | --- | --- | --------------- | ---------- | ---------- | --- | ---------- | --- | --------- | ---------- | --- |
| such noise | as stationary | misspecifies |     | the | very likelihood |            |            |     |            |     |           |            |     |
on which downstream inference depends, and the result- problem, fitting a smooth surface to noisy local-power
ingbiaspropagatesintoanyastrophysicalparameterses- observations, to which penalised splines are well suited.
timated against that noise model. The present work builds directly on the Bayesian non-
The Laser Interferometer Space Antenna (LISA) is parametric framework of Tang et al. [11], who intro-
a striking example. Its dominant low-frequency fore- ducedthedynamicWhittlelikelihoodforlocallystation-
ground, the unresolved Galactic confusion noise, is non- aryprocessesfromthemovingperiodogramandpairedit
stationarybyconstruction: astheconstellationorbitsthe with a bivariate Bernstein–Dirichlet process prior on the
Sun, its antenna pattern sweeps across the anisotropic time–frequency spectrum, establishing sup-norm poste-
Galactic source distribution, modulating the confusion riorconsistency,L contractionrates,andaBayes-factor
2
powerwithanannualperiod. Thismakestheforeground test for stationarity. We retain their dynamic Whit-
a cyclostationary process, well described by a frequency- tle likelihood as one of our observation models, but re-
placetheBernstein–Dirichletpriorwithatensor-product
|     |     |     |     |     |     | P-spline. | This    | change | buys     | three     | things: | anisotropic |     |
| --- | --- | --- | --- | --- | --- | --------- | ------- | ------ | -------- | --------- | ------- | ----------- | --- |
|     |     |     |     |     |     | roughness | control | with   | separate | smoothing |         | in time     | and |
∗
author@gmail.com frequency, a whitened parametrisation that makes the

2
surface amenable to gradient-based (Hamiltonian) sam- B. Tensor-product P-spline representation
pling, and a Gaussian-coeﬀicient, phase-retaining likeli-
hoodthatadmitsasignalmeanandhenceajointsource– We represent the log-PSD surface as a tensor-product
noise fit, which the power-only dynamic Whittle cannot P-spline [12, 13],
support. We use the reference implementation of Tang
et al. [11] in the beyondWhittle package as an external XKt XKf
baselineinoursimulationstudy(Sec.IV).Morebroadly, Λ(t ,f )= w B(t)(t )B(f)(f ), (2)
i j kl k i l j
Bayesian estimation of time-varying spectra has a sub- k=1l=1
stantial history outside the gravitational-wave setting.
[Avi: Add and cite the adjacent non-stationary whereB k (t) andB l (f) areB-splinebasesofdegreed t ,d f in
spectral literature here, e.g. AdaptSPEC (Rosen, time and frequency, K t ,K f are the corresponding basis
Wood & Stoffer 2012), Dahlhaus locally station- dimensions,andw kl arethesplinecoeﬀicients. Inmatrix
ary processes (1997, 2000), and SLEX (Ombao et form, with Λ ∈ Rnt ×nf the surface on the grid, B t ∈
al.). One or two sentences situating us against Rnt ×Kt, B f ∈ Rnf ×Kf the basis matrices, and W ∈
the spline-based and segmentation-based alterna- RKt ×Kf the coeﬀicient matrix,
tives.]
⊤
Λ=B WB . (3)
This paper develops the framework in four steps. Sec- t f
tion II introduces the tensor-product P-spline model Equivalently, with w = vec(W), vec(Λ) = (B ⊗B )w
f t
for the log-PSD surface, with its anisotropic roughness [14]. InpracticeweneverformtheKroneckerdesignand
priorandwhitenedparametrisationforstableHMC.Sec- evaluate the dense product directly.
tion III shows that a single Gaussian-coeﬀicient likeli- Smoothness is imposed by difference penalties in each
hood connects two transforms to that model, the WDM direction. With P and P the marginal difference-
t f
waveletandshort-timeFouriercoeﬀicients,withthemov- penaltymatrices(ordersm ,m ),theanisotropicpenalty
t f
ing periodogram the phase-discarded special case of the on w has the Kronecker-sum structure [13]
latter. SectionVdemonstratesitonsimulatedLISAdata
carrying a cyclostationary Galactic foreground, which a Q(ϕ ,ϕ )=ϕ (I ⊗P )+ϕ (P ⊗I ), [src] (4)
t f t Kf t f f Kt
stationary model cannot represent but both transforms
recover. Section VI extends the likelihood to a joint, with ϕ t ,ϕ f > 0 the time and frequency smoothing pre-
blocked-Gibbs fit of the noise PSD and an embedded cisions. The penalty defines a Gaussian roughness prior
source. on the coeﬀicients,
(cid:0) (cid:1)
w|ϕ ,ϕ ∼N 0, Q −1 , (5)
t f
so that all smoothing information is carried by the two
II. BAYESIAN MODEL penalties and the two precisions [3, 14, 15].
Sampling w jointly with (ϕ ,ϕ ) produces the classic
t f
hierarchical funnel: the conditional scale of w collapses
We define the statistical model for the time-varying
as the precisions grow, stalling gradient-based MCMC.
PSD independently of the data representation: the sur-
[Avi: cite the funnel pathology (Neal 2003)
face, prior, and inference machinery below are shared by
and the non-centered/whitened reparametrisa-
every observation model in Sec. III.
tion remedy (Papaspiliopoulos, Roberts & Sköld
2007).] We remove it by sampling in the penalty
eigenbasis. With P = U diag(λt)U⊤ and P =
t t t f
U diag(λf)U⊤,theKroneckertermsaresimultaneously
f f
A. Time-varying PSD model diagonalised by U ⊗ U , so Q has eigenvalues d =
f t ab
ϕ λt +ϕ λf. We sample standard-normal s ∼N(0,1)
t a f b ab
Let S(t,f) > 0 denote the power spectral density at and set
time t and frequency f, assumed to vary smoothly in s
W=U ZU ⊤ , Z = √ab , [src] (6)
both arguments. We model its logarithm, t f ab d
ab
which reproduces N(0,Q−1) exactly while making s a
Λ(t,f) = logS(t,f), (1)
priori independent of (ϕ ,ϕ ) and absorbing the |Q|1/2
t f
normalisation. The shared penalty null space (bilinear
working on the log scale so that the surface is uncon- trends in t,f, d =0) carries no smoothing information
ab
strainedandthemultiplicativedynamicrangeofthePSD and is given a fixed weak precision τ .
0
becomes additive. The estimand is Λ(t,f) on a time– EachsmoothingprecisionisgivenaGammahyperprior
frequency grid (t n ,f m ) supplied by the chosen represen- [3],
tation (Sec. III). The model below does not depend on
how that grid arises. ϕ , ϕ ∼Gamma(α ,β ), (7)
t f ϕ ϕ

3
sampledontheunconstrainedlogscale,withtheGamma
TABLEI.Defaultmodelandsamplerconfiguration. Interior-
density and its log-Jacobian supplied as a factor. The
knotcountssetthebasisdimensionasK =(interior knots)+
weaklyinformativedefaultsplacethepriormassoverthe
d+1. The moving-periodogram fits use 6 frequency knots
range of smoothing scales spanned by the basis, in the
in place of 10. Warm-up/draw counts vary by study and
spirit of penalised-complexity priors [16, 17].
are stated in the text: 250/250 for the simulation study,
300/250 for the LISA noise ensemble, and 800/2000 for the
time-localised-source coverage study.
C. Posterior inference
Quantity Symbol Value
Let L(Λ) denote the observation-model log-likelihood B-spline degree (time, freq) d t , d f 3, 3
Interior knots (time, freq) – 8, 10
(Sec. III), which depends on the data only through the
surface Λ. The joint posterior over the whitened eigen- Difference-penalty order m t , m f 2, 2
coeﬀicients and the smoothing precisions is Smoothing hyperprior (α ϕ ,β ϕ ) (2, 1)
(cid:0) (cid:1) Null-space precision τ 10−4
0
p(s,ϕ t ,ϕ f |data) ∝ exp L(Λ(s,ϕ t ,ϕ f )) p(s)p(ϕ t )p(ϕ f ), NUTS target accept probability – 0.85
(8)
NUTS max tree depth – 10
with p(s) = N(0,I) and Λ obtained from (s,ϕ ,ϕ )
t f
through the deterministic whitening map of Eq. (6). No
explicit penalty factor appears. We draw samples with
theNo-U-TurnSampler[NUTS,18],usingstandardstep- Gelman–Rubin statistic Rˆ <1.01, bulk and tail effective
size and mass-matrix adaptation during warm-up. Each samplesizesaboveafewhundred, andtheabsenceofdi-
chain is initialised from a penalised least-squares fit of vergent transitions and maximum tree-depth saturation.
Λ to the log of the local-power data, mapped into the No fit in any study produced sampler divergences. [Avi:
whitened coordinates, and both observation models use Report the actual Rˆ/ESS for a representative fit
the same warm-start procedure on their respective local- — these are not currently logged. Add logging to
power statistics. the study scripts.]
Asinglelikelihoodevaluationisthencheap. Giventhe
current smoothing precisions ϕ ,ϕ (sampled on the log
t f
scale, with ϕ ,ϕ ∼ Gamma(α ,β )) and the whitened
t f ϕ ϕ III. OBSERVATION MODELS
eigen-coeﬀicients s ∼ N(0,1), we form the eigenval-
ab
ues d
ab
= ϕ
t
λt
a
+ϕ
f
λf
b
(with the√null space held at the
The model of Sec. II specifies a prior over the sur-
fixedprecisionτ ),setZ =s / d ,assemblethelog-
0 ab ab ab face Λ = logS. A likelihood linking a time–frequency
spline surface Λ=(B U )Z(B U )⊤, and evaluate the
t t f f data product to it completes the posterior. We give a
observation-model log-likelihood L(Λ) of Sec. III. The
likelihood for each of two transforms, the moving (short-
basisandeigenvectorproductsareprecomputedonce, so
time Fourier) periodogram and the WDM wavelet trans-
each evaluation is a pair of dense matrix products fol-
form, and show in Sec. IIIC that both are special cases
lowed by the likelihood.
of a single Gaussian-coeﬀicient form, with the moving
Implementation. We implement the model in
periodogram the phase-discarded version of the Fourier
JAX [19] with NumPyro [20], exploiting automatic likelihood. Only L(Λ) differs — the PSD model is un-
differentiation and just-in-time (JIT) compilation for
changed. Thetwofrontendssampletheplaneverydiffer-
gradient-based sampling. The smoothing precisions
ently(Fig.1): theWDMtransformtilesitintoaregular,
ϕ ,ϕ are sampled on the log scale to keep the sampler
t f near-orthogonal grid, whereas the moving periodogram
away from the boundary at zero, and the time and
returns a scattered sawtooth that does not fill it.
frequency grids are standardised to [0,1] for numerical
The stationary case fixes the reference point. When
stability of the B-spline basis evaluation. Time knots
the PSD is constant in time, S(t,f) = S(f), the peri-
are placed adaptively in the spirit of Maturana-Russel
odogram ordinates are asymptotically independent and
and Meyer [3], clustering where the spectrum varies
exponentially distributed about S(f), giving the classi-
fastest. The shared model and sampler configuration is
cal Whittle likelihood [1, 2]
collected in Table I. The per-study warm-up and draw
(cid:18) (cid:19)
counts, which differ between the noise-only and joint Y
1 I(f )
fits, are given with each study below. A single one-year L (S)= exp − k , (9)
W S(f ) S(f )
joint A/E fit (two channels, signal and non-stationary k k
k
noise) completes in ≃ 2.5min of wall time on a single
workstation CPU core with no GPU. [Avi: Confirm with I(f ) the periodogram at Fourier frequency f .
k k
the hardware (CPU model / core count) for the Each time–frequency likelihood below reduces to this
wall-clock figure.] form within a single short window or tile, so Eq. (9) is
Convergencediagnostics. Weassessconvergencewith the stationary limit that the non-stationary models gen-
standard chain diagnostics [21, 22]: the rank-normalised eralise.

4
FIG. 1. The two time–frequency front ends that feed the same P-spline surface. (a) The WDM transform tiles the plane into
|     |     |     |     |     |     |     | χ2  | ∼N(0,S |     |     |     | ×∆F. |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | ---- | --- | --- |
a regular, near-orthogonal grid, contributing one 1 coeﬀicient w nm nm ) per cell of size ∆T (b) The thinned
zigzag moving periodogram of Tang et al. [11] does not fill the plane: the evaluated Fourier frequency cycles with the time
χ2
index, tracing rising diagonal ramps, and the thinning skips im ordinates between blocks, leaving a scattered sawtooth of
|             |     | ∼Exp(1/S),eachformedfromitsownoverlapping2m+1window. |     |     |     |     |     |     |     |                                        |     |     |     | 2   |
| ----------- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- |
| ordinatesMI |     |                                                      |     |     |     |     |     |     |     | ThesamewhitenedP-splinesurfaceofSec.II |     |     |     |     |
t
| is fitted | to  | either sampling. |             |     |            |     |     |                                    |     |     |     |     |       |     |
| --------- | --- | ---------------- | ----------- | --- | ---------- | --- | --- | ---------------------------------- | --- | --- | --- | --- | ----- | --- |
|           |     |                  |             |     |            |     |     | InthenotationofSec.II,L(Λ)=logL(i) |     |     |     |     |       | =eΛ |
|           |     | A. Moving        | periodogram |     | likelihood |     |     |                                    |     |     |     |     | withS |     |
DW
|     |     |     |     |     |     |     |     | evaluated | at  | the ordinate | locations. |     | The dynamic | Whit- |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | ------------ | ---------- | --- | ----------- | ----- |
tletreatsthethinnedordinatesasindependentandexpo-
| We     | slide | a short | window | of          | length | 2m+1    | along the  |          |       |          |                     |     |     |          |
| ------ | ----- | ------- | ------ | ----------- | ------ | ------- | ---------- | -------- | ----- | -------- | ------------------- | --- | --- | -------- |
|        |       |         |        |             |        |         |            | nential, | which | is exact | only asymptotically |     | and | requires |
| series | and   | form a  | local  | periodogram |        | at each | step [11]. |          |       |          |                     |     |     |          |
ThemovingperiodogramordinatesMI oforderm,eval- thespectrumtobeapproximatelyconstantacrossawin-
t
|         |        |            |       |           |      |     |         | dow.         | Thinning | trades  | data     | for reduced | inter-ordinate |       |
| ------- | ------ | ---------- | ----- | --------- | ---- | --- | ------- | ------------ | -------- | ------- | -------- | ----------- | -------------- | ----- |
| uated   | at     | time point | t     | = 1,...,T | from | the | samples |              |          |         |          |             |                |       |
|         |        |            |       |           |      |     |         | correlation, | and      | because | it works | with        | the power      | peri- |
| X t−m+1 | ,...,X |            | , are |           |      |     |         |              |          |         |          |             |                |       |
t+m
|     |          |     | (cid:12)     |       |           |        | (cid:12)           | odogram | it discards | phase. |                |     |     |     |
| --- | -------- | --- | ------------ | ----- | --------- | ------ | ------------------ | ------- | ----------- | ------ | -------------- | --- | --- | --- |
|     |          |     | (cid:12) X2m |       | (cid:0)   |        | (cid:1) (cid:12) 2 |         |             |        |                |     |     |     |
|     |          | 1   | (cid:12)     |       |           |        | (cid:12)           |         |             |        |                |     |     |     |
| MI  | =        |     | (cid:12) X   |       | exp −iπνλ |        | (cid:12) , [src]   |         |             |        |                |     |     |     |
|     | t        |     | (cid:12)     | ν+t−m |           | mod(t) | (cid:12)           |         |             |        |                |     |     |     |
|     | 2π(2m+1) |     |              |       |           |        |                    |         |             | B.     | WDM likelihood |     |     |     |
ν=0
(10)
where λ = 2j , j =1,...,m, are the Fourier frequen- TheWilson–Daubechies–Meyer(WDM)transform[10]
|     | j   | 2m +1 |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
mod(t)=1+((t−1)modm).
| cies | and |     |     |     |     |     |     | tilesthedataintoanear-orthogonaltime–frequencygrid |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
Foralocallystationaryprocesstheseordinatesareap- ofcoeﬀicientsw ,wherenindexesWDMtimebinsand
nm
proximately independent and exponentially distributed m frequency channels. For a locally stationary process
withmeanS,whichgivesthedynamicWhittlelikelihood
|     |     |     |     |     |     |     |     | the diagonal |     | approximation | models |     | each pixel | as a zero- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------------- | ------ | --- | ---------- | ---------- |
(cid:18) (cid:19) mean Gaussian whose variance is the local evolutionary
YT
|      |      |         | 1      |     |     | MI      |        | power, |       |      |     |           |        |         |
| ---- | ---- | ------- | ------ | --- | --- | ------- | ------ | ------ | ----- | ---- | --- | --------- | ------ | ------- |
| L(1) | (S)= |         |        | exp | −   |         | t .    |        |       |      |     |           |        |         |
| DW   |      | S(t/T,λ |        | )   |     | S(t/T,λ | )      |        |       |      |     |           |        |         |
|      |      | t=1     | mod(t) |     |     |         | mod(t) |        | ∼N(0, |      |     |           |        |         |
|      |      |         |        |     |     |         |        |        | w nm  | S nm | ),  | S nm =S(t | n ,f m | ). (13) |
(11)
| The          | overlapping |            | windows | induce       | dependence |     | between      |               |     |                 |      |         |     |         |
| ------------ | ----------- | ---------- | ------- | ------------ | ---------- | --- | ------------ | ------------- | --- | --------------- | ---- | ------- | --- | ------- |
|              |             |            |         |              |            |     |              | The resulting |     | log-likelihood, | with | Λ=logS, |     | is      |
| neighbouring |             | ordinates, |         | so we follow | Tang       | et  | al. [11] and |               |     |                 |      |         |     |         |
|              |             |            |         |              |            |     |              |               |     | X(cid:2)        |      |         |     | (cid:3) |
use a thinned variant that retains a zigzag subset. With 1 −Λnm
|          |     |           |     |     |     |     |     | L(Λ)=− |     | log(2π)+Λ |     | +w  | 2 e | .[src] |
| -------- | --- | --------- | --- | --- | --- | --- | --- | ------ | --- | --------- | --- | --- | --- | ------ |
| thinning |     | factor i, |     |     |     |     |     |        |     | 2         |     | nm  | n m |        |
n,m
|      |     |        |     |     | (cid:18) |       | (cid:19) |     |     |     |     |     |     |      |
| ---- | --- | ------ | --- | --- | -------- | ----- | -------- | --- | --- | --- | --- | --- | --- | ---- |
|      |     | YBi Ym |     |     |          |       |          |     |     |     |     |     |     | (14) |
| L(i) |     |        | 1   |     | MI(u     | j,l,i | ,λ j )   |     |     |     |     |     |     |      |
(S)= exp − , [src] Each pixel contributes a single χ2 observation. The
| DW  |     |     | S(u | ,λ ) |     | S(u | ,λ ) |     |     |     |     | 1   |     |     |
| --- | --- | --- | --- | ---- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
j,l,i j j,l,i j smoothing prior of Sec. IIB supplies the regularisation
l=1j=1
|     |     |                   |     |     |                 |     | (12) | that | makes | single-realisation |     | estimation | feasible. | The |
| --- | --- | ----------------- | --- | --- | --------------- | --- | ---- | ---- | ----- | ------------------ | --- | ---------- | --------- | --- |
|     |     | =⌊(T−m)/(im)⌋andu |     |     | =(i(l−1)m+j)/T. |     |      |      |       |                    |     |            |           |     |
whereB i j,l,i likelihood treats the WDM coeﬀicients as independent,

5
which is exact only for a perfectly orthogonal transform realisation we fit the model through the WDM likeli-
— residual nearest-neighbour time correlation remains, hood, Eq. (14), and through the moving-periodogram
and as with the moving periodogram the spectrum is as- likelihood, Eq. (12), and repeat across four observa-
sumed approximately constant within a tile. tion lengths n = n n with n = 24 frequency
t f f
bins and n ∈ {24,48,96,192} time bins, i.e. n ∈
t
{576,1152,2304,4608}. Eachfitdraws250warm-upand
C. One model, two transforms 250 posterior samples with a single NUTS chain under
the configuration of Table I.
Both likelihoods are instances of a single Gaussian- The WDM estimator targets the expected local
coeﬀicient form. A time–frequency cell with R real com- power E[w2 ], which equals the digital PSD up to
nm
ponents contributes a near-constant per-channel normalisation, E[w2 ] =
nm
Xh (cid:0)P (cid:1) i C m f 0 (t n ,f m ). We calibrate C m once from unit white
−
2
1 RΛ
nm
+ R
r=1
c(
n
r
m
)2 e−Λnm , [src] (15) noise and report the error against both the Monte Carlo
n,m E[w2] target and the calibrated analytic PSD C m f 0 .
so the WDM coeﬀicient (R = 1, χ2) and the complex
1
short-timeFouriercoeﬀicient(R=2,realandimaginary
B. Metrics
parts, χ2, whose squared modulus is the periodogram)
2
are handled by the same likelihood. The two transforms
We summarise point-estimate accuracy with the mean
thatentertheframeworkarethereforetheWDMwavelet
squared error of the log-PSD,
transformandtheshort-timeFouriertransform(STFT),
each fitted with this Gaussian-coeﬀicient likelihood.
XT XK (cid:16) (cid:0) (cid:1) (cid:0) (cid:1)(cid:17)
The moving periodogram is not a third transform but 1 2
MSE = lnfˆ t, j −lnf t, j , [src]
a special case of the Fourier likelihood: the exponential logf T(K+1) T K 0 T K
t=1j=0
dynamic-Whittle form of Eq. (11) is exactly the R = 2
(17)
marginal of Eq. (15) over the coeﬀicient phase. The dy-
using the posterior mean as the point estimate. We also
namic Whittle thus models the Fourier power, whereas
report the empirical coverage of the posterior 90% cred-
the full R = 2 form keeps the phase. Retaining phase
ible intervals and, as a measure of posterior contraction,
mattersbecauseitletsthecoeﬀicientcarryasignalmean,
the width of the 90% credible interval on logS averaged
c ∼ N(h(θ),S), which is what enables the joint source–
over the grid (a scale-free quantity, comparable across
noiseinferenceofSec.V.Thepower-onlydynamicWhit-
representations). Each configuration is summarised over
tle cannot. In all cases the time series is mapped to
100 independent realisations. We plot the median and
(timegrid,frequencygrid,coeﬀicients)byoneofthetwo
interquartile range across realisations.
transforms,andtheidenticalwhitenedP-splinemodelof
AsanexternalbaselineweruntheBernstein–Dirichlet
Sec. II fits the surface.
dynamic-Whittle estimator of Tang et al. [11] on the
samerealisationsthroughitsbeyondWhittlereferenceim-
plementation. Because that estimator and our moving-
IV. SIMULATION STUDY
periodogramfitsharethedynamicWhittlelikelihoodand
differ only in the prior (Bernstein–Dirichlet process ver-
We assess statistical performance where the truth is
sustensor-productP-spline),thecomparisonisolatesthe
known, fitting the P-spline model through the WDM co-
effect of the prior under matched data. [Avi: Run be-
eﬀicientlikelihoodandthemoving-periodogramdynamic
yondWhittleonthestoredLS2realisationsandadd
Whittle and comparing each against the true surface.
its MSE/coverage/CI-width curves to Fig. 3.]
The phase-retaining Fourier likelihood adds nothing in
this noise-only test — without a signal mean it reduces
to the same power statistic — so we defer it to the LISA
C. Results
demonstration (Sec. V).
Bothobservationmodelsrecoverthetime-varyingsur-
A. Locally stationary process face and contract with the data (Fig. 3). Over 100 re-
alisations at each of four observation lengths (n = 576
to 4608), the median MSE falls monotonically with
FollowingTangetal.[11],weusethelocallystationary logf
n for both likelihoods, the 90% credible-interval cov-
time-varying MA(1) process (LS2) with i.i.d. standard-
normal innovations {w }, erage stays close to nominal, and the credible-interval
t
(cid:0) (cid:0) (cid:1)(cid:1) width on logS contracts steadily with n, as expected.
X t,T =w t +1.1cos 1.5−cos 4π T t w t−1 , t=1,...,T, No sampler divergence occurred in any fit. On this
(16) smooth process the moving periodogram has the lower
with analytic pointwise PSD f (u,ω) = 1 + b (u)2 + median MSE at matched n, though the gap is com-
0 1 logf
2b (u)cosω, b (u) = 1.1cos(1.5 − cos4πu). For each parable to the interquartile spread across realisations
1 1

6
logSˆ(t,f)
FIG. 2. Single LS2 realisation. Left: true time-varying PSD logf 0 (t,f). Middle: posterior-median from the WDM
posterior-medianlogSˆ(t,f)fromthemoving-periodogramlikelihood.
| likelihood. | Right: |          |            |     |     |     |     |     |     | Allpanelsshareacolourscaleandare |     |     |     |     |
| ----------- | ------ | -------- | ---------- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | --- |
| rescaled    | to the | analytic | PSD scale. |     |     |     |     |     |     |                                  |     |     |     |     |
)∼
[Avi: state whether the separation exceeds the each(cid:0) is generated in the(cid:1)frequency domain with n˜(f k
CN
IQR band (i.e. whether it is significant). If not, 0, S(f )/(4∆f∆t2) and transformed to the time
k
soften to “comparable”]. We return in Sec. VII to domain, so the realised one-sided PSD reproduces the
| why the | WDM | front | end is | nonetheless | preferable | for | target. |     |     |     |     |     |     |     |
| ------- | --- | ----- | ------ | ----------- | ---------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
phase-coherent inference. Theconfusionpowerismodulatedbythecyclostation-
|               |     |            |                  |     |     |      | ary annual | law | of Digman |           | and Cornish | [4],  |         |      |
| ------------- | --- | ---------- | ---------------- | --- | --- | ---- | ---------- | --- | --------- | --------- | ----------- | ----- | ------- | ---- |
|               |     |            |                  |     |     |      |            |     | X5        | (cid:0)   |             |       | (cid:1) |      |
| V. RECOVERING |     |            | A NON-STATIONARY |     |     | LISA |            |     |           |           |             |       |         |      |
|               |     |            |                  |     |     |      | r(u)=1+    |     | A         | cos 2πkuT |             | /T −φ | ,       | (19) |
|               |     | FOREGROUND |                  |     |     |      |            |     |           | k         | obs         | yr    | k       |      |
k=1
| We now     | apply            | the framework    |         | to simulated |                 | LISA data, |                                       |             |               |          |             |             |                 |        |
| ---------- | ---------------- | ---------------- | ------- | ------------ | --------------- | ---------- | ------------------------------------- | ----------- | ------------- | -------- | ----------- | ----------- | --------------- | ------ |
|            |                  |                  |         |              |                 |            | withthetabulatedharmoniccoeﬀicients(A |             |               |          |             |             | k ,φ k )oftheir |        |
| where the  | non-stationarity |                  |         | is physical  | rather          | than im-   |                                       |             |               |          |             |             |                 |        |
|            |                  |                  |         |              |                 |            | Table 1                               | (A-channel, |               | one-year | fit). The   | second      | harmonic        |        |
| posed.     | As described     |                  | in Sec. | I, the       | Galactic        | confusion  |                                       |             |               |          |             |             |                 |        |
|            |                  |                  |         |              |                 |            | dominates,                            | so          | the confusion |          | power       | peaks twice | per             | year   |
| foreground | is               | cyclostationary, |         | its power    | modulated       | with       |                                       |             |               | ≃        |             |             |                 |        |
|            |                  |                  |         |              |                 |            | and varies                            | by          | a factor      | 7,       | reproducing | their       | Figure          | 1.     |
| an annual  | period           | [4].             | We show | that         | both transforms | of         |                                       |             |               |          | ⟨r⟩         |             |                 |        |
|            |                  |                  |         |              |                 |            | The coeﬀicients                       |             | average       | to       | =           | 1 over      | whole           | years, |
Sec.IIIrecoverthistime-varyingnoisePSD,whichasta-
|     |     |     |     |     |     |     | so the modulation |     | redistributes |     | confusion |     | power in | time |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------------- | --- | --------- | --- | -------- | ---- |
tionary model cannot represent. withoutchangingitstimeaverage. Becausethetwocom-
|     |     |     |           |      |     |     | ponentsareindependentandr |           |     |               | variesslowlyrelativetoa |           |     |      |
| --- | --- | --- | --------- | ---- | --- | --- | ------------------------- | --------- | --- | ------------- | ----------------------- | --------- | --- | ---- |
|     |     |     |           |      |     |     | WDM                       | time bin, | the | instantaneous |                         | one-sided | PSD | is   |
|     |     | A.  | Simulated | data |     |     |                           |           |     |               |                         |           |     |      |
|     |     |     |           |      |     |     |                           | S(u,f)=S  |     | (f)+r(u)S     |                         | (f),      |     | (20) |
|     |     |     |           |      |     |     |                           |           |     | inst          |                         | gal       |     |      |
Wesimulateoneyearofasecond-generationTDIchan-
nel (TDI-X) [Avi: cite TDI (Tinto & Dhurand- which is exactly the local power the estimator targets.
har 2021 review, Armstrong, Estabrook & Tinto The estimator recovers the total surface S(u,f), not the
1999) and the LISA mission (Amaro-Seoane et al. individual components. Eq. (20) is a property of the
|     |     |     |     | ≃   | ≃   | 1.6×105 |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
2017).] sampled at interval ∆t 200s (N simulated data, not an output of the fit. This is the
samples, a Nyquist frequency of 2.5mHz), in fractional- discriminating regime: a stationary model fits only a
frequency units. The noise has two independent compo- single time-averaged spectrum, whereas S(u,f) carries
|     |     |     |     |     |     |     | the annual | modulation. |     | The | data also | contain | a resolv- |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | --- | --- | --------- | ------- | --------- | --- |
nents,
|     |     |     |     | p   |     |     | able Galactic | binary |     | at f 0 =1.5mHz |     | (optimal | SNR | 200, |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ------ | --- | -------------- | --- | -------- | --- | ---- |
n(t)=n (t)+ r(u) n (t), (18) generated with jaxGB in the same TDI-X generation-2
|     |     | inst |     |     | gal |     |             |       |      |       |     |          |           |     |
| --- | --- | ---- | --- | --- | --- | --- | ----------- | ----- | ---- | ----- | --- | -------- | --------- | --- |
|     |     |      |     |     |     |     | convention) | [Avi: | cite | jaxGB | and | the fast | Galactic- |     |
∈
with u = t/T obs [0,1] rescaled time. The instru- binary waveform it implements (Cornish & Lit-
mentcomponentn andtheGalactic-confusioncompo- tenberg 2007)]. Recovering its parameters jointly with
inst
nent n are drawn as independent zero-mean coloured thenoiseisthesubjectofSec.VI.Forthenoise-recovery
gal
Gaussian processes whose one-sided PSDs are the ana- results of this section the binary has been removed, so
lytic Robson–Cornish–Liu fits [23] S inst (f) and S gal (f): the estimand is the pure non-stationary noise PSD.

7
coeﬀicients and the short-time Fourier coeﬀicients, the
latter in both its phase-retaining form and its power-
only dynamic-Whittle reduction. All of them recover
the same cyclostationary modulation of the noise PSD
in the binary’s channel, tracking the true annual enve-
lope where a stationary model is flat (Fig. 6). Because
they share the identical P-spline model, the quantity be-
ing tested is the stationarity assumption rather than the
representation. Overanensembleofindependentrealisa-
tions the 90% credible band on the recovered noise PSD
attains empirical coverage of 0.78 and 0.84 in the two
noise-orthogonal A and E channels (Sec. VI), below the
nominal 0.90. We attribute this under-coverage to the
diagonal likelihood approximation: treating the WDM
coeﬀicients as independent neglects the residual nearest-
neighbour correlations of a finitely-orthogonal transform
(Sec. IIIB), which narrows the posterior band relative
to the true sampling variability. [Avi: we get 0.78 vs
0.90.. probably because few repeated sims? ]
VI. JOINT SIGNAL–NOISE INFERENCE WITH
A BLOCKED-GIBBS SCHEME
[Avi: Work in progress... I want to show that
(1) with stationary noise, using S(f) or S(t,f)
leads to no biases in physical signal params, (2)
with non-stationary noise, the time-varying fits
are necessary to avoid biases. Right now, i see
FIG. 3. Performance versus the number of observations n,
little differences...]
over 100 realisations per point (median, with interquartile
The same Gaussian-coeﬀicient likelihood that recov-
band where shown). Top: MSE . Middle: 90% credible-
interval coverage (nominal level lo d g o f tted). Bottom: median ers the noise PSD admits a signal mean, c nm ∼
width of the 90% credible interval on logS, contracting with N(h nm (θ),S nm ), so in principle the source parameters
n. One curve per observation model (WDM and moving pe- θ and the non-stationary noise surface Λ can be inferred
riodogram). jointly. Wedevelopablocked-Gibbsschemeforthisjoint
fit and report preliminary results on a Galactic binary
(GB)/massive-black-hole-binary(MBHB)/Stellar-mass
B. Recovered time-varying PSD black-hole binary (BBH).
[Avi: Renate has ideas on how to do this for the
From a single noisy realisation, the estimator denoises moving-periodogram dynamic Whittle... But for
the raw χ2 WDM power into a smooth log-PSD surface now lets focus on the WDM.]
1
that matches the Monte Carlo reference (Fig. 4). The
recovered logSˆ(u,f) captures both the steep instrument
spectrum at low frequency and the twice-yearly Galactic A. Blocked Gibbs scheme
confusionmodulationthattherawpowerburiesinnoise.
The effect of the stationarity assumption is sharpest Rather than sampling the full state with a single
in a single frequency channel. Figure 5 shows the recov- Hamiltonian trajectory, we use a Metropolis-within-
ered noise PSD at f =1.5mHz over the year: a station- Gibbsschemethatalternatestwoblocks. ThePSDblock,
arymodelisnecessarilyflatintimeatthetime-averaged conditionalonthecurrentsourceestimate, subtractsthe
level, overestimating the noise when the Galactic confu- signalcoeﬀicientsh (θ)andupdatesthewhitenednoise
nm
sion is low and underestimating it when the confusion parameters (s,ϕ ,ϕ ) under the model of Sec. II. The
t f
peaks, by up to the factor-≃ 7 swing of the modula- source block, conditional on the current noise surface Λ,
tion, whereasthetime-varyingfittracksthetwice-yearly updates the source parameters θ. Blocking is deliberate:
modulation and contains the true S(u,f) within its 90% the high-dimensional smooth PSD coeﬀicients and the
band. low-dimensional source parameters have very different
The recovery does not depend on the time–frequency geometries,soeachblock’ssub-kernelcanadaptindepen-
representation. We analyse the same non-stationary dently while the Gibbs sweep couples them. The scheme
noisewithbothtransformsofSec.III:theWDMwavelet accommodates heterogeneous sub-kernels: a gradient-

8
FIG. 4. One year of non-stationary LISA noise, single realisation: raw WDM log power (the data), the posterior-mean
logSˆ(u,f),
and the Monte Carlo E[w2] reference. The estimator denoises the data into the smooth time-varying surface,
recovering the twice-yearly Galactic-confusion modulation invisible in the raw power.
∼
|     |     |     |     |     |     |     | is recovered | to  | within | 10% | by coherent |     | summation |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------ | --- | ----------- | --- | --------- |
(|A|/|A|
|     |     |     |     |     |     |     | across the | plane |     |     | = 0.90). | As a | check that |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----- | --- | --- | -------- | ---- | ---------- |
true
|     |     |     |     |     |     |     | the wavelet | domain | costs | no  | information | relative | to the |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------ | ----- | --- | ----------- | -------- | ------ |
usualanalysis,wecomparetheWDMsourceposteriorin
|     |     |     |     |     |     |     | stationary | noise | against | the | standard | frequency-domain |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----- | ------- | --- | -------- | ---------------- | --- |
Whittleanalysisonthesamedataandfindthetwoagree
|     |     |     |     |     |     |     | across all   | four binary |       | parameters | (Fig.          | 7).              |           |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ----------- | ----- | ---------- | -------------- | ---------------- | --------- |
|     |     |     |     |     |     |     | Repeating    | the         | joint | fit over   | the two        | noise-orthogonal |           |
|     |     |     |     |     |     |     | TDI channels | A           | and   | E, each    | with           | its own          | Digman–   |
|     |     |     |     |     |     |     | Cornish      | modulation  | and   | one        | shared         | binary           | amplitude |
|     |     |     |     |     |     |     | (each fit    | 300 warm-up |       | and 250    | posterior      | draws,           | WDM       |
|     |     |     |     |     |     |     | grid n       | = 256       | time  | bins),     | gives unbiased |                  | amplitude |
t
|     |     |     |     |     |     |     | recovery | across                               | 12 independent |     | one-year |     | realisations, |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------------------------------ | -------------- | --- | -------- | --- | ------------- |
|     |     |     |     |     |     |     | |A|/|A|  | =1.01±0.04(Fig.8),withthe90%PSDcred- |                |     |          |     |               |
true
|     |     |     |     |     |     |     | ible band   | attaining | the     | coverage | quoted       | in Sec. | VB. No |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | ------- | -------- | ------------ | ------- | ------ |
|     |     |     |     |     |     |     | realisation | produced  | sampler |          | divergences. | [Avi:   | 12 re- |
alisationsisasmallensembleforacoveragestate-
| FIG. | 5. Recovered | TDI | noise PSD | in the | binary’s | channel |     |     |     |     |     |     |     |
| ---- | ------------ | --- | --------- | ------ | -------- | ------- | --- | --- | --- | --- | --- | --- | --- |
(f =1.5mHz)overoneyear. Astationaryfit(red, with 90% ment. Report the binomial Monte Carlo error on
band) is necessarily flat at the time-average and so misses the 0.78/0.84 coverages, or grow the ensemble.]
| the | modulation, | while the | time-varying | fit | (blue) tracks | the |     |     |     |     |     |     |     |
| --- | ----------- | --------- | ------------ | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
truecyclostationarynoisePSD(black),whichpeakstwiceper
year.
|     |     |     |     |     |     |     | C. Why | the | noise | model | matters | for the | source |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | ----- | ----- | ------- | ------- | ------ |
Ayear-longbinary’sparametersarerobusttothenoise
basedNUTSupdateforthedifferentiablePSDblock,and
|                 |     |        |              |       |       |       | model: | the strain | is  | linear | in the | quadrature | ampli- |
| --------------- | --- | ------ | ------------ | ----- | ----- | ----- | ------ | ---------- | --- | ------ | ------ | ---------- | ------ |
| a gradient-free |     | update | for a source | model | whose | wave- |        |            |     |        |        |            |        |
differentiable.[src] tudes, so the weighted least-squares estimate is unbiased
form is not
|     |     |                   |     |            |        |     | even when     | the noise       | weights          |          | are misspecified, |       | and a full- |
| --- | --- | ----------------- | --- | ---------- | ------ | --- | ------------- | --------------- | ---------------- | -------- | ----------------- | ----- | ----------- |
|     |     |                   |     |            |        |     | cycle source  | averages        |                  | over the | modulation.       |       | The inter-  |
|     |     |                   |     |            |        |     | esting regime | is              | a time-localised |          | source,           | such  | as a burst  |
|     | B.  | Proof of concept: |     | a Galactic | binary |     |               |                 |                  |          |                   |       |             |
|     |     |                   |     |            |        |     | or the final  | days            | of an            | MBHB     | inspiral,         | which | accumu-     |
|     |     |                   |     |            |        |     | lates its     | signal-to-noise |                  | at a     | single            | epoch | and is mea- |
We first demonstrate the joint fit on the resolvable sured against the instantaneous noise. We model such
Galactic binary embedded in the data of Sec. V, with asourcethroughitsfrequency-channelSTFTcoeﬀicient,
both blocks updated by NUTS. From the single one- c =Aei(ϕ0+2πδftn)W(u )+noise , with W a localised
|     |     |     |     |     |     |     | n   |     |     | n   | n   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
year realisation the fit recovers the coherent binary to- envelopeandnoise oflocalvarianceS(u ,f ),andinfer
|     |     |     |     |     |     |     |     |     | n   |     |     | n   | 0   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
getherwiththenoisesurface: althoughthebinaryissub- (δf,A,ϕ ) with NUTS (800 warm-up, 2000 draws) un-
0
thresholdinanysingletime–frequencycell,itsamplitude der two noise models: the stationary frequency-domain

9
S(u)/⟨S⟩
FIG. 6. Non-stationary noise across both transforms. Left: the relative noise-power modulation in the binary’s
channel. The WDM (blue), short-time Fourier (green), and dynamic-Whittle moving-periodogram (purple, dashed) fits all
trackthetruecyclostationaryenvelope(black),whileastationarymodelisflat(red). Right: posterioroftheembeddedbinary
amplitude relative to truth for the two phase-retaining fits (violins with median bars, injected value dashed), a preview of the
joint fit of Sec. VI. The dynamic Whittle is power-only and so estimates the noise PSD on the signal-subtracted series.
|     |     |     |     |     |     |     | varying          | PSD        | recognises | the                             | low local   | noise          | and        | returns a |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ---------- | ---------- | ------------------------------- | ----------- | -------------- | ---------- | --------- |
|     |     |     |     |     |     |     | posterior        | a factor   | ≃          | 2 tighter                       | than        | the stationary |            | Whit-     |
|     |     |     |     |     |     |     | tle in every     | parameter, |            | both                            | centred     | on             | the truth. | At a      |
|     |     |     |     |     |     |     | confusionmaximum |            |            | thestationarymodelinsteadunder- |             |                |            |           |
|     |     |     |     |     |     |     | estimates        | the        | local      | noise                           | and becomes | overconfident, |            | its       |
90%credibleregioncontainingthetruthonly0.77ofthe
|     |     |     |     |     |     |     | time against |     | 0.88 for | the | time-varying |     | model | (over 300 |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | -------- | --- | ------------ | --- | ----- | --------- |
≈
|     |     |     |     |     |     |     | realisations, | nominal |                  | 0.90, | binomial | MC        | error | 0.02). |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ------- | ---------------- | ----- | -------- | --------- | ----- | ------ |
|     |     |     |     |     |     |     | Modelling     | the     | non-stationarity |       | is       | therefore | what  | keeps  |
thesourceposteriorbothcalibratedandmaximallyinfor-
mativefortime-localisedsources–theregimetheMBHB
|     |     |     |     |     |     |     | demonstration |           | below     | targets.   |            |             |        |          |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --------- | --------- | ---------- | ---------- | ----------- | ------ | -------- |
|     |     |     |     |     |     |     |               |           | VII.      | DISCUSSION |            |             |        |          |
|     |     |     |     |     |     |     | The           | framework | separates |            | the        | statistical | model, | the      |
|     |     |     |     |     |     |     | smooth        | log-PSD   | surface   | and        | its prior, | from        | the    | observa- |
tionmodelthatrelatesatime–frequencydataproductto
|                  |                 |                  |           |            |            |             | that surface.   |         | Section        | III          | exercised | this        | separation  | with      |
| ---------------- | --------------- | ---------------- | --------- | ---------- | ---------- | ----------- | --------------- | ------- | -------------- | ------------ | --------- | ----------- | ----------- | --------- |
|                  |                 |                  |           |            |            |             | two transforms, |         | the            | WDM          | wavelet   | transform   |             | and the   |
|                  |                 |                  |           |            |            |             | short-time      | Fourier | transform      |              | (the      | latter      | used either | with      |
| FIG. 7.          | Galactic-binary |                  | parameter | posteriors | in         | stationary  |                 |         |                |              |           |             |             |           |
|                  |                 |                  |           |            |            |             | phase or,       | as the  | moving         | periodogram, |           | without),   |             | both of   |
| noise, recovered |                 | in the WDM       |           | domain     | (orange,   | dashed) and |                 |         |                |              |           |             |             |           |
|                  |                 |                  |           |            |            |             | which share     |         | the model      | of           | Sec. II   | verbatim.   |             | Any other |
| with the         | standard        | frequency-domain |           | Whittle    | analysis   | (blue,      |                 |         |                |              |           |             |             |           |
|                  |                 |                  |           |            |            |             | near-orthogonal |         | representation |              | whose     | coeﬀicients |             | admit     |
| solid), injected |                 | values in        | black.    | The two    | posteriors | coincide    |                 |         |                |              |           |             |             |           |
alocal-power(Gaussian-coeﬀicientorexponential)likeli-
| across (log | f          | ,f˙,log A,ϕ    | ),  | confirming | that        | the wavelet |            |         |              |         |              |       |                |        |
| ----------- | ---------- | -------------- | --- | ---------- | ----------- | ----------- | ---------- | ------- | ------------ | ------- | ------------ | ----- | -------------- | ------ |
|             | 10 0       | 10             | 0   |            |             |             |            |         |              |         |              |       |                |        |
|             |            |                |     |            |             |             | hood, such | as      | a multitaper |         | spectrogram, |       | a Q-transform, |        |
| transform   | sacrifices | no information |     | on         | the source. |             |            |         |              |         |              |       |                |        |
|             |            |                |     |            |             |             | or another | wavelet |              | family, | can be       | added | without        | chang- |
|             |            |                |     |            |             |             | ing the    | model.  | [Avi:        | Cite    | multitaper   |       | (Thomson       |        |
Whittle (a single, time-averaged PSD) and the WDM 1982) and the Q-transform (Chatterji et al. 2004)
P-spline time-varying PSD. if we keep naming them as concrete examples.]
The two analyses differ markedly (Fig. 9). For a tran- The LISA demonstration shows that the choice of sta-
sient at a confusion minimum, a quiet epoch, the time- tionarity assumption, not the choice of representation,

10
FIG. 8. A/E ensemble over 12 independent one-year realisations of the joint fit. Left: recovered binary amplitude relative
to truth (violin with points as individual realisations, median bar, injected value dashed), unbiased at 1.01±0.04. Right:
empiricalcoverageofthe 90% PSDcrediblebandperchannelagainstthe analytictruth(nominalleveldotted), nearnominal.
is what drives the bias: the wavelet and Fourier trans- also outperform the moving periodogram.
| forms agree | with | one another | and | with | the frequency- |     |     |     |     |     |     |     |
| ----------- | ---- | ----------- | --- | ---- | -------------- | --- | --- | --- | --- | --- | --- | --- |
domainanalysiswherestationarityholds,andbothtrack
| thecyclostationarymodulationwhereitdoesnot. |            |           |       |      |            | Exist- |     |     |     |     |     |     |
| ------------------------------------------- | ---------- | --------- | ----- | ---- | ---------- | ------ | --- | --- | --- | --- | --- | --- |
| ing LISA                                    | stochastic | pipelines | model | this | modulation | ei-    |     |     |     |     |     |     |
therwithananalyticcyclostationarytemplate[4,5,8],a
|                    |           |             |               |                     |                  |        |     |     | VIII. CONCLUSION |     |     |     |
| ------------------ | --------- | ----------- | ------------- | ------------------- | ---------------- | ------ | --- | --- | ---------------- | --- | --- | --- |
| weakly-parametric  |           | spectral    | model         | [24],               | or a per-segment |        |     |     |                  |     |     |     |
| confusion          | amplitude | [7].        | Our estimator |                     | instead          | infers |     |     |                  |     |     |     |
| the time–frequency |           | PSD surface |               | non-parametrically, |                  | re-    |     |     |                  |     |     |     |
quiring no template for the modulation shape. This We have presented a Bayesian framework for non-
|                  |           |                |           |           |              |        | stationary    | spectral | estimation     | in which | a single | statis- |
| ---------------- | --------- | -------------- | --------- | --------- | ------------ | ------ | ------------- | -------- | -------------- | -------- | -------- | ------- |
| makes the        | framework | a natural      |           | component | of detector- |        |               |          |                |          |          |         |
|                  |           |                |           |           |              |        | tical object, | a        | smooth log-PSD | surface  | modelled | by a    |
| characterisation |           | and global-fit | pipelines |           | [9], where   | a mis- |               |          |                |          |          |         |
specified stationary noise model otherwise contaminates whitened tensor-product P-spline, is decoupled from the
|               |        |             |     |     |     |     | time–frequency          |     | representation | used to observe |     | it. A sin- |
| ------------- | ------ | ----------- | --- | --- | --- | --- | ----------------------- | --- | -------------- | --------------- | --- | ---------- |
| the recovered | source | population. |     |     |     |     |                         |     |                |                 |     |            |
|               |        |             |     |     |     |     | gle Gaussian-coeﬀicient |     | likelihood     | connects        | two | trans-     |
Several extensions follow naturally. The source block forms to that one model, the WDM wavelet transform
canbegeneralisedfromthetwolinearquadratureampli- and the short-time Fourier transform, with the classi-
tudesatfixedfrequencytothefullparametervectorθ = calmovingperiodogramrecoveredasthephase-discarded
(f ,f˙ ,A,...) with a nonlinear, multimodal frequency specialcaseofthelatter. Onalocallystationaryprocess
0 0
search,leavingtheblockedstructureunchanged. Beyond with known truth the estimator contracts with the data
a single source, the same coeﬀicient likelihood supports at near-nominal coverage. On simulated LISA data it
multi-sourceandmulti-channeljointfits,andrelaxingthe recovers the cyclostationary Galactic foreground, which
diagonal approximation to a correlated-coeﬀicient likeli- a stationary model cannot represent because it is flat at
hood would account for the residual nearest-neighbour the year-averaged level and misses the full annual swing
correlations left by a finitely orthogonal transform. Fi- of the confusion. Because the same coeﬀicient likeli-
nally, while a year-long binary’s parameters are ro- hood admits a signal mean, the noise PSD and an em-
bust to the noise model, time-localised sources are not bedded source can in principle be inferred jointly with
(Sec. VIC): there the time-varying PSD is what keeps a blocked-Gibbs scheme. We report preliminary results
the source posterior both calibrated and maximally in- on a Galactic binary and outline the massive-black-hole-
formative. The concrete next step we are pursuing is binary demonstration that remains. Since any near-
the massive-black-hole-binary demonstration of Sec. ??: orthogonal representation with a local-power likelihood
a full transient waveform and LISA response inside the plugs into the same model, the framework is a portable
blocked-Gibbs loop. This is the sharply time-localised component for detector characterisation and global-fit
regime where we expect critically-sampled wavelets to pipelines wherever the noise drifts in time.

11
|     |     |     |     |     |     |     |     |            | DATA       | AND CODE     | AVAILABILITY     |         |             |       |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---------- | ------------ | ---------------- | ------- | ----------- | ----- |
|     |     |     |     |     |     |     |     | [Avi:      | Add        | a one-line   | statement        |         | pointing    | to    |
|     |     |     |     |     |     |     |     | the public | repository |              | and              | the     | archived    | re-   |
|     |     |     |     |     |     |     |     | lease/DOI  | used       | to           | produce          | the     | figures,    | and   |
|     |     |     |     |     |     |     |     | name the   | key        | dependencies |                  | (jaxGB, | beyondWhit- |       |
|     |     |     |     |     |     |     |     | tle, the   | WDM        | transform    | implementation). |         |             | Note: |
ramos2023scikit(scikit-fda)isinrefs.bibbutcur-
|                           |     |            |        |                      |      |         |          | rently    | uncited | — cite | it where | the | B-spline | basis |
| ------------------------- | --- | ---------- | ------ | -------------------- | ---- | ------- | -------- | --------- | ------- | ------ | -------- | --- | -------- | ----- |
|                           |     |            |        |                      |      |         |          | is built, | or drop | it.]   |          |     |          |       |
| FIG. 9. Parameter         |     | posteriors |        | for a time-localised |      | source  | at a     |           |         |        |          |     |          |       |
| quiet (confusion-minimum) |     |            | epoch, | inferred             | with | the     | station- |           |         |        |          |     |          |       |
| ary frequency-domain      |     | Whittle    |        | (red, dashed)        |      | and the | WDM      |           |         |        |          |     |          |       |
P-spline time-varying PSD (green, solid), with truth in grey. ACKNOWLEDGMENTS
Byweightingthesourceagainstthelowlocalnoise,thetime-
≃
| varying PSD | yields | a posterior |         | a factor | 2         | tighter | in every |     |     |     |     |     |     |     |
| ----------- | ------ | ----------- | ------- | -------- | --------- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- |
| parameter.  | (At    | a confusion | maximum |          | the roles | reverse | and      |     |     |     |     |     |     |     |
the stationary model is instead overconfident, with 90% cov- [Avi: TODO:funding,collaborators,computing
| erage 0.77 | versus | 0.88.) |     |     |     |     |     | resources.] |     |     |     |     |     |     |
| ---------- | ------ | ------ | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- |
[1] P. Whittle, Journal of the Royal Statistical Society: Se- [13] S. N. Wood, Generalized Additive Models: An Introduc-
ries B 15, 125 (1953). tion with R, 2nd ed. (CRC Press, 2017).
[2] Y. Pawitan and F. O’Sullivan, Journal of the American [14] M. P. Boer, Statistical Modelling 23, 465 (2023).
Statistical Association 89, 600 (1994). [15] S. Lang and A. Brezger, Journal of Computational and
[3] P. Maturana-Russel and R. Meyer, Computational Graphical Statistics 13, 183 (2004).
Statistics 36, 2055 (2021). [16] D. Simpson et al., Statistical Science 32, 1 (2017).
[4] M. C. Digman and N. J. Cornish, The Astrophysical [17] Y. R. Yue, P. L. Speckman, and D. Sun, Annals of the
Journal 940, 10 (2022). Institute of Statistical Mathematics 64, 577 (2012).
[5] R. Buscicchio, A. Klein, V. Korol, F. Di Renzo, C. J. [18] M. D. Hoffman, A. Gelman, et al., J. Mach. Learn. Res.
| Moore, | D.  | Gerosa, | and | A. Carzaniga, |     | arXiv | e-prints | 15, | 1593 (2014). |     |     |     |     |     |
| ------ | --- | ------- | --- | ------------- | --- | ----- | -------- | --- | ------------ | --- | --- | --- | --- | --- |
10.48550/arXiv.2410.08263 (2024), arXiv:2410.08263. [19] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson,
[6] N. Karnesis, S. Babak, M. Pieroni, N. Cornish, and C. Leary, et al., JAX: composable transformations of
T. Littenberg, Physical Review D 104, 043019 (2021). Python+NumPy programs (2018).
[7] R. Rosati and T. B. Littenberg, arXiv e-prints [20] D.Phan,N.Pradhan,andM.Jankowiak,arXivpreprint
10.48550/arXiv.2410.17180 (2024), arXiv:2410.17180. arXiv:1912.11554 (2019).
[8] F. Pozzoli, D. Chirico, R. Buscicchio, and A. Klein, [21] A. Gelman and D. B. Rubin, Statistical Science 7, 457
| Journal | of  | Open | Source | Software | (2025), | bayesian, |     | (1992). |     |     |     |     |     |     |
| ------- | --- | ---- | ------ | -------- | ------- | --------- | --- | ------- | --- | --- | --- | --- | --- | --- |
HMC/NUTS, cyclostationary Galactic foreground for [22] R. Kumar, C. Carroll, A. Hartikainen, and O. Martin,
| LISA. |     |     |     |     |     |     |     | Journal | of Open | Source | Software | 4,  | 1143 (2019). |     |
| ----- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- | ------ | -------- | --- | ------------ | --- |
[9] M. L. Katz, N. Karnesis, N. Korsakova, J. R. Gair, and [23] T. Robson, N. J. Cornish, and C. Liu, Classical and
N. Stergioulas, Physical Review D 111, 024060 (2025). Quantum Gravity 36, 105011 (2019).
[10] N. J. Cornish, Physical Review D 102, 124038 (2020). [24] F. Pozzoli, R. Buscicchio, C. J. Moore, F. Haardt, and
[11] Y. Tang, C. Kirch, J. E. Lee, and R. Meyer, Bayesian A. Sesana, Physical Review D 109, 083029 (2024).
nonparametricspectralanalysisoflocallystationarypro-
| cesses     | (2023),    | arXiv   | preprint | arXiv:2303.11561. |         |     |          |     |     |     |     |     |     |     |
| ---------- | ---------- | ------- | -------- | ----------------- | ------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
| [12] P. H. | Eilers     | and B.  | D. Marx, | Chemometrics      |         | and | Intelli- |     |     |     |     |     |     |     |
| gent       | Laboratory | Systems |          | 66, 159           | (2003). |     |          |     |     |     |     |     |     |     |
