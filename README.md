## TREAT


For my project, I am interested in using simulations to determine the bias associated with estimating the effects of treatment with fixed effect models under situations where the treatment is staggered through time. A very simple example of this in practice would be the estimation of the effect of a social program (for example, enhanced unemployment insurance benefits) on the unemployment rate by state, where these enhanced benefits are passed into law gradually state by state. A typical fixed-effect regression approach would take the form:

$$ UnemploymentRate = \beta_0 + \bm{\beta_{1 ... r}^T}\textbf{DemographicCovariates} + \alpha_s + \lambda_t + \delta EnhancedUI + \epsilon_st $$ 

where the treatment effect of EnhancedUI is captured in $\hat{\delta}$. Only recently, [as of Boruysak et al's August 2021 discovery](https://arxiv.org/pdf/2108.12419.pdf), has it been discovered that in such cases $\delta$ are asymptotically biased, weighting observations that have been treated earlier with higher weight than those treated later - effectively the weights correspond to the length of treatment, and thus the derived value of $\delta$ is a function of the study duration. 

My goal is to use simulation across a range of parameters, to investigate whether the estimation of $\delta$ is biased or unbiased.

Case A) when treament occurs _simultaneously_  and:

* treatment effect is homogenous (that is, the effect is the same across all states/locations and $\delta_{st} \sim N(\delta, \sigma_{\delta})$). If this is _not_ the case asymptotically, something has gone very wrong. 
* treatment effect is _heterogenous_ across state but not time ($\delta_{st} \sim N(\delta_s, \sigma_{\delta})$)
* treatment effect is _heterogenous_  across time but not state ($\delta_{st} \sim N(\delta_t, \sigma_{\delta})$)
* treatment effect is _heterogenous_  across _both time and state_ ($\delta_{st} \sim N(\delta_{st}, \sigma_{\delta})$)

Case B) when treament is _staggered_  and the same conditions apply. 

Case C, D, E, F.....) possibilities are wide-ranging, but could change the size of the treatment effect, could change the variance of the random draw of treatment effects, could introduce demographic aspects or transition of population from region/state to others. 

_Brief note: I had planned to work on this project with a few collaborators until this evening, Nov 10, when we dissolved our group due to general disagreement on a topic. I plan to work out the details of this project both with MACS30123 staff TA's and Jon, as well as with Jeff Grogger, my Harris professor of Program Evaluation who has already told me that this would be useful science. In particular, I recognize that the large-scale components these test may conflict with the simplicity necessary for the results to be generalizable and I plan to speak with Prof. Grogger again to see what parameters of interest would be useful to investigate that could also increase the complexity a bit without sacrificing its usefulness. Also, I want to be sure I'm not investigating things that introduce bias simply due to model misspecification._

My hope is that I can highlight this phenomenon to help program evaluators and policy makers avoid using biased estimates of program effects and thus promote evidence-based policy making in my own small way. Further, I plan to use these simulated data to test an imputation method that the authors of the Borusyak paper propose, potentially giving weight to their new methodology or providing counter-examples for when their method fails to fix issues of bias. 