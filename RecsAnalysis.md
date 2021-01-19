RECS Analysis
================
Arfa Aijazi
12/22/2020

# The Contribution of Building Characteristics on Heat and Cold-Related Morbidity: Evidence from Household Surveys in the United States

## Pre-Processing

### Load libraries

``` r
library(tidyverse)
library(knitr)
library(kableExtra)
library(caret)
library(randomForest)
library(doParallel)
library(ggsci)
library(Hmisc)
library(PerformanceAnalytics)
```

### Load 2015 RECS microdata from EIA

``` r
RECS_2015 <- read_csv("https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv")
```

### Explore data distribution of output classes

![](RecsAnalysis_files/figure-gfm/unnamed-chunk-3-1.png)<!-- --> Medical
assistance needed because of thermal issues is a rare event (phew\!),
\<2% of all homes in the RECS 2015 microdata. More homes were too cold
than too hot and some homes had both types of thermal issues. In
classification ML problems, imbalance in observed classes can have a
negative impact on model fitting. Approaches to mitigate this issue are
to sub-sample the training data by *down-sampling*: subsets all classes
in the training set so that their class frequencies match the least
prevalent class or *up-sampling*: randomly sample (with replacement) the
minority class to be the same size as the majority class. I will test
the impact of both of these approaches on the predictive power of the ML
model [caret
documentation](https://topepo.github.io/caret/subsampling-for-class-imbalances.html).

Governments usually plan for heat and cold-related vulnerability
separately, so I will develop two separate sets of models to predict
whether a home is *too hot* or *too cold*, referred to as overheat and
overcool. The RECS 2015 encodes “too hot” issues in the variable HOTMA
and “too cold” issues in the variable COLDMA. In the overheat model
HOTMA is the dependent variable and in the overcool model COLDMA is the
dependent variable. Both dependent variables have two classes, they can
either be “yes” or “no”. The independent variables in both models are a
combination of climate, demographic, and building stock features. The
independent variables for both types of models is kept the same for
simplicity, but I will compare the relative importance of different
variables in predicting the different thermal issues.

### Filter and recode RECS 2015 data

``` r
thermalResilience <- RECS_2015 %>%
  select(CDD30YR, HDD30YR, DBT99, DBT1, NHSLDMEM, SDESCENT, HOUSEHOLDER_RACE, EDUCATION, EMPLOYHH, MONEYPY, HHAGE, UATYP10, KOWNRENT, ELPAY, NGPAY, LPGPAY, FOPAY, YEARMADERANGE, AIRCOND, WALLTYPE, ROOFTYPE, TYPEGLASS, WINFRAME, ADQINSUL, DRAFTY, TYPEHUQ, COOLTYPE, SWAMPCOL, HEATHOME, EQUIPM, NOHEATBROKE, NOHEATEL, NOHEATNG, NOHEATBULK, NOACBROKE, NOACEL, WINDOWS, NUMCFAN, NUMFLOORFAN, HOTMA, COLDMA) %>%
  mutate(UATYP10 = case_when(UATYP10 == "R" ~ 0,
                             UATYP10 == "C" ~ 1, 
                             UATYP10 == "U" ~ 2))

glimpse(thermalResilience)
```

    ## Rows: 5,686
    ## Columns: 41
    ## $ CDD30YR          <dbl> 1332, 2494, 2059, 1327, 871, 396, 2546, 597, 820, ...
    ## $ HDD30YR          <dbl> 2640, 2178, 2714, 4205, 5397, 7224, 1795, 7191, 55...
    ## $ DBT99            <dbl> 33.0, 31.7, 24.9, 15.9, 15.5, 0.4, 34.6, -1.6, 15....
    ## $ DBT1             <dbl> 97.9, 96.8, 92.5, 91.3, 88.4, 84.5, 86.2, 86.6, 88...
    ## $ NHSLDMEM         <dbl> 4, 2, 4, 1, 3, 1, 1, 2, 4, 4, 1, 1, 1, 2, 5, 5, 2,...
    ## $ SDESCENT         <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ HOUSEHOLDER_RACE <dbl> 1, 1, 1, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,...
    ## $ EDUCATION        <dbl> 2, 2, 1, 4, 2, 1, 5, 2, 3, 4, 3, 3, 2, 4, 5, 4, 3,...
    ## $ EMPLOYHH         <dbl> 1, 2, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0,...
    ## $ MONEYPY          <dbl> 8, 2, 2, 3, 3, 1, 4, 3, 5, 6, 3, 1, 2, 2, 6, 7, 2,...
    ## $ HHAGE            <dbl> 42, 60, 73, 69, 51, 33, 53, 67, 56, 48, 64, 24, 83...
    ## $ UATYP10          <dbl> 2, 0, 2, 1, 2, 1, 2, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2,...
    ## $ KOWNRENT         <dbl> 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1,...
    ## $ ELPAY            <dbl> 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1,...
    ## $ NGPAY            <dbl> 1, -2, 1, -2, 1, -2, 1, 1, -2, -2, -2, 1, 1, 2, 1,...
    ## $ LPGPAY           <dbl> -2, -2, -2, 1, -2, 2, -2, -2, -2, -2, -2, -2, -2, ...
    ## $ FOPAY            <dbl> -2, -2, -2, -2, -2, -2, -2, -2, 1, -2, 1, -2, -2, ...
    ## $ YEARMADERANGE    <dbl> 7, 5, 4, 2, 4, 5, 3, 4, 5, 2, 3, 1, 8, 4, 7, 8, 6,...
    ## $ AIRCOND          <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,...
    ## $ WALLTYPE         <dbl> 4, 3, 2, 3, 1, 3, 1, 3, 3, 1, 3, 4, 1, 1, 1, 4, 3,...
    ## $ ROOFTYPE         <dbl> 1, 3, 5, 5, 5, -2, 5, 5, 5, 5, 5, 5, 5, -2, 7, -2,...
    ## $ TYPEGLASS        <dbl> 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2,...
    ## $ WINFRAME         <dbl> 2, 2, 1, 2, 3, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 2,...
    ## $ ADQINSUL         <dbl> 2, 2, 2, 2, 2, 1, 3, 2, 2, 3, 2, 3, 3, 2, 1, 2, 1,...
    ## $ DRAFTY           <dbl> 4, 4, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 3,...
    ## $ TYPEHUQ          <dbl> 2, 2, 2, 2, 2, 5, 4, 2, 2, 2, 2, 2, 2, 5, 2, 5, 1,...
    ## $ COOLTYPE         <dbl> 1, 2, 3, 1, 1, 2, 1, 2, 1, 1, 2, -2, 1, 2, 1, 1, 1...
    ## $ SWAMPCOL         <dbl> 0, 0, 0, -2, -2, -2, 0, -2, -2, 0, -2, -2, 0, -2, ...
    ## $ HEATHOME         <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,...
    ## $ EQUIPM           <dbl> 3, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3,...
    ## $ NOHEATBROKE      <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,...
    ## $ NOHEATEL         <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ NOHEATNG         <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ NOHEATBULK       <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ NOACBROKE        <dbl> 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ NOACEL           <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ WINDOWS          <dbl> 41, 20, 41, 42, 30, 20, 30, 30, 41, 42, 41, 30, 41...
    ## $ NUMCFAN          <dbl> 5, 3, 7, 9, 2, 0, 2, 3, 2, 4, 1, 2, 5, 0, 2, 4, 4,...
    ## $ NUMFLOORFAN      <dbl> 0, 1, 0, 0, 2, 2, 0, 3, 0, 0, 2, 1, 0, 1, 2, 0, 0,...
    ## $ HOTMA            <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ COLDMA           <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...

### Identifying Zero and Near Zero-Variance Predictors

Constant or almost constant predictors across samples, called zero or
near zero-variance predictors, can cause some machine learning model
types to crash or produce unstable results. The concern is that near
zero-variance predictors may become zero-variance predictors when the
data is split into sub-samples for training, testing, and
cross-validation. This step flags zero or near-zero variance predictors,
by computing two metrics for the subsetted RECS 2015 data. The frequency
ratio (freqRatio) is the frequency of the most prevalent value to the
second most frequent value. This value will be near 1 for well-behaved
predictors and very large for highly-unbalanced data. The percent of
unique values (percentUnique) is the number of unique values divided by
the total number of samples (times 100). This value approaches zero for
highly granular data. The default threshold for near zero-variance is a
frequency ratio greater than 19 or a percentage of unique values less
than 10.

<table class="table" style="margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

Variable

</th>

<th style="text-align:right;">

freqRatio

</th>

<th style="text-align:right;">

percentUnique

</th>

<th style="text-align:left;">

zeroVar

</th>

<th style="text-align:left;">

nzv

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

NOHEATNG

</td>

<td style="text-align:right;">

209.59259

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

NOHEATBULK

</td>

<td style="text-align:right;">

148.63158

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

HOTMA

</td>

<td style="text-align:right;">

144.79487

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

COLDMA

</td>

<td style="text-align:right;">

104.29630

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

NOACEL

</td>

<td style="text-align:right;">

92.21311

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

NOHEATEL

</td>

<td style="text-align:right;">

76.89041

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

NOHEATBROKE

</td>

<td style="text-align:right;">

35.68387

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

ELPAY

</td>

<td style="text-align:right;">

22.14050

</td>

<td style="text-align:right;">

0.0703482

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

HEATHOME

</td>

<td style="text-align:right;">

21.03876

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

FOPAY

</td>

<td style="text-align:right;">

20.58015

</td>

<td style="text-align:right;">

0.0879353

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

<tr>

<td style="text-align:left;">

NOACBROKE

</td>

<td style="text-align:right;">

20.37594

</td>

<td style="text-align:right;">

0.0351741

</td>

<td style="text-align:left;">

FALSE

</td>

<td style="text-align:left;">

TRUE

</td>

</tr>

</tbody>

</table>

The table above show that in addition to the 2 target variables, 9 input
variables have near zero-variance. These are primarily related to the
function and operation of heating and cooling systems or who pays for
energy. Several of the near zero-variance predictors have frequency
ratios close to 19. Namely ELPAY, HEATHOME, FOPAY, and NOACBROKE.
However, others predictors have frequency ratios an order of magnitude
higher, such as NOHEATNG and NOHEATBULK.

### Identifying Correlated Predictors

Some machine learning model types thrive on correlated predictors and
others benefit from reducing the level of correlation between
predictors. This exercise is to understand the degree of correlation
between the selected variables.

``` r
# ++++++++++++++++++++++++++++
# flattenCorrMatrix
# ++++++++++++++++++++++++++++
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor = (cormat)[ut],
    p = pmat[ut]
    )
}
```

``` r
temp <- rcorr(as.matrix(thermalResilience)) 
highCorr <- flattenCorrMatrix(temp$r, temp$P) %>%
  mutate(cor = round(cor, 2)) %>%
  mutate(p = round(p, 2)) %>%
  arrange(desc(abs(cor)), desc(p)) %>%
  filter(abs(cor) > 0.5) %>% # criteria for correlation
  rename(Variable1 = row, Variable2 = column)
  
  
  highCorr_kbl <- highCorr %>%
  select(-p) %>%
  kbl() %>%
  kable_styling()

highCorr_kbl
```

<table class="table" style="margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

Variable1

</th>

<th style="text-align:left;">

Variable2

</th>

<th style="text-align:right;">

cor

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

HDD30YR

</td>

<td style="text-align:left;">

DBT99

</td>

<td style="text-align:right;">

\-0.94

</td>

</tr>

<tr>

<td style="text-align:left;">

AIRCOND

</td>

<td style="text-align:left;">

COOLTYPE

</td>

<td style="text-align:right;">

0.90

</td>

</tr>

<tr>

<td style="text-align:left;">

CDD30YR

</td>

<td style="text-align:left;">

HDD30YR

</td>

<td style="text-align:right;">

\-0.76

</td>

</tr>

<tr>

<td style="text-align:left;">

ROOFTYPE

</td>

<td style="text-align:left;">

TYPEHUQ

</td>

<td style="text-align:right;">

\-0.76

</td>

</tr>

<tr>

<td style="text-align:left;">

DBT99

</td>

<td style="text-align:left;">

SWAMPCOL

</td>

<td style="text-align:right;">

0.72

</td>

</tr>

<tr>

<td style="text-align:left;">

HDD30YR

</td>

<td style="text-align:left;">

SWAMPCOL

</td>

<td style="text-align:right;">

\-0.69

</td>

</tr>

<tr>

<td style="text-align:left;">

CDD30YR

</td>

<td style="text-align:left;">

DBT1

</td>

<td style="text-align:right;">

0.67

</td>

</tr>

<tr>

<td style="text-align:left;">

TYPEHUQ

</td>

<td style="text-align:left;">

WINDOWS

</td>

<td style="text-align:right;">

\-0.62

</td>

</tr>

<tr>

<td style="text-align:left;">

CDD30YR

</td>

<td style="text-align:left;">

DBT99

</td>

<td style="text-align:right;">

0.60

</td>

</tr>

<tr>

<td style="text-align:left;">

KOWNRENT

</td>

<td style="text-align:left;">

TYPEHUQ

</td>

<td style="text-align:right;">

0.58

</td>

</tr>

<tr>

<td style="text-align:left;">

NOHEATEL

</td>

<td style="text-align:left;">

NOACEL

</td>

<td style="text-align:right;">

0.53

</td>

</tr>

<tr>

<td style="text-align:left;">

HEATHOME

</td>

<td style="text-align:left;">

EQUIPM

</td>

<td style="text-align:right;">

0.51

</td>

</tr>

<tr>

<td style="text-align:left;">

ROOFTYPE

</td>

<td style="text-align:left;">

WINDOWS

</td>

<td style="text-align:right;">

0.51

</td>

</tr>

</tbody>

</table>

I considered absolute correlation greater than 0.5 as the threshold for
a linear relationship. For variable combinations that met this criteria,
I produced scatterplots to confirm if there was indeed a relationship as
the correlation coefficient can be a misleading metric.

``` r
for (i in 1:nrow(highCorr)){
  plot_data <- thermalResilience %>%
    select(c(highCorr$Variable1[i], highCorr$Variable2[i]))
  plot(plot_data)
}
```

![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-3.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-4.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-5.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-6.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-7.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-8.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-9.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-10.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-11.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-12.png)<!-- -->![](RecsAnalysis_files/figure-gfm/unnamed-chunk-8-13.png)<!-- -->
The scatter plots show correlations between the climatic variables,
DBT1, DBT99, HDD30YR, and CDD30YR, which makes intuitive sense. There is
also a correlation between variables describing the presence of a
heating or cooling system (HEATHOM and AIRCOND) and variables describing
the type of heating and cooling system (EQUIPM and COOLTYPE).

### Linear Dependencies

``` r
comboInfo <- findLinearCombos(thermalResilience)
comboInfo
```

    ## $linearCombos
    ## list()
    ## 
    ## $remove
    ## NULL

Based on analysis of zero and near-zero variance predictors, linear
correlations, and linear dependencies, I will build the machine learning
model with one climate variable. Typical HVI use outdoor surface
temperature, but the RECS survey data only includes HDD, CDD, 99% design
temperature, 1% design temperature. The linear correlation analysis
showed that degree days was highly correlated to design temperature, so
I will move forward with HDD and CDD to represent climate as that metric
captures both the intensity (difference between outdoor temperature and
base temperature) and duration (cummulative) of temperatures requiring
active systems. The linear correlation analysis also found redundancy in
including a variable for the presence of an active system and the type
of active system. Since the variable for the type of active system also
captures the lackthereof.

Prepare: overheat

``` r
overheat <- thermalResilience %>%
  select(CDD30YR, NHSLDMEM, SDESCENT, HOUSEHOLDER_RACE, EDUCATION, EMPLOYHH, MONEYPY, HHAGE, UATYP10, KOWNRENT, ELPAY, NGPAY, LPGPAY, FOPAY, YEARMADERANGE, WALLTYPE, ROOFTYPE, TYPEGLASS, WINFRAME, ADQINSUL, DRAFTY, TYPEHUQ, COOLTYPE, SWAMPCOL, NOACBROKE, NOACEL, WINDOWS, NUMCFAN, NUMFLOORFAN, HOTMA) %>%
  mutate(HOTMA = ifelse(HOTMA == 1, "Yes", "No")) %>%
  mutate(HOTMA = factor(HOTMA, levels = c("Yes", "No")))
```

Prepare: overcool

``` r
overcool <- thermalResilience %>%
  select(HDD30YR, NHSLDMEM, SDESCENT, HOUSEHOLDER_RACE, EDUCATION, EMPLOYHH, MONEYPY, HHAGE, UATYP10, KOWNRENT, ELPAY, NGPAY, LPGPAY, FOPAY, YEARMADERANGE, WALLTYPE, ROOFTYPE, TYPEGLASS, WINFRAME, ADQINSUL, DRAFTY, TYPEHUQ, EQUIPM, NOHEATBROKE, NOHEATEL, NOHEATNG, NOHEATBULK, WINDOWS, COLDMA) %>%
  mutate(COLDMA = ifelse(COLDMA == 1, "Yes", "No")) %>%
  mutate(COLDMA = factor(COLDMA, levels = c("Yes", "No")))
```

## Data Spliting

Partition data into an 80/20% split of the RECS 2015 microdata. I
created separate data partitions of the overheat and overcool model so
that the training and test data sets of each model preserves the overall
distribution of the thermal issue of interest.

``` r
set.seed(789) #to ensure reproducible results
hotPartition <- createDataPartition(overheat$HOTMA, p = 0.8, list = FALSE)
coldPartition <- createDataPartition(overcool$COLDMA, p = 0.8, list = FALSE)

overheat_train <- overheat[hotPartition,] 
```

    ## Warning: The `i` argument of ``[`()` can't be a matrix as of tibble 3.0.0.
    ## Convert to a vector.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_warnings()` to see where this warning was generated.

``` r
overheat_test <- overheat[-hotPartition,] 

overcool_train <- overcool[coldPartition,] 
overcool_test <- overcool[-coldPartition,]
```

## Cross-validation parameters

10-fold cross-validation repeated 10 times. Due to class imbalance in
the underlying RECS 2015 data, I will also test model performance with
down-sampling and up-sampling, which are both completed as part of
cross-validation sub-sampling.

## Model Performance

Since thermal issues are rare events in the RECS 2015 microdata, the
overall accuracy is not a meaningful metric as a model that
misclassifies all overheating events would still have an accuracy of
99.5%. Balanced accuracy is a more useful metric for evaluating a binary
classifier, particularly when there are class imbalances, as is the case
with this data set. The balanced accuracy is the average of the
sensitivity, accuracy of detecting “positive” cases, and specificity,
accuracy of detecting “negative” cases. I will also track sensitivity
separately, since I am specifically interested in model performance in
predicting “positive” cases i.e. cases of thermal issues.

## Model Training

I will compare performance of overheat and overcool models developed
from different ML algorithms that are suited for two-category
classification problems: logistic regression, k-nearest neighbors,
random forests, and support vector machines.

Define model iterations

``` r
thermal <- c("overheat", "overcool")
mlMethods <- c("glm", "knn", "rf", "svmLinear", "svmRadial")
trainSubsample <- c("none", "down", "up")
modelPerformance <- matrix(NA, nrow = 0, ncol = 5)
colnames(modelPerformance) <- c("MLMethods", "SubSample", "Thermal", "Sensitivity", "BalancedAccuracy")
variableImportance <- matrix(NA, nrow = 0, ncol = 5)
colnames(variableImportance) <- c("MLMethods", "SubSample", "Thermal", "Variable", "Overall")
```

Parallel processing

``` r
cl <- makePSOCKcluster(5)
registerDoParallel(cl)
```

Train and test model performance

Plot model performance Clean up outputs for plotting

Plot model performance

Balanced accuracy evaluates the overall performance of the machine
learning model by ave raging sensitivity and specificity. A balanced
accuracy of 50% (reference line) is analogous to random chance,
i.e. using a coin flip to predict the outcome. As with sensitivity, we
see that none of the ML model types evaluated performed well when I did
not sub-sample the training data set. Sub-sampling the training data

Sensitivity answers the question, how many of the “positive”,
i.e. overheating or overcooling cases did the model correctly detect?
The results show, that none of the ML models correctly identified
positive cases when I did not sub-sample the training data set.
Down-sampling improved performance of all model types, and up-sampling
improved performance of glm and svmLinear models.

With the selected input variables, model performance did not vary
significantly based on whether the model was predicting overheating or
overcooling.

The three top performing models have identical AUC
