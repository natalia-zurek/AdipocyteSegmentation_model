# Title:    Adipocyte analysis
# Author:   Natalia Zurek : natalia.zurek@cshs.org
# Project:  Ovarian cancer Computational Pathology

rm(list = ls())
source('C:/_research_projects/Ovarian cancer project/R project/Ovarian cancer/Statistical-analysis-code/functions/statistical_tests.R')
library(readxl)
library(dunn.test)
library(gdata)
library(reshape2)
library(tidyverse)
library(openxlsx)
library(effsize)
library(writexl)

# ========= MORPHOLOGICAL FEATURES ========== 
## ========= OM3  ========== 
### ========= correlation ========== 

data_M2Fraw <- read_excel('C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/OM3/adipocyte features/Adipocyte_ft_M2Fraw_Omental_mets_clinical.xlsx')
data_M2Fpost <- read_excel('C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/OM3/adipocyte features/Adipocyte_ft_M2Fpost_Omental_mets_clinical.xlsx')
data_DLVraw <- read_excel('C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/OM3/adipocyte features/Adipocyte_ft_DLVraw_Omental_mets_clinical.xlsx')
data_DLVpost <- read_excel('C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/OM3/adipocyte features/Adipocyte_ft_DLVpost_Omental_mets_clinical.xlsx')

filename = 'DLVpost_vs_clinical_sp_correlation'
AllData <- data_DLVpost
# data <- data_M2Fraw
# 
# clinical_data <- read.csv('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/Omental_clinical_data.csv', stringsAsFactors = TRUE, na.strings=c("nd", "ND", "not_defined", "", "na", "NaN", "NA"))%>%
#   as_tibble()
# 
# #delete patients that were used for training
# clinical_data <- clinical_data[!clinical_data$Patient_Deidentified_ID %in% c(896, 12245, 12261), ]
# 
# data <- data %>%
#   mutate(Patient_Deidentified_ID = as.numeric(sub("\\.svs$", "", `Slide name`)))
# 
# AllData <- merge(clinical_data, data, by = "Patient_Deidentified_ID")
# 
# write_xlsx(AllData, path = "C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/Adipocyte_ft_M2Fraw_Omental_mets_clinical.xlsx")

clinical_correlation <- as_tibble(AllData %>%
  select(Age_at_Diagnosis,
         BMI,
         Weight_Pounds,
         ))

features <- as_tibble(AllData[, 37:length(AllData)])


pth = file.path('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/spearman correlation')
dir.create(pth, recursive = TRUE)

spearman_correlation(clinical_correlation, features, pth, filename)
### ========= mwu-test ========== 

filename = 'DLVpost_early_survival'
AllData <- data_DLVpost

clinical_correlation <- as_tibble(AllData %>%
                                    select(Early.survival.status
                                    ))

features <- as_tibble(AllData[, 37:length(AllData)])

pth = file.path('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/early survival analysis mwu')
dir.create(pth, recursive = TRUE)

mann_whitney_u_test(clinical_correlation, features, pth, filename)

## ========= PLCO ==========
### ========= correlation ========== 
data_M2Fraw <- read_excel('C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/PLCO/adipocyte morphology features/Adipocyte_ft_M2Fraw_PLCO_clinical.xlsx')
data_M2Fpost <- read_excel('C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/PLCO/adipocyte morphology features/Adipocyte_ft_M2Fpost_PLCO_clinical.xlsx')

data_DLVraw <- read_excel('C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/PLCO/adipocyte morphology features/Adipocyte_ft_DLVraw_PLCO_clinical.xlsx')
data_DLVpost <- read_excel('C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/PLCO/adipocyte morphology features/Adipocyte_ft_DLVraw_PLCO_clinical.xlsx')

filename = 'DLVraw_vs_clinical_sp_'
AllData <- data_DLVraw

# clinical_data <- read.csv('C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/PLCO_clinical_data.csv', stringsAsFactors = TRUE, na.strings=c("nd", "ND", "not_defined", "", "na", "NaN", "NA"))%>%
#   as_tibble()
# names(clinical_data)[3]<-paste("Patient_ID")
# 
# AllData <- inner_join(clinical_data[, 3:ncol(clinical_data)], data, by = "Patient_ID")
# 
# # Keep only unique "Patient_ID" entries
# AllData <- AllData %>% distinct(Patient_ID, .keep_all = TRUE)
# write_xlsx(AllData, path = "C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/Adipocyte_ft_DLVpost_Omental_mets_clinical.xlsx")

clinical_correlation <- as_tibble(AllData %>%
                                    select(Age_at_diagnosis,
                                           bmi_curr,
                                           weight_f
                                    ))

features <- as_tibble(AllData[, 110:length(AllData)])
pth = file.path('C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/spearman correlation')
dir.create(pth, recursive = TRUE)

spearman_correlation(clinical_correlation, features, pth, filename)


### ========= mwu-test ========== 

filename = 'M2Fpost_early_survival'
AllData <- data_M2Fpost

clinical_correlation <- as_tibble(AllData %>%
                                    select(Early.survival.status
                                    ))

features <- as_tibble(AllData[, 110:length(AllData)])
pth = file.path('C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/early survival analysis mwu')
dir.create(pth, recursive = TRUE)
mann_whitney_u_test(clinical_correlation, features, pth, filename)


# ========= FRACTAL ANALYSIS ========== 
## ========= OM3 ========== 

AllData <- read.csv(paste("C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/OM3/fat tissue fractal features/","OM3_fractal_features_aggregated_clinical.csv", sep = ''), stringsAsFactors = TRUE, na.strings=c("nd", "ND", "U", "", "na", "NaN"))%>%
  print()

clinical_correlation <- as_tibble(AllData %>%
                                    select(Age_at_Diagnosis,
                                           BMI,
                                           Weight_Pounds,
                                    ))

features <- as_tibble(AllData[, 37:length(AllData)])

## ========= correlation ========== 

pth = 'C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/DeepLabV3plus/DeepLabV3plus fat tissue/sp correlation analysis'
dir.create(pth, recursive = TRUE)
filename = 'OM3_DLV_fat_tissue_vs_fractal'

spearman_correlation(clinical_correlation, features, pth, filename)
## ========= mwu-test ========== 
early_surv <- as_tibble(AllData %>%
                          select(Early.survival.status))

pth = 'C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/DeepLabV3plus/DeepLabV3plus fat tissue/early survival mwu'
dir.create(pth, recursive = TRUE)
filename = 'OM3_DLV_fat_tissue_vs_fractal_early_survival_mwu'
mann_whitney_u_test(early_surv, features, pth, filename)


# ========= PLCO ========== 
AllData <- read.csv(paste("C:/Users/wylezoln/OneDrive - Cedars-Sinai Health System/_my_projects/Adipocyte project/Statistical analysis/PLCO/fat tissue fractal features/","PLCO_fractal_features_aggregated_clinical.csv", sep = ''), stringsAsFactors = TRUE, na.strings=c("nd", "ND", "U", "", "na", "NaN"))%>%
  print()

clinical_correlation <- as_tibble(AllData %>%
                                    select(#Early.survival.status
                                      Age_at_diagnosis,
                                      bmi_curr,
                                      weight_f,
                                    ))

features <- as_tibble(AllData[, 112:length(AllData)])
## ========= correlation ========== 

pth = 'C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/DeepLabV3plus/DeepLabV3plus fat tissue/sp correlation analysis'
dir.create(pth, recursive = TRUE)
filename = 'PLCO_DLV_fat_tissue_vs_fractal'

spearman_correlation(clinical_correlation, features, pth, filename)
## ========= mwu-test ========== 
early_surv <- as_tibble(AllData %>%
                          select(Early.survival.status))

pth = 'C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/DeepLabV3plus/DeepLabV3plus fat tissue/early survival mwu'
dir.create(pth, recursive = TRUE)
filename = 'PLCO_DLV_fat_tissue_vs_fractal_early_survival_mwu'
mann_whitney_u_test(early_surv, features, pth, filename)
# ========= KM plot (median surv) ========== 

library(survival)
library(survminer)

data <- read_excel('C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/Adipocyte_ft_DLVpost_Omental_mets_clinical.xlsx')
mySurv <- Surv(time=data$Overall.Survival..Time.to.Death.or.to.last.survival.status.if.alive..months., event=data$Patient.status..1.dead..0.alive.)

data <- read_excel('C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/Adipocyte_ft_DLVpost_Omental_mets_clinical.xlsx')
mySurv <- Surv(time=data$Overall_Survival_calculated, event=data$is_dead)
survfit(mySurv ~ 1)
fit <- survfit(mySurv ~ 1, data = data)

# Plot with customization
ggsurvplot(fit,
           data = data,
           title = "PLCO Kaplan-Meier Survival Curve",         # Add title
           xlab = "Time (years)",                        # X-axis label
           ylab = "Survival Probability",                 # Y-axis label
           ggtheme = theme_minimal(),                     # Minimal theme for a clean look
           break.y.by = 0.1,                              # Set y-axis ticks 0.1 apart
           break.x.by = 1,
           risk.table = TRUE,                             # Add risk table below the plot
           conf.int = TRUE,                               # Show confidence intervals
           legend.title = "Survival Probability", 
           legend.labs = "Overall Survival")             # Legend label
